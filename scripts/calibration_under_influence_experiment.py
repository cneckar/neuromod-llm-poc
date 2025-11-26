#!/usr/bin/env python3
"""
Calibration Under the Influence Experiment

Tests the hypothesis that Over-Stimulation (Focus) leads to Overfitting Errors
(Brittleness/Overconfidence) by measuring calibration and OOD performance under
varying "dosages" of stimulant packs.

Hypothesis: Increasing "Stimulation" (via QK-Scaling and Temperature reduction)
will decrease entropy but increase Expected Calibration Error (ECE) and degrade
OOD Generalization.
"""

import argparse
import json
import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch
from tqdm import tqdm

# Import neuromodulation system
from neuromod import NeuromodTool
from neuromod.pack_system import PackRegistry
from neuromod.model_support import create_model_support

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class MMLUQuestion:
    """Represents a single MMLU question"""
    question: str
    choices: List[str]  # [A, B, C, D]
    correct_answer: str  # 'A', 'B', 'C', or 'D'
    subject: str


@dataclass
class CalibrationMetrics:
    """Calibration metrics for a model/pack combination"""
    ece: float  # Expected Calibration Error
    mce: float  # Maximum Calibration Error
    brier_score: float
    accuracy: float
    confidence: float
    num_samples: int
    calibration_bins: Dict[str, float]  # Bin-wise accuracy and confidence


@dataclass
class OODMetrics:
    """Out-of-distribution performance metrics"""
    perplexity: float
    unique_ngrams: int
    total_ngrams: int
    diversity_ratio: float
    repetition_rate: float
    avg_response_length: float


@dataclass
class ExperimentResult:
    """Complete experiment result for a model/pack combination"""
    model_name: str
    pack_name: str
    intensity: float
    calibration: CalibrationMetrics
    ood_metrics: OODMetrics
    generation_time: float
    timestamp: str


class MMLULoader:
    """Loads MMLU dataset for evaluation"""
    
    def __init__(self, data_dir: str = "datasets/mmlu"):
        self.data_dir = Path(data_dir)
        self.questions: List[MMLUQuestion] = []
    
    def load_from_hf(self, subset: str = "all", max_questions: Optional[int] = None):
        """Load MMLU from HuggingFace datasets"""
        try:
            from datasets import load_dataset
            
            logger.info(f"Loading MMLU dataset (subset: {subset})...")
            if subset == "all":
                # Load a representative subset of subjects
                subjects = ["abstract_algebra", "anatomy", "astronomy", "business_ethics",
                           "clinical_knowledge", "college_biology", "college_chemistry",
                           "college_computer_science", "college_mathematics", "college_physics",
                           "computer_security", "conceptual_physics", "econometrics",
                           "electrical_engineering", "elementary_mathematics", "formal_logic",
                           "global_facts", "high_school_biology", "high_school_chemistry",
                           "high_school_computer_science", "high_school_european_history",
                           "high_school_geography", "high_school_government_and_politics",
                           "high_school_macroeconomics", "high_school_mathematics",
                           "high_school_microeconomics", "high_school_physics",
                           "high_school_psychology", "high_school_statistics", "high_school_us_history",
                           "high_school_world_history", "human_aging", "human_sexuality",
                           "international_law", "jurisprudence", "logical_fallacies", "machine_learning",
                           "management", "marketing", "medical_genetics", "miscellaneous",
                           "moral_disputes", "moral_scenarios", "nutrition", "philosophy",
                           "prehistory", "professional_accounting", "professional_law",
                           "professional_medicine", "professional_psychology", "public_relations",
                           "security_studies", "sociology", "us_foreign_policy", "virology",
                           "world_religions"]
            else:
                subjects = [subset]
            
            dataset = load_dataset("cais/mmlu", "all", split="test")
            
            for item in dataset:
                if max_questions and len(self.questions) >= max_questions:
                    break
                
                question = MMLUQuestion(
                    question=item['question'],
                    choices=[item['A'], item['B'], item['C'], item['D']],
                    correct_answer=item['answer'],
                    subject=item.get('subject', 'unknown')
                )
                self.questions.append(question)
            
            logger.info(f"Loaded {len(self.questions)} MMLU questions")
            
        except ImportError:
            logger.warning("datasets library not available, using fallback loader")
            self._load_fallback(max_questions)
        except Exception as e:
            logger.error(f"Failed to load MMLU from HuggingFace: {e}")
            self._load_fallback(max_questions)
    
    def _load_fallback(self, max_questions: Optional[int] = None):
        """Fallback: create a small synthetic dataset for testing"""
        logger.warning("Using synthetic MMLU dataset (install 'datasets' for full dataset)")
        
        synthetic_questions = [
            MMLUQuestion(
                question="What is the capital of France?",
                choices=["London", "Berlin", "Paris", "Madrid"],
                correct_answer="C",
                subject="geography"
            ),
            MMLUQuestion(
                question="What is 2 + 2?",
                choices=["3", "4", "5", "6"],
                correct_answer="B",
                subject="mathematics"
            ),
            MMLUQuestion(
                question="Which planet is closest to the Sun?",
                choices=["Venus", "Mercury", "Earth", "Mars"],
                correct_answer="B",
                subject="astronomy"
            ),
        ]
        
        if max_questions:
            self.questions = synthetic_questions[:max_questions]
        else:
            self.questions = synthetic_questions


class CalibrationCalculator:
    """Calculates calibration metrics (ECE, MCE, Brier Score)"""
    
    def __init__(self, num_bins: int = 10):
        self.num_bins = num_bins
    
    def calculate_ece(self, confidences: np.ndarray, accuracies: np.ndarray) -> Tuple[float, Dict[str, float]]:
        """
        Calculate Expected Calibration Error (ECE)
        
        Args:
            confidences: Array of confidence scores (0-1)
            accuracies: Array of binary accuracy (0 or 1)
        
        Returns:
            ECE value and bin-wise statistics
        """
        bin_boundaries = np.linspace(0, 1, self.num_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        bin_stats = {}
        
        for i in range(self.num_bins):
            in_bin = (confidences > bin_lowers[i]) & (confidences <= bin_uppers[i])
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = accuracies[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                
                bin_stats[f"bin_{i}"] = {
                    "accuracy": float(accuracy_in_bin),
                    "confidence": float(avg_confidence_in_bin),
                    "count": int(in_bin.sum()),
                    "prop": float(prop_in_bin)
                }
            else:
                bin_stats[f"bin_{i}"] = {
                    "accuracy": 0.0,
                    "confidence": 0.0,
                    "count": 0,
                    "prop": 0.0
                }
        
        return float(ece), bin_stats
    
    def calculate_mce(self, confidences: np.ndarray, accuracies: np.ndarray) -> float:
        """Calculate Maximum Calibration Error (MCE)"""
        bin_boundaries = np.linspace(0, 1, self.num_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        mce = 0.0
        
        for i in range(self.num_bins):
            in_bin = (confidences > bin_lowers[i]) & (confidences <= bin_uppers[i])
            
            if in_bin.sum() > 0:
                accuracy_in_bin = accuracies[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                mce = max(mce, np.abs(avg_confidence_in_bin - accuracy_in_bin))
        
        return float(mce)
    
    def calculate_brier_score(self, confidences: np.ndarray, accuracies: np.ndarray) -> float:
        """Calculate Brier Score (lower is better)"""
        return float(np.mean((confidences - accuracies) ** 2))
    
    def calculate_metrics(self, confidences: np.ndarray, accuracies: np.ndarray) -> CalibrationMetrics:
        """Calculate all calibration metrics"""
        ece, bin_stats = self.calculate_ece(confidences, accuracies)
        mce = self.calculate_mce(confidences, accuracies)
        brier = self.calculate_brier_score(confidences, accuracies)
        
        return CalibrationMetrics(
            ece=ece,
            mce=mce,
            brier_score=brier,
            accuracy=float(accuracies.mean()),
            confidence=float(confidences.mean()),
            num_samples=len(confidences),
            calibration_bins=bin_stats
        )


class OODEvaluator:
    """Evaluates out-of-distribution performance"""
    
    def __init__(self):
        self.ood_prompts = [
            "Write a creative story about a robot who discovers emotions.",
            "Solve this riddle: I speak without a mouth and hear without ears. What am I?",
            "Describe an ethical dilemma where there is no clear right answer.",
            "Explain a complex concept using only simple words.",
            "Write a poem about the feeling of uncertainty.",
        ]
    
    def calculate_perplexity(self, model, tokenizer, text: str) -> float:
        """Calculate perplexity of generated text"""
        try:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
                perplexity = torch.exp(loss).item()
            
            return perplexity
        except Exception as e:
            logger.warning(f"Failed to calculate perplexity: {e}")
            return float('inf')
    
    def calculate_diversity(self, texts: List[str], n: int = 3) -> Tuple[int, int, float]:
        """Calculate n-gram diversity"""
        all_ngrams = set()
        total_ngrams = 0
        
        for text in texts:
            words = text.lower().split()
            for i in range(len(words) - n + 1):
                ngram = tuple(words[i:i+n])
                all_ngrams.add(ngram)
                total_ngrams += 1
        
        unique_ngrams = len(all_ngrams)
        diversity_ratio = unique_ngrams / total_ngrams if total_ngrams > 0 else 0.0
        
        return unique_ngrams, total_ngrams, diversity_ratio
    
    def calculate_repetition_rate(self, text: str) -> float:
        """Calculate repetition rate (proportion of repeated bigrams)"""
        words = text.lower().split()
        if len(words) < 2:
            return 0.0
        
        bigrams = [tuple(words[i:i+2]) for i in range(len(words) - 1)]
        unique_bigrams = set(bigrams)
        
        if len(bigrams) == 0:
            return 0.0
        
        return 1.0 - (len(unique_bigrams) / len(bigrams))
    
    def evaluate(self, model, tokenizer, neuromod_tool: Optional[NeuromodTool] = None) -> OODMetrics:
        """Evaluate OOD performance"""
        responses = []
        perplexities = []
        
        for prompt in self.ood_prompts:
            try:
                # Generate response
                inputs = tokenizer(prompt, return_tensors="pt")
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=150,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                responses.append(response)
                
                # Calculate perplexity
                full_text = prompt + " " + response
                perplexity = self.calculate_perplexity(model, tokenizer, full_text)
                perplexities.append(perplexity)
                
            except Exception as e:
                logger.warning(f"Failed to generate OOD response: {e}")
                responses.append("")
                perplexities.append(float('inf'))
        
        # Calculate metrics
        unique_ngrams, total_ngrams, diversity_ratio = self.calculate_diversity(responses)
        
        avg_repetition = np.mean([self.calculate_repetition_rate(r) for r in responses])
        avg_perplexity = np.mean([p for p in perplexities if p != float('inf')]) if perplexities else float('inf')
        avg_length = np.mean([len(r.split()) for r in responses])
        
        return OODMetrics(
            perplexity=avg_perplexity,
            unique_ngrams=unique_ngrams,
            total_ngrams=total_ngrams,
            diversity_ratio=diversity_ratio,
            repetition_rate=avg_repetition,
            avg_response_length=avg_length
        )


class CalibrationExperiment:
    """Main experiment runner"""
    
    def __init__(self, model_name: str, test_mode: bool = False):
        self.model_name = model_name
        self.test_mode = test_mode
        self.model_manager = None
        self.model = None
        self.tokenizer = None
        self.neuromod_tool = None
        self.registry = None
        
        # Evaluation components
        self.mmlu_loader = MMLULoader()
        self.calibration_calc = CalibrationCalculator()
        self.ood_evaluator = OODEvaluator()
    
    def setup(self):
        """Setup model and neuromodulation system"""
        logger.info(f"Setting up experiment with model: {self.model_name}")
        
        # Load model
        self.model_manager = create_model_support(test_mode=self.test_mode)
        self.model, self.tokenizer, self.model_info = self.model_manager.load_model(self.model_name)
        
        # Setup neuromodulation
        self.registry = PackRegistry('packs/config.json')
        self.neuromod_tool = NeuromodTool(self.registry, self.model, self.tokenizer)
        
        logger.info("Setup complete")
    
    def evaluate_mmlu_with_pack(self, pack_name: str, intensity: float = 1.0, 
                                max_questions: int = 100) -> CalibrationMetrics:
        """Evaluate MMLU and calculate calibration metrics"""
        logger.info(f"Evaluating MMLU with pack: {pack_name} (intensity: {intensity})")
        
        # Apply pack
        if pack_name != "none":
            result = self.neuromod_tool.apply(pack_name, intensity=intensity)
            if not result or not result.get('ok'):
                logger.warning(f"Failed to apply pack {pack_name}")
        
        # Load MMLU questions
        self.mmlu_loader.load_from_hf(max_questions=max_questions)
        
        confidences = []
        accuracies = []
        
        for question in tqdm(self.mmlu_loader.questions, desc="Evaluating MMLU"):
            try:
                # Format question as multiple choice
                prompt = f"{question.question}\n"
                for i, choice in enumerate(question.choices):
                    prompt += f"{chr(65+i)}. {choice}\n"
                prompt += "\nAnswer:"
                
                # Get logits for answer choices
                inputs = self.tokenizer(prompt, return_tensors="pt")
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits[0, -1, :]  # Last token logits
                
                # Get probabilities for A, B, C, D tokens
                answer_tokens = []
                for letter in ['A', 'B', 'C', 'D']:
                    token_id = self.tokenizer.encode(letter, add_special_tokens=False)
                    if token_id:
                        answer_tokens.append((letter, token_id[0]))
                
                if not answer_tokens:
                    # Fallback: try to find tokens that represent the letters
                    vocab = self.tokenizer.get_vocab()
                    for letter in ['A', 'B', 'C', 'D']:
                        if letter in vocab:
                            answer_tokens.append((letter, vocab[letter]))
                
                if not answer_tokens:
                    logger.warning("Could not find answer tokens, skipping question")
                    continue
                
                # Get probabilities
                probs = torch.softmax(logits, dim=-1)
                answer_probs = {letter: float(probs[token_id].item()) for letter, token_id in answer_tokens}
                
                # Get confidence (max probability)
                max_prob = max(answer_probs.values())
                predicted_answer = max(answer_probs, key=answer_probs.get)
                
                # Check accuracy
                is_correct = (predicted_answer == question.correct_answer)
                
                confidences.append(max_prob)
                accuracies.append(1.0 if is_correct else 0.0)
                
            except Exception as e:
                logger.warning(f"Error evaluating question: {e}")
                continue
        
        if len(confidences) == 0:
            logger.error("No valid evaluations completed")
            return CalibrationMetrics(
                ece=float('inf'), mce=float('inf'), brier_score=float('inf'),
                accuracy=0.0, confidence=0.0, num_samples=0, calibration_bins={}
            )
        
        # Calculate calibration metrics
        confidences = np.array(confidences)
        accuracies = np.array(accuracies)
        
        return self.calibration_calc.calculate_metrics(confidences, accuracies)
    
    def evaluate_ood_with_pack(self, pack_name: str, intensity: float = 1.0) -> OODMetrics:
        """Evaluate OOD performance with pack"""
        logger.info(f"Evaluating OOD with pack: {pack_name} (intensity: {intensity})")
        
        # Apply pack
        if pack_name != "none":
            result = self.neuromod_tool.apply(pack_name, intensity=intensity)
            if not result or not result.get('ok'):
                logger.warning(f"Failed to apply pack {pack_name}")
        
        return self.ood_evaluator.evaluate(self.model, self.tokenizer, self.neuromod_tool)
    
    def run_experiment(self, packs: List[str], intensities: List[float] = None,
                      max_mmlu_questions: int = 100) -> List[ExperimentResult]:
        """Run complete experiment"""
        if intensities is None:
            intensities = [1.0] * len(packs)
        
        results = []
        
        for pack_name, intensity in zip(packs, intensities):
            logger.info(f"\n{'='*60}")
            logger.info(f"Testing pack: {pack_name} (intensity: {intensity})")
            logger.info(f"{'='*60}\n")
            
            start_time = time.time()
            
            # Evaluate calibration
            calibration = self.evaluate_mmlu_with_pack(pack_name, intensity, max_mmlu_questions)
            
            # Clear pack before OOD evaluation
            self.neuromod_tool.clear()
            
            # Evaluate OOD
            ood_metrics = self.evaluate_ood_with_pack(pack_name, intensity)
            
            generation_time = time.time() - start_time
            
            result = ExperimentResult(
                model_name=self.model_name,
                pack_name=pack_name,
                intensity=intensity,
                calibration=calibration,
                ood_metrics=ood_metrics,
                generation_time=generation_time,
                timestamp=time.strftime("%Y%m%d_%H%M%S")
            )
            
            results.append(result)
            
            # Clear pack for next iteration
            self.neuromod_tool.clear()
        
        return results
    
    def save_results(self, results: List[ExperimentResult], output_dir: str = "outputs/calibration_experiments"):
        """Save experiment results"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Convert to JSON-serializable format
        results_dict = []
        for result in results:
            result_dict = {
                "model_name": result.model_name,
                "pack_name": result.pack_name,
                "intensity": result.intensity,
                "calibration": asdict(result.calibration),
                "ood_metrics": asdict(result.ood_metrics),
                "generation_time": result.generation_time,
                "timestamp": result.timestamp
            }
            results_dict.append(result_dict)
        
        filename = output_path / f"calibration_experiment_{results[0].timestamp}.json"
        with open(filename, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        logger.info(f"Results saved to {filename}")
        
        # Also save summary
        self._save_summary(results, output_path / f"calibration_summary_{results[0].timestamp}.txt")
    
    def _save_summary(self, results: List[ExperimentResult], filename: Path):
        """Save human-readable summary"""
        with open(filename, 'w') as f:
            f.write("Calibration Under the Influence - Experiment Summary\n")
            f.write("=" * 60 + "\n\n")
            
            for result in results:
                f.write(f"Pack: {result.pack_name} (Intensity: {result.intensity})\n")
                f.write(f"  ECE: {result.calibration.ece:.4f}\n")
                f.write(f"  MCE: {result.calibration.mce:.4f}\n")
                f.write(f"  Brier Score: {result.calibration.brier_score:.4f}\n")
                f.write(f"  Accuracy: {result.calibration.accuracy:.4f}\n")
                f.write(f"  Avg Confidence: {result.calibration.confidence:.4f}\n")
                f.write(f"  OOD Perplexity: {result.ood_metrics.perplexity:.2f}\n")
                f.write(f"  OOD Diversity: {result.ood_metrics.diversity_ratio:.4f}\n")
                f.write(f"  OOD Repetition Rate: {result.ood_metrics.repetition_rate:.4f}\n")
                f.write("\n")


def main():
    parser = argparse.ArgumentParser(
        description="Calibration Under the Influence Experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This experiment tests the hypothesis that over-stimulation leads to overfitting errors
by measuring calibration and OOD performance under varying stimulant dosages.

Example usage:
  python scripts/calibration_under_influence_experiment.py \\
    --model meta-llama/Llama-3.1-8B-Instruct \\
    --packs none cocaine_10 cocaine_50 cocaine_100 \\
    --max-questions 50
        """
    )
    
    parser.add_argument('--model', type=str, required=True,
                       help='Model to evaluate (e.g., meta-llama/Llama-3.1-8B-Instruct)')
    parser.add_argument('--packs', nargs='+', 
                       default=['none', 'cocaine_10', 'cocaine_50', 'cocaine_100'],
                       help='Packs to test')
    parser.add_argument('--intensities', nargs='+', type=float, default=None,
                       help='Intensities for each pack (default: 1.0 for all)')
    parser.add_argument('--max-questions', type=int, default=100,
                       help='Maximum MMLU questions to evaluate')
    parser.add_argument('--test-mode', action='store_true',
                       help='Use test mode (smaller models)')
    parser.add_argument('--output-dir', type=str, default='outputs/calibration_experiments',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Run experiment
    experiment = CalibrationExperiment(args.model, test_mode=args.test_mode)
    experiment.setup()
    
    results = experiment.run_experiment(
        packs=args.packs,
        intensities=args.intensities,
        max_mmlu_questions=args.max_questions
    )
    
    experiment.save_results(results, args.output_dir)
    
    # Print summary
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    for result in results:
        print(f"\n{result.pack_name} (intensity: {result.intensity}):")
        print(f"  ECE: {result.calibration.ece:.4f}")
        print(f"  Accuracy: {result.calibration.accuracy:.4f}")
        print(f"  Avg Confidence: {result.calibration.confidence:.4f}")
        print(f"  OOD Diversity: {result.ood_metrics.diversity_ratio:.4f}")


if __name__ == "__main__":
    main()

