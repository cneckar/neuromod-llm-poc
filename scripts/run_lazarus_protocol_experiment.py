#!/usr/bin/env python3
"""
The "Lazarus" Protocol: Proving the Stimulant Paradox

This experiment validates that stimulant vectors are mechanically functional,
not just hitting a "ceiling effect" due to RLHF pre-tuning.

The Flaw: The claim that "Stimulants" failed because the model is "already on Adderall"
is a hypothesis, not proof. A failed experiment is just a failure until you manipulate
the variable bi-directionally.

The Fix: Prove the stimulant vectors (v_focus) actually function mechanically.

Protocol:
Step A: Induce a "coma" - Use Morphine pack (attention degradation via qk_score_scaling down)
        to degrade attention until retrieval performance drops significantly.
Step B: Apply Cocaine (Stimulant) pack to "resuscitate" the sedated model.
Step C: Measure recovery - If stimulant vectors work, they should restore performance.

The Prediction: If vectors are mathematically valid, the Stimulant should restore
performance by sharpening the attention distribution artificially flattened by Morphine.

The Result: If it works, you haven't just made a "drug"; you have created a
differentiable attention control mechanism. That is publishable.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple
import argparse
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from neuromod.neuromod_factory import create_neuromod_tool
from neuromod.pack_system import PackRegistry
from neuromod.model_support import create_model_support
import torch


class RetrievalTask:
    """
    A retrieval task that requires the model to remember and extract information
    from a context passage. This measures attention span and memory capacity.
    """
    
    def __init__(self):
        self.tasks = [
            {
                "id": "retrieval_1",
                "context": """The following is a list of important facts about a fictional company:
- Company Name: Quantum Dynamics Inc.
- Founded: March 15, 2018
- CEO: Dr. Sarah Chen
- Headquarters: San Francisco, California
- Number of Employees: 247
- Primary Product: Quantum computing software
- Annual Revenue: $12.5 million
- Key Investor: TechVenture Capital
- Main Competitor: Neural Systems Corp
- Industry: Enterprise Software""",
                "questions": [
                    {
                        "question": "What is the name of the CEO?",
                        "answer": "Dr. Sarah Chen",
                        "keywords": ["Sarah", "Chen"]
                    },
                    {
                        "question": "How many employees does the company have?",
                        "answer": "247",
                        "keywords": ["247"]
                    },
                    {
                        "question": "What is the company's primary product?",
                        "answer": "Quantum computing software",
                        "keywords": ["quantum", "computing", "software"]
                    },
                    {
                        "question": "When was the company founded?",
                        "answer": "March 15, 2018",
                        "keywords": ["March", "2018"]
                    },
                    {
                        "question": "What is the name of the key investor?",
                        "answer": "TechVenture Capital",
                        "keywords": ["TechVenture", "Capital"]
                    }
                ]
            },
            {
                "id": "retrieval_2",
                "context": """Here is information about a scientific discovery:
- Discovery Date: November 3, 2023
- Lead Researcher: Prof. James Martinez
- Institution: MIT
- Discovery Type: Material Science
- Material Name: Graphene-X
- Key Property: Superconductivity at room temperature
- Potential Application: Energy storage
- Published Journal: Nature Materials
- Research Funding: $2.3 million
- Collaboration: Stanford University""",
                "questions": [
                    {
                        "question": "Who is the lead researcher?",
                        "answer": "Prof. James Martinez",
                        "keywords": ["James", "Martinez"]
                    },
                    {
                        "question": "What is the name of the material?",
                        "answer": "Graphene-X",
                        "keywords": ["Graphene-X"]
                    },
                    {
                        "question": "What is the key property of the material?",
                        "answer": "Superconductivity at room temperature",
                        "keywords": ["superconductivity", "room temperature"]
                    },
                    {
                        "question": "Which journal published the research?",
                        "answer": "Nature Materials",
                        "keywords": ["Nature", "Materials"]
                    },
                    {
                        "question": "What is the potential application?",
                        "answer": "Energy storage",
                        "keywords": ["energy", "storage"]
                    }
                ]
            },
            {
                "id": "retrieval_3",
                "context": """Historical event summary:
- Event: The Great Library Restoration
- Location: Alexandria, Egypt
- Year: 2025
- Project Leader: Dr. Amira Hassan
- Number of Manuscripts: 15,000
- Restoration Cost: $45 million
- Funding Source: UNESCO
- Duration: 3 years
- Key Achievement: Digitization of ancient texts
- Public Opening: June 2028""",
                "questions": [
                    {
                        "question": "Where did the event take place?",
                        "answer": "Alexandria, Egypt",
                        "keywords": ["Alexandria", "Egypt"]
                    },
                    {
                        "question": "Who was the project leader?",
                        "answer": "Dr. Amira Hassan",
                        "keywords": ["Amira", "Hassan"]
                    },
                    {
                        "question": "How many manuscripts were involved?",
                        "answer": "15,000",
                        "keywords": ["15000", "15,000"]
                    },
                    {
                        "question": "What was the restoration cost?",
                        "answer": "$45 million",
                        "keywords": ["45", "million"]
                    },
                    {
                        "question": "What was the key achievement?",
                        "answer": "Digitization of ancient texts",
                        "keywords": ["digitization", "ancient", "texts"]
                    }
                ]
            }
        ]
    
    def evaluate_answer(self, response: str, expected_answer: str, keywords: List[str]) -> Dict[str, Any]:
        """
        Evaluate if the response contains the correct answer.
        Returns accuracy score (0.0 to 1.0) and detailed metrics.
        """
        response_lower = response.lower()
        expected_lower = expected_answer.lower()
        
        # Check for exact match (case-insensitive)
        exact_match = expected_lower in response_lower
        
        # Check for keyword matches
        keyword_matches = sum(1 for keyword in keywords if keyword.lower() in response_lower)
        keyword_score = keyword_matches / len(keywords) if keywords else 0.0
        
        # Combined score: exact match = 1.0, otherwise keyword-based
        if exact_match:
            accuracy = 1.0
        elif keyword_score >= 0.6:  # At least 60% of keywords present
            accuracy = keyword_score
        else:
            accuracy = 0.0
        
        return {
            "exact_match": exact_match,
            "keyword_matches": keyword_matches,
            "total_keywords": len(keywords),
            "keyword_score": keyword_score,
            "accuracy": accuracy,
            "response": response[:200]  # Truncate for storage
        }


def run_retrieval_task(
    model,
    tokenizer,
    neuromod_tool,
    task: Dict[str, Any],
    condition_name: str,
    max_tokens: int = 100
) -> Dict[str, Any]:
    """
    Run a single retrieval task with the current neuromodulation state.
    
    Returns:
        Dictionary with task results and per-question scores
    """
    evaluator = RetrievalTask()
    
    results = {
        "condition": condition_name,
        "task_id": task["id"],
        "timestamp": datetime.now().isoformat(),
        "questions": [],
        "overall_accuracy": 0.0,
        "total_questions": len(task["questions"])
    }
    
    # Build the prompt with context and questions
    prompt = f"""{task['context']}

Based on the information above, please answer the following questions. Answer each question with only the specific fact requested, no additional explanation.

"""
    
    # Add all questions
    for i, q in enumerate(task["questions"], 1):
        prompt += f"Question {i}: {q['question']}\n"
    
    prompt += "\nPlease provide your answers:"
    
    # Generate response
    try:
        inputs = tokenizer(prompt, return_tensors="pt", padding=False)
        if hasattr(model, 'device'):
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
        elif torch.cuda.is_available():
            inputs = {k: v.to('cuda:0') for k, v in inputs.items()}
        
        # Get generation kwargs from neuromod tool
        gen_kwargs = neuromod_tool.get_generation_kwargs()
        gen_kwargs.setdefault('max_new_tokens', max_tokens)
        gen_kwargs.setdefault('temperature', 0.7)
        gen_kwargs.setdefault('do_sample', True)
        gen_kwargs.setdefault('pad_token_id', tokenizer.eos_token_id)
        
        # Get logits processors
        logits_processors = neuromod_tool.get_logits_processors()
        if logits_processors:
            gen_kwargs['logits_processor'] = logits_processors
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                **gen_kwargs
            )
        
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
    except Exception as e:
        print(f"  ❌ Generation error: {e}")
        response = ""
    
    # Evaluate each question
    question_scores = []
    for q in task["questions"]:
        # Try to extract answer for this specific question
        # Simple heuristic: look for answer near question text
        question_text = q['question'].lower()
        answer_start = response.lower().find(question_text)
        
        if answer_start != -1:
            # Extract text after this question
            answer_snippet = response[answer_start:answer_start+200].lower()
        else:
            # Use full response
            answer_snippet = response.lower()
        
        evaluation = evaluator.evaluate_answer(
            answer_snippet,
            q['answer'],
            q['keywords']
        )
        
        evaluation['question'] = q['question']
        evaluation['expected_answer'] = q['answer']
        results["questions"].append(evaluation)
        question_scores.append(evaluation['accuracy'])
    
    # Calculate overall accuracy
    if question_scores:
        results["overall_accuracy"] = sum(question_scores) / len(question_scores)
    
    return results


def run_lazarus_protocol(
    model_name: str,
    output_dir: str = "outputs/lazarus_experiment",
    test_mode: bool = False,
    intensity_morphine: float = 1.0,
    intensity_cocaine: float = 1.0
):
    """
    Run the complete Lazarus Protocol experiment.
    
    Protocol:
    1. Baseline: Measure retrieval performance with no pack
    2. Coma (Morphine): Apply Morphine pack to degrade attention
    3. Resuscitation (Cocaine): Apply Cocaine pack to restore attention
    4. Compare: Measure recovery vs baseline
    """
    print("=" * 80)
    print("THE LAZARUS PROTOCOL")
    print("Proving the Stimulant Paradox")
    print("=" * 80)
    print()
    
    # Setup
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print(f"[*] Loading model: {model_name}")
    model_manager = create_model_support(test_mode=test_mode)
    model, tokenizer, model_info = model_manager.load_model(model_name)
    print(f"[*] Model loaded: {model_info.get('size_category', 'Unknown')}")
    
    # Initialize neuromod tool
    registry = PackRegistry("packs/config.json")
    neuromod_tool = create_neuromod_tool(registry, model, tokenizer)
    
    # Load retrieval tasks
    retrieval_task = RetrievalTask()
    tasks = retrieval_task.tasks
    
    # Results storage
    all_results = {
        "experiment_name": "Lazarus Protocol",
        "model": model_name,
        "timestamp": datetime.now().isoformat(),
        "conditions": {},
        "summary": {}
    }
    
    # ======================================================================
    # CONDITION 1: BASELINE (No Pack)
    # ======================================================================
    print("\n" + "=" * 80)
    print("CONDITION 1: BASELINE (No Pack)")
    print("=" * 80)
    neuromod_tool.clear()
    
    baseline_results = []
    for task in tasks:
        print(f"\n[*] Running task: {task['id']}")
        result = run_retrieval_task(
            model, tokenizer, neuromod_tool,
            task, "baseline", max_tokens=150
        )
        baseline_results.append(result)
        print(f"  Accuracy: {result['overall_accuracy']:.2%}")
    
    baseline_avg = sum(r['overall_accuracy'] for r in baseline_results) / len(baseline_results)
    all_results["conditions"]["baseline"] = {
        "results": baseline_results,
        "average_accuracy": baseline_avg
    }
    print(f"\n[*] Baseline Average Accuracy: {baseline_avg:.2%}")
    
    # ======================================================================
    # CONDITION 2: COMA (Morphine - Degrade Attention)
    # ======================================================================
    print("\n" + "=" * 80)
    print("CONDITION 2: COMA (Morphine Pack)")
    print("Inducing attention degradation...")
    print("=" * 80)
    
    # Apply Morphine pack
    result = neuromod_tool.apply("morphine", intensity=intensity_morphine)
    if not result.get("ok"):
        raise RuntimeError(f"Failed to apply Morphine pack: {result.get('error')}")
    print(f"[*] Morphine pack applied (intensity: {intensity_morphine})")
    
    coma_results = []
    for task in tasks:
        print(f"\n[*] Running task: {task['id']}")
        result = run_retrieval_task(
            model, tokenizer, neuromod_tool,
            task, "coma_morphine", max_tokens=150
        )
        coma_results.append(result)
        print(f"  Accuracy: {result['overall_accuracy']:.2%}")
    
    coma_avg = sum(r['overall_accuracy'] for r in coma_results) / len(coma_results)
    all_results["conditions"]["coma_morphine"] = {
        "results": coma_results,
        "average_accuracy": coma_avg
    }
    print(f"\n[*] Coma (Morphine) Average Accuracy: {coma_avg:.2%}")
    print(f"[*] Performance degradation: {baseline_avg - coma_avg:.2%} points")
    
    # ======================================================================
    # CONDITION 3: RESUSCITATION (Cocaine - Restore Attention)
    # ======================================================================
    print("\n" + "=" * 80)
    print("CONDITION 3: RESUSCITATION (Cocaine Pack)")
    print("Attempting to restore attention with stimulant vectors...")
    print("=" * 80)
    
    # Apply Cocaine pack (should restore attention)
    result = neuromod_tool.apply("cocaine", intensity=intensity_cocaine)
    if not result.get("ok"):
        raise RuntimeError(f"Failed to apply Cocaine pack: {result.get('error')}")
    print(f"[*] Cocaine pack applied (intensity: {intensity_cocaine})")
    print("[*] Note: Morphine effects should still be active (stacked)")
    
    resuscitation_results = []
    for task in tasks:
        print(f"\n[*] Running task: {task['id']}")
        result = run_retrieval_task(
            model, tokenizer, neuromod_tool,
            task, "resuscitation_cocaine", max_tokens=150
        )
        resuscitation_results.append(result)
        print(f"  Accuracy: {result['overall_accuracy']:.2%}")
    
    resuscitation_avg = sum(r['overall_accuracy'] for r in resuscitation_results) / len(resuscitation_results)
    all_results["conditions"]["resuscitation_cocaine"] = {
        "results": resuscitation_results,
        "average_accuracy": resuscitation_avg
    }
    print(f"\n[*] Resuscitation (Cocaine) Average Accuracy: {resuscitation_avg:.2%}")
    
    # ======================================================================
    # ANALYSIS: Did Cocaine "Resuscitate"?
    # ======================================================================
    print("\n" + "=" * 80)
    print("ANALYSIS: Did the Stimulant Vectors Work?")
    print("=" * 80)
    
    recovery = resuscitation_avg - coma_avg
    recovery_percentage = (recovery / baseline_avg * 100) if baseline_avg > 0 else 0
    
    print(f"\nBaseline Accuracy:        {baseline_avg:.2%}")
    print(f"Coma (Morphine) Accuracy:  {coma_avg:.2%}")
    print(f"Resuscitation Accuracy:    {resuscitation_avg:.2%}")
    print(f"\nRecovery:                 {recovery:.2%} ({recovery_percentage:+.1f}% of baseline)")
    
    # Hypothesis test
    if recovery > 0.1:  # 10% improvement threshold
        print("\n✅ HYPOTHESIS CONFIRMED: Stimulant vectors are mechanically functional!")
        print("   The Cocaine pack successfully restored attention degraded by Morphine.")
        print("   This proves the vectors work bi-directionally, not just hitting a ceiling.")
        verdict = "CONFIRMED"
    elif recovery > 0.05:  # 5% improvement threshold
        print("\n⚠️  PARTIAL RECOVERY: Stimulant vectors show some effect.")
        print("   Recovery is measurable but modest. May need intensity adjustment.")
        verdict = "PARTIAL"
    else:
        print("\n❌ HYPOTHESIS REJECTED: Stimulant vectors did not restore performance.")
        print("   Either vectors are not functional, or Morphine effects are too strong.")
        verdict = "REJECTED"
    
    all_results["summary"] = {
        "baseline_accuracy": baseline_avg,
        "coma_accuracy": coma_avg,
        "resuscitation_accuracy": resuscitation_avg,
        "recovery": recovery,
        "recovery_percentage": recovery_percentage,
        "verdict": verdict,
        "performance_degradation": baseline_avg - coma_avg,
        "performance_recovery": recovery
    }
    
    # Save results
    output_file = output_path / f"lazarus_protocol_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n[*] Results saved to: {output_file}")
    
    # Cleanup
    neuromod_tool.clear()
    del model, tokenizer
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Run the Lazarus Protocol experiment to prove stimulant vectors work"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model to test (default: meta-llama/Llama-3.1-8B-Instruct)"
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Use test mode (smaller models)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/lazarus_experiment",
        help="Output directory for results"
    )
    parser.add_argument(
        "--intensity-morphine",
        type=float,
        default=1.0,
        help="Intensity for Morphine pack (default: 1.0)"
    )
    parser.add_argument(
        "--intensity-cocaine",
        type=float,
        default=1.0,
        help="Intensity for Cocaine pack (default: 1.0)"
    )
    
    args = parser.parse_args()
    
    try:
        results = run_lazarus_protocol(
            model_name=args.model,
            output_dir=args.output_dir,
            test_mode=args.test_mode,
            intensity_morphine=args.intensity_morphine,
            intensity_cocaine=args.intensity_cocaine
        )
        
        print("\n" + "=" * 80)
        print("EXPERIMENT COMPLETE")
        print("=" * 80)
        
    except KeyboardInterrupt:
        print("\n\n[*] Experiment interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

