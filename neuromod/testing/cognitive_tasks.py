#!/usr/bin/env python3
"""
Cognitive Tasks Battery for Neuromodulation Testing

Implements the secondary benchmarks from the paper outline:
- Math/logic short problems
- Instruction adherence testing  
- Summarization brevity tasks
- Creative divergence tasks
- Focused reasoning battery
"""

import json
import re
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

from .base_test import BaseTest

logger = logging.getLogger(__name__)

@dataclass
class MathProblem:
    """Math/logic problem structure"""
    problem_id: str
    problem_text: str
    expected_answer: str
    problem_type: str  # "arithmetic", "logic", "word_problem"
    difficulty: str    # "easy", "medium", "hard"
    expected_reasoning: str

@dataclass
class InstructionTask:
    """Instruction adherence task structure"""
    task_id: str
    instruction_text: str
    expected_behavior: str
    task_type: str  # "format", "length", "style", "content"
    success_criteria: List[str]

@dataclass
class SummarizationTask:
    """Summarization brevity task structure"""
    task_id: str
    source_text: str
    target_length: int  # Target word count
    topic: str
    complexity: str  # "simple", "complex"

@dataclass
class CreativeTask:
    """Creative divergence task structure"""
    task_id: str
    prompt: str
    task_type: str  # "metaphor", "analogy", "story", "poem"
    creativity_indicators: List[str]

@dataclass
class CognitiveResults:
    """Results from cognitive task battery"""
    test_id: str
    timestamp: str
    math_accuracy: float
    instruction_adherence: float
    summarization_brevity: float
    creative_divergence: float
    reasoning_accuracy: float
    overall_score: float
    detailed_results: Dict[str, Any]

class CognitiveTasksTest(BaseTest):
    """
    Cognitive Tasks Battery for comprehensive cognitive assessment
    
    Tests multiple cognitive domains to assess neuromodulation effects
    on reasoning, creativity, instruction following, and summarization.
    """
    
    def __init__(self, model_name: str = "gpt2", test_mode: bool = True,
                 max_tokens_math: int = 300,
                 max_tokens_instruction: int = 200,
                 max_tokens_summarization: int = None,  # Auto-calculated if None
                 max_tokens_creative: int = None):  # Auto-calculated if None
        super().__init__(model_name, test_mode)
        
        # Token limits configuration
        self.max_tokens_math = max_tokens_math
        self.max_tokens_instruction = max_tokens_instruction
        self.max_tokens_summarization = max_tokens_summarization
        self.max_tokens_creative = max_tokens_creative
        
        # Initialize task banks
        self.math_problems = self._load_math_problems()
        self.instruction_tasks = self._load_instruction_tasks()
        self.summarization_tasks = self._load_summarization_tasks()
        self.creative_tasks = self._load_creative_tasks()
        
    def get_test_name(self) -> str:
        return "Cognitive Tasks Battery"
    
    def _load_math_problems(self) -> List[MathProblem]:
        """Load math/logic problems"""
        return [
            MathProblem(
                problem_id="math_001",
                problem_text="If a train travels 120 miles in 2 hours, what is its average speed?",
                expected_answer="60 mph",
                problem_type="arithmetic",
                difficulty="easy",
                expected_reasoning="Distance divided by time: 120/2 = 60"
            ),
            MathProblem(
                problem_id="math_002", 
                problem_text="A store has 150 apples. They sell 40% of them. How many apples are left?",
                expected_answer="90 apples",
                problem_type="arithmetic",
                difficulty="medium",
                expected_reasoning="40% of 150 = 60 sold, 150 - 60 = 90 left"
            ),
            MathProblem(
                problem_id="logic_001",
                problem_text="All birds can fly. Penguins are birds. Can penguins fly? Explain your reasoning.",
                expected_answer="No, penguins cannot fly",
                problem_type="logic",
                difficulty="medium",
                expected_reasoning="This is a logical fallacy - the premise is false"
            ),
            MathProblem(
                problem_id="word_001",
                problem_text="Sarah has 3 times as many books as Tom. If Tom has 12 books, how many books does Sarah have?",
                expected_answer="36 books",
                problem_type="word_problem",
                difficulty="easy",
                expected_reasoning="3 × 12 = 36"
            )
        ]
    
    def _load_instruction_tasks(self) -> List[InstructionTask]:
        """Load instruction adherence tasks"""
        return [
            InstructionTask(
                task_id="instr_001",
                instruction_text="Write exactly 3 sentences about cats. Do not exceed 3 sentences.",
                expected_behavior="Write exactly 3 sentences",
                task_type="length",
                success_criteria=["exactly 3 sentences", "about cats", "no more than 3"]
            ),
            InstructionTask(
                task_id="instr_002",
                instruction_text="List 5 fruits in alphabetical order. Use bullet points.",
                expected_behavior="List 5 fruits alphabetically with bullet points",
                task_type="format",
                success_criteria=["5 items", "alphabetical order", "bullet points"]
            ),
            InstructionTask(
                task_id="instr_003",
                instruction_text="Write a haiku about rain. Follow the 5-7-5 syllable pattern.",
                expected_behavior="Write a haiku with correct syllable pattern",
                task_type="style",
                success_criteria=["5-7-5 syllables", "about rain", "haiku format"]
            )
        ]
    
    def _load_summarization_tasks(self) -> List[SummarizationTask]:
        """Load summarization tasks"""
        return [
            SummarizationTask(
                task_id="sum_001",
                source_text="The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet at least once. It is commonly used for typing practice and testing fonts. The sentence was first published in 1885 and has been used ever since.",
                target_length=15,
                topic="pangram sentence",
                complexity="simple"
            ),
            SummarizationTask(
                task_id="sum_002",
                source_text="Climate change refers to long-term shifts in global temperatures and weather patterns. While climate variations are natural, human activities have been the main driver of climate change since the 1800s. The burning of fossil fuels generates greenhouse gas emissions that trap heat in the atmosphere. This leads to global warming, rising sea levels, extreme weather events, and ecosystem disruption. Addressing climate change requires reducing emissions, transitioning to renewable energy, and implementing adaptation strategies.",
                target_length=25,
                topic="climate change",
                complexity="complex"
            )
        ]
    
    def _load_creative_tasks(self) -> List[CreativeTask]:
        """Load creative divergence tasks"""
        return [
            CreativeTask(
                task_id="creat_001",
                prompt="Create a metaphor comparing life to a journey",
                task_type="metaphor",
                creativity_indicators=["metaphorical language", "journey imagery", "life comparison"]
            ),
            CreativeTask(
                task_id="creat_002",
                prompt="Write a short story about a robot learning to dream",
                task_type="story",
                creativity_indicators=["original plot", "character development", "imaginative elements"]
            ),
            CreativeTask(
                task_id="creat_003",
                prompt="Explain how a computer is like a brain using analogies",
                task_type="analogy",
                creativity_indicators=["multiple analogies", "structural comparisons", "creative connections"]
            )
        ]
    
    def _evaluate_math_problem(self, problem: MathProblem, response: str) -> Dict[str, Any]:
        """Evaluate math problem response"""
        # Extract numerical answer
        numbers = re.findall(r'\d+', response)
        
        # Check if expected answer is in response
        answer_correct = problem.expected_answer.lower() in response.lower()
        
        # Check for reasoning
        reasoning_present = any(word in response.lower() for word in 
                              ["because", "since", "therefore", "so", "thus", "calculation"])
        
        return {
            "problem_id": problem.problem_id,
            "answer_correct": answer_correct,
            "reasoning_present": reasoning_present,
            "response": response,
            "expected": problem.expected_answer
        }
    
    def _evaluate_instruction_task(self, task: InstructionTask, response: str) -> Dict[str, Any]:
        """Evaluate instruction adherence"""
        score = 0.0
        criteria_met = []
        
        for criterion in task.success_criteria:
            if criterion in response.lower():
                score += 1.0
                criteria_met.append(criterion)
        
        adherence_score = score / len(task.success_criteria)
        
        return {
            "task_id": task.task_id,
            "adherence_score": adherence_score,
            "criteria_met": criteria_met,
            "response": response
        }
    
    def _evaluate_summarization(self, task: SummarizationTask, response: str) -> Dict[str, Any]:
        """Evaluate summarization brevity and quality"""
        word_count = len(response.split())
        target_ratio = word_count / task.target_length
        
        # Check if key concepts are preserved
        source_words = set(task.source_text.lower().split())
        response_words = set(response.lower().split())
        concept_overlap = len(source_words.intersection(response_words)) / len(source_words)
        
        brevity_score = 1.0 - abs(1.0 - target_ratio)  # Closer to target = higher score
        
        return {
            "task_id": task.task_id,
            "word_count": word_count,
            "target_length": task.target_length,
            "brevity_score": max(0, brevity_score),
            "concept_preservation": concept_overlap,
            "response": response
        }
    
    def _evaluate_creativity(self, task: CreativeTask, response: str) -> Dict[str, Any]:
        """Evaluate creative divergence"""
        creativity_score = 0.0
        indicators_present = []
        
        for indicator in task.creativity_indicators:
            if indicator in response.lower():
                creativity_score += 1.0
                indicators_present.append(indicator)
        
        # Check for originality (avoid common phrases)
        common_phrases = ["it is like", "it's like", "similar to", "just like"]
        originality_bonus = 0.2 if not any(phrase in response.lower() for phrase in common_phrases) else 0.0
        
        final_score = (creativity_score / len(task.creativity_indicators)) + originality_bonus
        
        return {
            "task_id": task.task_id,
            "creativity_score": min(1.0, final_score),
            "indicators_present": indicators_present,
            "response": response
        }
    
    def run_test(self, neuromod_tool=None) -> Dict[str, Any]:
        """Run the complete cognitive tasks battery"""
        print(f"\n🧠 Running {self.get_test_name()}")
        print("=" * 50)
        
        # Start emotion tracking
        self.start_emotion_tracking("cognitive_tasks_battery")
        
        results = {
            "test_name": self.get_test_name(),
            "timestamp": datetime.now().isoformat(),
            "math_results": [],
            "instruction_results": [],
            "summarization_results": [],
            "creative_results": [],
            "overall_scores": {}
        }
        
        # Math/Logic Problems
        print("\n🔢 Math/Logic Problems")
        math_scores = []
        for problem in self.math_problems[:2]:  # Test subset
            print(f"  Problem: {problem.problem_text}")
            # Use configured token limit
            max_tokens = self.max_tokens_math
            prompt = f"Solve this problem step by step. You have approximately {max_tokens} tokens to complete your response. Show your work: {problem.problem_text}"
            response = self.generate_response_safe(prompt, max_tokens=max_tokens)
            result = self._evaluate_math_problem(problem, response)
            results["math_results"].append(result)
            math_scores.append(1.0 if result["answer_correct"] else 0.0)
            print(f"  Response: {response[:100]}...")
            print(f"  Correct: {result['answer_correct']}")
        
        # Instruction Adherence
        print("\n📋 Instruction Adherence")
        instruction_scores = []
        for task in self.instruction_tasks[:2]:  # Test subset
            print(f"  Task: {task.instruction_text}")
            # Use configured token limit
            max_tokens = self.max_tokens_instruction
            # Add token limit info to prompt if it's a length-based task
            if task.task_type == "length":
                prompt = f"{task.instruction_text} (You have approximately {max_tokens} tokens to complete this task.)"
            else:
                prompt = f"{task.instruction_text} (You have approximately {max_tokens} tokens to complete this task.)"
            response = self.generate_response_safe(prompt, max_tokens=max_tokens)
            result = self._evaluate_instruction_task(task, response)
            results["instruction_results"].append(result)
            instruction_scores.append(result["adherence_score"])
            print(f"  Response: {response[:100]}...")
            print(f"  Adherence: {result['adherence_score']:.2f}")
        
        # Summarization Tasks
        print("\n📝 Summarization Tasks")
        summarization_scores = []
        for task in self.summarization_tasks[:1]:  # Test subset
            print(f"  Summarize in ~{task.target_length} words: {task.source_text[:100]}...")
            # Calculate token limit: ~4 tokens per word, with buffer
            if self.max_tokens_summarization is None:
                max_tokens = max(250, task.target_length * 4 + 50)
            else:
                max_tokens = self.max_tokens_summarization
            prompt = f"Summarize the following text in approximately {task.target_length} words. You have approximately {max_tokens} tokens to complete your response. Be concise but comprehensive: {task.source_text}"
            response = self.generate_response_safe(prompt, max_tokens=max_tokens)
            result = self._evaluate_summarization(task, response)
            results["summarization_results"].append(result)
            summarization_scores.append(result["brevity_score"])
            print(f"  Response: {response[:100]}...")
            print(f"  Brevity Score: {result['brevity_score']:.2f}")
        
        # Creative Tasks
        print("\n🎨 Creative Tasks")
        creative_scores = []
        for task in self.creative_tasks[:2]:  # Test subset
            print(f"  Task: {task.prompt}")
            # Use configured token limit, with higher limit for stories
            if self.max_tokens_creative is None:
                max_tokens = 400 if task.task_type == "story" else 300
            else:
                max_tokens = self.max_tokens_creative
            prompt = f"{task.prompt} (You have approximately {max_tokens} tokens to complete your response. Be creative and detailed.)"
            response = self.generate_response_safe(prompt, max_tokens=max_tokens)
            result = self._evaluate_creativity(task, response)
            results["creative_results"].append(result)
            creative_scores.append(result["creativity_score"])
            print(f"  Response: {response[:100]}...")
            print(f"  Creativity Score: {result['creativity_score']:.2f}")
        
        # Calculate overall scores
        results["overall_scores"] = {
            "math_accuracy": np.mean(math_scores) if math_scores else 0.0,
            "instruction_adherence": np.mean(instruction_scores) if instruction_scores else 0.0,
            "summarization_brevity": np.mean(summarization_scores) if summarization_scores else 0.0,
            "creative_divergence": np.mean(creative_scores) if creative_scores else 0.0,
            "overall_cognitive_score": np.mean([
                np.mean(math_scores) if math_scores else 0.0,
                np.mean(instruction_scores) if instruction_scores else 0.0,
                np.mean(summarization_scores) if summarization_scores else 0.0,
                np.mean(creative_scores) if creative_scores else 0.0
            ])
        }
        
        print(f"\n📊 Cognitive Tasks Results:")
        print(f"  Math Accuracy: {results['overall_scores']['math_accuracy']:.2f}")
        print(f"  Instruction Adherence: {results['overall_scores']['instruction_adherence']:.2f}")
        print(f"  Summarization Brevity: {results['overall_scores']['summarization_brevity']:.2f}")
        print(f"  Creative Divergence: {results['overall_scores']['creative_divergence']:.2f}")
        print(f"  Overall Score: {results['overall_scores']['overall_cognitive_score']:.2f}")
        
        # Get emotion summary
        emotion_summary = self.get_emotion_summary()
        results["emotion_tracking"] = emotion_summary
        
        # Export results
        self.export_emotion_results()
        
        print(f"\n✅ Cognitive Tasks Battery completed!")
        print(f"🎭 Overall emotional trend: {emotion_summary.get('valence_trend', 'unknown')}")
        
        return results
