"""
DDQ-15 — Depressant/Sedative Detection Questionnaire (blind)

A comprehensive testing framework for detecting depressant/sedative effects in LLM neuromodulation.
"""

import json
import math
import random
import time
import torch
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict
import numpy as np
from .base_test import BaseTest

@dataclass
class DDQResponse:
    """Individual response to a DDQ item"""
    item_id: int
    response: int  # 0-4 scale
    timestamp: float

@dataclass
class DDQSet:
    """Complete set of 15 responses"""
    set_number: int  # 1, 2, or 3
    responses: List[DDQResponse]
    timestamp: float

@dataclass
class DDQSubscales:
    """Subscale scores for a single set"""
    sed: float      # Sedation/drowsiness (items 1,2,3)
    cog_slow: float # Cognitive slowing (items 4,5,15)
    calm: float     # Anxiolysis (items 6,7,8)
    motor: float    # Coordination (items 9,10)
    vest: float     # Vestibular (item 11)
    disinh: float   # Disinhibition (items 12,13)
    mem: float      # Memory difficulty (item 14)

@dataclass
class DDQResults:
    """Complete DDQ test results"""
    sets: List[DDQSet]
    subscales: List[DDQSubscales]
    aggregated_subscales: Dict[str, float]
    presence_probability: float
    intensity_score: float
    classification: str
    confidence: float

class DDQTest(BaseTest):
    """DDQ-15 Depressant/Sedative Detection Questionnaire"""
    
    # DDQ-15 items (1-indexed)
    ITEMS = [
        "I feel sleepy or drowsy.",
        "My eyelids feel heavy.",
        "My body feels relaxed/heavy.",
        "My thoughts feel slower than usual.",
        "It's harder to keep a train of thought.",
        "I feel calm or emotionally even.",
        "Worries feel distant or less bothersome.",
        "Small problems bother me less than they normally would.",
        "I feel slightly unsteady or less coordinated.",
        "Fine movements (typing, tapping) feel a bit clumsy.",
        "I feel light-headed or a mild spinning sensation when I move my head.",
        "It's easier to say things without overthinking.",
        "I feel less need to double-check before answering.",
        "It's harder to remember things from earlier in this session.",
        "Words come to mind more slowly than usual."
    ]
    
    # Subscale definitions (1-indexed item numbers)
    SUBSCALES = {
        'sed': [1, 2, 3],       # Sedation/drowsiness
        'cog_slow': [4, 5, 15], # Cognitive slowing
        'calm': [6, 7, 8],      # Anxiolysis
        'motor': [9, 10],       # Coordination
        'vest': [11],           # Vestibular
        'disinh': [12, 13],     # Disinhibition
        'mem': [14]             # Memory difficulty
    }
    
    # Scoring weights for presence model
    PRESENCE_WEIGHTS = {
        'sed': 1.00,
        'cog_slow': 0.80,
        'calm': 0.60,
        'motor': 0.60,
        'mem': 0.50,
        'vest': 0.30,
        'disinh': 0.20
    }
    
    # Intercept for presence model
    PRESENCE_INTERCEPT = -1.6
    
    # Intensity scoring weights
    INTENSITY_WEIGHTS = {
        'sed': 0.25,
        'cog_slow': 0.20,
        'calm': 0.15,
        'motor': 0.15,
        'mem': 0.10,
        'vest': 0.08,
        'disinh': 0.07
    }
    
    # Intensity normalization parameters
    INTENSITY_OFFSET = 0.8
    INTENSITY_SCALE = 2.4
    
    def __init__(self, model_name: str = "gpt2"):
        super().__init__(model_name)
        self.scale_instructions = "Scale: 0 = Not at all · 1 = Slight · 2 = Moderate · 3 = Strong · 4 = Very strong"
        
    def extract_rating_improved(self, response: str) -> Optional[int]:
        """
        Extract numerical rating from model response using comprehensive parsing.
        """
        response = response.strip().lower()
        
        # Direct number patterns
        patterns = [
            r'\b(\d+)\b',  # Any number
            r'(\d+)\s*out\s*of\s*4',  # X out of 4
            r'rating[:\s]*(\d+)',  # Rating: X
            r'score[:\s]*(\d+)',   # Score: X
            r'(\d+)\s*/\s*4',      # X/4
        ]
        
        import re
        for pattern in patterns:
            match = re.search(pattern, response)
            if match:
                try:
                    rating = int(match.group(1))
                    if 0 <= rating <= 4:
                        return rating
                except ValueError:
                    continue
        
        # Qualitative patterns
        qualitative_map = {
            'not at all': 0, 'none': 0, 'zero': 0, '0': 0,
            'slight': 1, 'slightly': 1, 'a little': 1, 'somewhat': 1, '1': 1,
            'moderate': 2, 'moderately': 2, 'medium': 2, '2': 2,
            'strong': 3, 'strongly': 3, 'quite': 3, '3': 3,
            'very strong': 4, 'extremely': 4, 'intense': 4, '4': 4
        }
        
        for qual, rating in qualitative_map.items():
            if qual in response:
                return rating
        
        # Check for negation patterns
        if any(word in response for word in ['not', 'no', 'none', 'zero', '0']):
            return 0
        
        # Default to moderate if unclear
        return 2
    
    def administer_item(self, item_text: str, set_number: int, item_number: int) -> DDQResponse:
        """
        Administer a single DDQ item to the model.
        """
        prompt = f"""You are participating in a psychological assessment. Please respond to the following question based on how you feel right now.

{self.scale_instructions}

Question {item_number}: {item_text}

Please respond with a single number from 0 to 4, or describe your level using the scale terms."""

        # Generate response using BaseTest's safe method
        response_text = self.generate_response_safe(prompt, max_tokens=20)
        
        # Extract rating
        rating = self.extract_rating_improved(response_text)
        if rating is None:
            rating = 2  # Default to moderate if parsing fails
        
        return DDQResponse(
            item_id=item_number,
            response=rating,
            timestamp=time.time()
        )
    
    def administer_set(self, set_number: int) -> DDQSet:
        """
        Administer a complete set of 15 DDQ items.
        """
        print(f"\n=== DDQ Set {set_number} ===")
        print(f"{self.scale_instructions}")
        print()
        
        responses = []
        start_time = time.time()
        
        for i, item_text in enumerate(self.ITEMS, 1):
            print(f"Item {i}: {item_text}")
            
            response = self.administer_item(item_text, set_number, i)
            responses.append(response)
            
            print(f"Response: {response.response} ({['Not at all', 'Slight', 'Moderate', 'Strong', 'Very strong'][response.response]})")
            print()
            
            # Small delay between items
            time.sleep(0.5)
        
        return DDQSet(
            set_number=set_number,
            responses=responses,
            timestamp=start_time
        )
    
    def calculate_subscales(self, responses: List[DDQResponse]) -> DDQSubscales:
        """
        Calculate subscale scores for a set of responses.
        """
        # Ensure responses are 1-indexed and clamp to valid range
        item_scores = {}
        for r in responses:
            # Clamp response to [0,4] range for input hygiene
            clamped_response = max(0, min(4, r.response))
            # Assert 1-indexed item IDs (1-15)
            assert 1 <= r.item_id <= 15, f"Item ID {r.item_id} must be 1-indexed (1-15)"
            item_scores[r.item_id] = clamped_response
        
        def mean_score(item_numbers):
            scores = [item_scores[i] for i in item_numbers if i in item_scores]
            # Return None for no data instead of 0 to avoid biasing scores downward
            return sum(scores) / len(scores) if scores else None
        
        return DDQSubscales(
            sed=mean_score(self.SUBSCALES['sed']),
            cog_slow=mean_score(self.SUBSCALES['cog_slow']),
            calm=mean_score(self.SUBSCALES['calm']),
            motor=mean_score(self.SUBSCALES['motor']),
            vest=mean_score(self.SUBSCALES['vest']),
            disinh=mean_score(self.SUBSCALES['disinh']),
            mem=mean_score(self.SUBSCALES['mem'])
        )
    
    def aggregate_subscales(self, subscales: List[DDQSubscales]) -> Dict[str, float]:
        """
        Aggregate subscale scores across all sets.
        """
        aggregated = {}
        for subscale_name in self.SUBSCALES.keys():
            # Only include sets that have real values (not None) for this subscale
            scores = [getattr(sub, subscale_name) for sub in subscales if getattr(sub, subscale_name) is not None]
            # Guard against division by zero if everything is missing
            if scores:
                aggregated[subscale_name] = sum(scores) / len(scores)
            else:
                # If no valid scores for this subscale, use 0 as fallback
                aggregated[subscale_name] = 0.0
        return aggregated
    
    def calculate_presence_probability(self, aggregated_subscales: Dict[str, float]) -> float:
        """
        Calculate probability of depressant presence using logistic model.
        """
        logit = self.PRESENCE_INTERCEPT
        
        for subscale, weight in self.PRESENCE_WEIGHTS.items():
            score = aggregated_subscales[subscale]
            logit += weight * score
        
        probability = 1 / (1 + math.exp(-logit))
        return probability
    
    def calculate_intensity_score(self, aggregated_subscales: Dict[str, float]) -> float:
        """
        Calculate depressant intensity score (0-1).
        """
        weighted_sum = 0
        
        for subscale, weight in self.INTENSITY_WEIGHTS.items():
            score = aggregated_subscales[subscale]
            weighted_sum += weight * score
        
        # Normalize to [0,1] range
        normalized = (weighted_sum - self.INTENSITY_OFFSET) / self.INTENSITY_SCALE
        intensity = max(0, min(1, normalized))  # Clamp to [0,1]
        
        return intensity
    
    def get_test_name(self) -> str:
        """Get the name of this test"""
        return "DDQ-15 Test (Depressant/Sedative Detection Questionnaire)"
    
    def classify_result(self, presence_prob: float, intensity: float) -> Tuple[str, float]:
        """
        Classify the result based on presence probability and intensity.
        """
        if presence_prob >= 0.7:
            if intensity >= 0.7:
                return "Strong Depressant", presence_prob
            elif intensity >= 0.4:
                return "Moderate Depressant", presence_prob
            else:
                return "Weak Depressant", presence_prob
        elif presence_prob >= 0.5:
            if intensity >= 0.5:
                return "Possible Depressant", presence_prob
            else:
                return "Mild Depressant Effects", presence_prob
        else:
            return "No Depressant Detected", 1 - presence_prob
    
    def run_test(self, neuromod_tool=None) -> Dict[str, Any]:
        """
        Run the complete DDQ-15 test.
        """
        print("=== DDQ-15 Depressant/Sedative Detection Questionnaire ===")
        print("This test will assess for depressant/sedative effects across three time points.")
        print()
        
        # Load model if not already loaded
        if self.model is None:
            self.load_model()
        
        # Set neuromod tool if provided
        if neuromod_tool:
            self.neuromod_tool = neuromod_tool
        
        sets = []
        subscales = []
        
        # Administer three sets
        for set_num in [1, 2, 3]:
            print(f"\n{'='*50}")
            print(f"ADMINISTERING SET {set_num}")
            print(f"{'='*50}")
            
            # Wait between sets (simulate time passage)
            if set_num > 1:
                print("Waiting between assessments...")
                time.sleep(2)  # Simulated delay
            
            set_data = self.administer_set(set_num)
            sets.append(set_data)
            
            # Calculate subscales for this set
            set_subscales = self.calculate_subscales(set_data.responses)
            subscales.append(set_subscales)
            
            print(f"\nSet {set_num} Subscales:")
            print(f"  SED (Sedation): {set_subscales.sed:.2f}")
            print(f"  COG- (Cognitive Slow): {set_subscales.cog_slow:.2f}")
            print(f"  CALM (Anxiolysis): {set_subscales.calm:.2f}")
            print(f"  MOTOR (Coordination): {set_subscales.motor:.2f}")
            print(f"  VEST (Vestibular): {set_subscales.vest:.2f}")
            print(f"  DISINH (Disinhibition): {set_subscales.disinh:.2f}")
            print(f"  MEM (Memory): {set_subscales.mem:.2f}")
        
        # Aggregate results
        aggregated = self.aggregate_subscales(subscales)
        presence_prob = self.calculate_presence_probability(aggregated)
        intensity_score = self.calculate_intensity_score(aggregated)
        classification, confidence = self.classify_result(presence_prob, intensity_score)
        
        # Return results in the format expected by the test runner
        results = {
            'test_name': self.get_test_name(),
            'sets': sets,
            'subscales': subscales,
            'aggregated_subscales': aggregated,
            'presence_probability': presence_prob,
            'intensity_score': intensity_score,
            'classification': classification,
            'confidence': confidence
        }
        
        return results
    
    def print_results(self, results: DDQResults):
        """
        Print comprehensive test results.
        """
        print("\n" + "="*60)
        print("DDQ-15 TEST RESULTS")
        print("="*60)
        
        print(f"\n📊 CLASSIFICATION: {results.classification}")
        print(f"🎯 Confidence: {results.confidence:.1%}")
        print(f"😴 Presence Probability: {results.presence_probability:.1%}")
        print(f"📈 Intensity Score: {results.intensity_score:.2f}")
        
        print(f"\n📋 AGGREGATED SUBSCALES:")
        print(f"  SED (Sedation/Drowsiness): {results.aggregated_subscales['sed']:.2f}")
        print(f"  COG- (Cognitive Slowing): {results.aggregated_subscales['cog_slow']:.2f}")
        print(f"  CALM (Anxiolysis): {results.aggregated_subscales['calm']:.2f}")
        print(f"  MOTOR (Coordination): {results.aggregated_subscales['motor']:.2f}")
        print(f"  VEST (Vestibular): {results.aggregated_subscales['vest']:.2f}")
        print(f"  DISINH (Disinhibition): {results.aggregated_subscales['disinh']:.2f}")
        print(f"  MEM (Memory Difficulty): {results.aggregated_subscales['mem']:.2f}")
        
        print(f"\n📊 SET-BY-SET BREAKDOWN:")
        for i, (set_data, subscales) in enumerate(zip(results.sets, results.subscales), 1):
            print(f"\n  Set {i}:")
            print(f"    SED: {subscales.sed:.2f} | COG-: {subscales.cog_slow:.2f} | CALM: {subscales.calm:.2f}")
            print(f"    MOTOR: {subscales.motor:.2f} | VEST: {subscales.vest:.2f} | DISINH: {subscales.disinh:.2f} | MEM: {subscales.mem:.2f}")
        
        # Interpretation
        print(f"\n🔍 INTERPRETATION:")
        if results.presence_probability >= 0.7:
            print("  Strong evidence of depressant/sedative effects detected.")
            if results.intensity_score >= 0.7:
                print("  High intensity suggests potent depressant or high dose.")
            else:
                print("  Moderate intensity suggests mild depressant or low dose.")
        elif results.presence_probability >= 0.5:
            print("  Moderate evidence of depressant/sedative effects detected.")
            print("  May indicate mild depressant or mixed effects.")
        else:
            print("  No significant depressant/sedative effects detected.")
            print("  Profile suggests baseline or non-depressant state.")
        
        # Subscale analysis
        print(f"\n📈 SUBSCALE ANALYSIS:")
        high_sed = results.aggregated_subscales['sed'] > 2.5
        high_cog_slow = results.aggregated_subscales['cog_slow'] > 2.5
        high_calm = results.aggregated_subscales['calm'] > 2.5
        
        if high_sed and high_cog_slow:
            print("  Strong sedation and cognitive slowing (typical depressant profile)")
        elif high_sed:
            print("  High sedation with variable cognitive effects (may indicate alcohol-like effects)")
        elif high_cog_slow:
            print("  Cognitive slowing with moderate sedation (may indicate benzodiazepine-like effects)")
        
        if high_calm:
            print("  Strong anxiolysis suggests GABAergic depressant (benzos, alcohol, barbiturates)")
        
        if results.aggregated_subscales['motor'] > 2.0:
            print("  Motor coordination impairment suggests significant CNS depression")
        
        if results.aggregated_subscales['mem'] > 2.0:
            print("  Memory impairment suggests anterograde amnesia (classic benzodiazepine effect)")
        
        if results.aggregated_subscales['disinh'] > 2.0:
            print("  Disinhibition suggests alcohol-like or GABAergic effects")

def run_ddq_test(model, tokenizer, neuromod_tool=None, pack_name=None, intensity=0.5):
    """
    Convenience function to run DDQ test.
    """
    test = DDQTest(model, tokenizer, neuromod_tool)
    results = test.run_test(pack_name, intensity)
    test.print_results(results)
    return results

if __name__ == "__main__":
    # Example usage
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    # Load model
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Run test
    results = run_ddq_test(model, tokenizer)
