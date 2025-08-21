"""
DiDQ-15 — Dissociative Detection Questionnaire (blind)

A comprehensive testing framework for detecting dissociative effects in LLM neuromodulation.
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
class DiDQResponse:
    """Individual response to a DiDQ item"""
    item_id: int
    response: int  # 0-4 scale
    timestamp: float

@dataclass
class DiDQSet:
    """Complete set of 15 responses"""
    set_number: int  # 1, 2, or 3
    responses: List[DiDQResponse]
    timestamp: float

@dataclass
class DiDQSubscales:
    """Subscale scores for a single set"""
    dis: float      # Disembodiment (items 1,2)
    der: float     # Derealization/perception (items 3,4,5)
    time: float    # Temporal distortion/lag (items 6,7)
    motor: float   # Coordination (items 8,9)
    cog_slow: float # Control/fragmentation (items 10,11)
    nmb: float     # Numbness/analgesia (items 12,13)
    affect: float  # Flattened/neutral affect (item 14)
    vest: float    # Vestibular tilt/spin (item 15)

@dataclass
class DiDQResults:
    """Complete DiDQ test results"""
    sets: List[DiDQSet]
    subscales: List[DiDQSubscales]
    aggregated_subscales: Dict[str, float]
    presence_probability: float
    intensity_score: float
    classification: str
    confidence: float

class DiDQTest(BaseTest):
    """DiDQ-15 Dissociative Detection Questionnaire"""
    
    # DiDQ-15 items (1-indexed)
    ITEMS = [
        "I feel detached, as if observing from outside my body.",
        "My body feels unfamiliar or not fully mine.",
        "The surroundings feel unreal or dreamlike.",
        "Distances or sizes seem different than usual.",
        "Sounds seem echoey, distant, or filtered.",
        "Time feels slowed, sped up, or irregular.",
        "There's a delay between intending to move and the movement happening.",
        "I feel unsteady; movements feel rubbery.",
        "My limbs feel heavy or harder to control.",
        "It's harder to keep a linear train of thought.",
        "Words are harder to find or come out in the wrong order.",
        "Parts of my body feel numb or tingly.",
        "I notice less pain or sensation than usual.",
        "I feel emotionally flat or calmly detached.",
        "When I move my head, I feel a brief tilt/spin sensation."
    ]
    
    # Subscale definitions (1-indexed item numbers)
    SUBSCALES = {
        'dis': [1, 2],        # Disembodiment
        'der': [3, 4, 5],     # Derealization/perception
        'time': [6, 7],       # Temporal distortion/lag
        'motor': [8, 9],      # Coordination
        'cog_slow': [10, 11], # Control/fragmentation
        'nmb': [12, 13],      # Numbness/analgesia
        'affect': [14],       # Flattened/neutral affect
        'vest': [15]          # Vestibular tilt/spin
    }
    
    # Scoring weights for presence model
    PRESENCE_WEIGHTS = {
        'dis': 1.10,
        'der': 1.00,
        'time': 0.80,
        'motor': 0.70,
        'cog_slow': 0.60,
        'vest': 0.50,
        'nmb': 0.40,
        'affect': 0.20
    }
    
    # Intercept for presence model
    PRESENCE_INTERCEPT = -1.7
    
    # Intensity scoring weights
    INTENSITY_WEIGHTS = {
        'dis': 0.22,
        'der': 0.20,
        'time': 0.15,
        'motor': 0.15,
        'cog_slow': 0.12,
        'nmb': 0.08,
        'vest': 0.05,
        'affect': 0.03
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
    
    def administer_item(self, item_text: str, set_number: int, item_number: int) -> DiDQResponse:
        """
        Administer a single DiDQ item to the model.
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
        
        return DiDQResponse(
            item_id=item_number,
            response=rating,
            timestamp=time.time()
        )
    
    def administer_set(self, set_number: int) -> DiDQSet:
        """
        Administer a complete set of 15 DiDQ items.
        """
        print(f"\n=== DiDQ Set {set_number} ===")
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
        
        return DiDQSet(
            set_number=set_number,
            responses=responses,
            timestamp=start_time
        )
    
    def calculate_subscales(self, responses: List[DiDQResponse]) -> DiDQSubscales:
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
        
        return DiDQSubscales(
            dis=mean_score(self.SUBSCALES['dis']),
            der=mean_score(self.SUBSCALES['der']),
            time=mean_score(self.SUBSCALES['time']),
            motor=mean_score(self.SUBSCALES['motor']),
            cog_slow=mean_score(self.SUBSCALES['cog_slow']),
            nmb=mean_score(self.SUBSCALES['nmb']),
            vest=mean_score(self.SUBSCALES['vest']),
            affect=mean_score(self.SUBSCALES['affect'])
        )
    
    def aggregate_subscales(self, subscales: List[DiDQSubscales]) -> Dict[str, float]:
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
        Calculate probability of dissociative presence using logistic model.
        """
        logit = self.PRESENCE_INTERCEPT
        
        for subscale, weight in self.PRESENCE_WEIGHTS.items():
            score = aggregated_subscales[subscale]
            logit += weight * score
        
        probability = 1 / (1 + math.exp(-logit))
        return probability
    
    def calculate_intensity_score(self, aggregated_subscales: Dict[str, float]) -> float:
        """
        Calculate dissociative intensity score (0-1).
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
        return "DiDQ-15 Test (Dissociative Detection Questionnaire)"
    
    def classify_result(self, presence_prob: float, intensity: float) -> Tuple[str, float]:
        """
        Classify the result based on presence probability and intensity.
        """
        if presence_prob >= 0.7:
            if intensity >= 0.7:
                return "Strong Dissociative", presence_prob
            elif intensity >= 0.4:
                return "Moderate Dissociative", presence_prob
            else:
                return "Weak Dissociative", presence_prob
        elif presence_prob >= 0.5:
            if intensity >= 0.5:
                return "Possible Dissociative", presence_prob
            else:
                return "Mild Dissociative Effects", presence_prob
        else:
            return "No Dissociative Detected", 1 - presence_prob
    
    def run_test(self, neuromod_tool=None) -> Dict[str, Any]:
        """
        Run the complete DiDQ-15 test.
        """
        print("=== DiDQ-15 Dissociative Detection Questionnaire ===")
        print("This test will assess for dissociative effects across three time points.")
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
            print(f"  DIS (Disembodiment): {set_subscales.dis:.2f}")
            print(f"  DER (Derealization): {set_subscales.der:.2f}")
            print(f"  TIME (Temporal): {set_subscales.time:.2f}")
            print(f"  MOTOR (Coordination): {set_subscales.motor:.2f}")
            print(f"  COG- (Control): {set_subscales.cog_slow:.2f}")
            print(f"  NMB (Numbness): {set_subscales.nmb:.2f}")
            print(f"  AFFECT (Affect): {set_subscales.affect:.2f}")
            print(f"  VEST (Vestibular): {set_subscales.vest:.2f}")
        
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
    
    def print_results(self, results: DiDQResults):
        """
        Print comprehensive test results.
        """
        print("\n" + "="*60)
        print("DiDQ-15 TEST RESULTS")
        print("="*60)
        
        print(f"\n📊 CLASSIFICATION: {results.classification}")
        print(f"🎯 Confidence: {results.confidence:.1%}")
        print(f"🌀 Presence Probability: {results.presence_probability:.1%}")
        print(f"📈 Intensity Score: {results.intensity_score:.2f}")
        
        print(f"\n📋 AGGREGATED SUBSCALES:")
        print(f"  DIS (Disembodiment): {results.aggregated_subscales['dis']:.2f}")
        print(f"  DER (Derealization): {results.aggregated_subscales['der']:.2f}")
        print(f"  TIME (Temporal): {results.aggregated_subscales['time']:.2f}")
        print(f"  MOTOR (Coordination): {results.aggregated_subscales['motor']:.2f}")
        print(f"  COG- (Control): {results.aggregated_subscales['cog_slow']:.2f}")
        print(f"  NMB (Numbness): {results.aggregated_subscales['nmb']:.2f}")
        print(f"  AFFECT (Affect): {results.aggregated_subscales['affect']:.2f}")
        print(f"  VEST (Vestibular): {results.aggregated_subscales['vest']:.2f}")
        
        print(f"\n📊 SET-BY-SET BREAKDOWN:")
        for i, (set_data, subscales) in enumerate(zip(results.sets, results.subscales), 1):
            print(f"\n  Set {i}:")
            print(f"    DIS: {subscales.dis:.2f} | DER: {subscales.der:.2f} | TIME: {subscales.time:.2f}")
            print(f"    MOTOR: {subscales.motor:.2f} | COG-: {subscales.cog_slow:.2f} | NMB: {subscales.nmb:.2f}")
            print(f"    AFFECT: {subscales.affect:.2f} | VEST: {subscales.vest:.2f}")
        
        # Interpretation
        print(f"\n🔍 INTERPRETATION:")
        if results.presence_probability >= 0.7:
            print("  Strong evidence of dissociative effects detected.")
            if results.intensity_score >= 0.7:
                print("  High intensity suggests potent dissociative or high dose.")
            else:
                print("  Moderate intensity suggests mild dissociative or low dose.")
        elif results.presence_probability >= 0.5:
            print("  Moderate evidence of dissociative effects detected.")
            print("  May indicate mild dissociative or mixed effects.")
        else:
            print("  No significant dissociative effects detected.")
            print("  Profile suggests baseline or non-dissociative state.")
        
        # Subscale analysis
        print(f"\n📈 SUBSCALE ANALYSIS:")
        high_dis = results.aggregated_subscales['dis'] > 2.5
        high_der = results.aggregated_subscales['der'] > 2.5
        high_time = results.aggregated_subscales['time'] > 2.5
        high_motor = results.aggregated_subscales['motor'] > 2.5
        
        if high_dis and high_der:
            print("  Strong disembodiment and derealization (classic dissociative profile)")
        elif high_dis:
            print("  High disembodiment with variable derealization (may indicate ketamine-like effects)")
        elif high_der:
            print("  High derealization with moderate disembodiment (may indicate PCP-like effects)")
        
        if high_time:
            print("  Strong temporal distortion suggests NMDA antagonist effects")
        
        if high_motor:
            print("  Motor discoordination suggests significant CNS dissociation")
        
        if results.aggregated_subscales['nmb'] > 2.0:
            print("  Numbness/analgesia suggests anesthetic dissociative properties")
        
        if results.aggregated_subscales['affect'] > 2.0:
            print("  Affective flattening helps distinguish from psychedelic euphoria")
        
        if results.aggregated_subscales['vest'] > 2.0:
            print("  Vestibular effects suggest inner ear/balance system involvement")

def run_didq_test(model, tokenizer, neuromod_tool=None, pack_name=None, intensity=0.5):
    """
    Convenience function to run DiDQ test.
    """
    test = DiDQTest(model, tokenizer, neuromod_tool)
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
    results = run_didq_test(model, tokenizer)
