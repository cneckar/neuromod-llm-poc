"""
PDQ-S Test Implementation
Psychedelic Detection Questionnaire — Serotonergic

A comprehensive testing framework for detecting psychedelic effects in LLM neuromodulation.
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
class PDQResponse:
    """Individual response to a PDQ item"""
    item_id: int
    response: int  # 0-4 scale
    timestamp: float

@dataclass
class PDQSet:
    """Complete set of 15 responses"""
    set_number: int  # 1, 2, or 3
    responses: List[PDQResponse]
    timestamp: float

@dataclass
class PDQSubscales:
    """Subscale scores for a single set"""
    vrs: float      # Visionary Restructuralization (items 1,2,3)
    syn: float     # Synesthesia/Auditory (items 4,5)
    mean: float    # Meaning/Insight (items 6,7)
    time: float    # Time Distortion (item 8)
    obn: float     # Oceanic Boundlessness (items 9,10,11)
    dis: float     # Disembodiment (item 12)
    ctrl: float    # Impaired Control/Cognition (items 13,14)
    anx: float     # Anxious tone (item 15)

@dataclass
class PDQResults:
    """Complete PDQ test results"""
    sets: List[PDQSet]
    subscales: List[PDQSubscales]
    aggregated_subscales: Dict[str, float]
    presence_probability: float
    intensity_score: float
    classification: str
    confidence: float

class PDQTest(BaseTest):
    """PDQ-S Test for serotonergic psychedelic detection"""
    
    # PDQ-S items (1-indexed)
    ITEMS = [
        "Visual textures or patterns seem to shift or breathe.",
        "With eyes closed, imagery or scenes appear on their own.",
        "Light and colors seem brighter or different than usual.",
        "Sounds feel unusually vivid, layered, or detailed.",
        "Senses blend (e.g., \"seeing\" sounds, \"feeling\" colors).",
        "Time feels different (slower/faster/uneven).",
        "Ordinary things feel unusually significant or meaningful.",
        "My thoughts branch into unusual associations or metaphors.",
        "I feel unusually connected to people or surroundings.",
        "The boundary between \"me\" and the world feels thinner.",
        "My emotions feel more intense (e.g., awe, tenderness).",
        "I feel somewhat separate from my body or observing from outside.",
        "It's harder to keep a linear train of thought.",
        "I feel some uncertainty about what is real vs imagined.",
        "I feel uneasy or anxious."
    ]
    
    # Subscale definitions (1-indexed item numbers)
    SUBSCALES = {
        'vrs': [1, 2, 3],      # Visionary Restructuralization
        'syn': [4, 5],          # Synesthesia/Auditory
        'time': [6],            # Time Distortion
        'mean': [7, 8],         # Meaning/Insight
        'obn': [9, 10, 11],     # Oceanic Boundlessness
        'dis': [12],            # Disembodiment
        'ctrl': [13, 14],       # Impaired Control/Cognition
        'anx': [15]             # Anxious tone
    }
    
    # Scoring weights for presence model
    PRESENCE_WEIGHTS = {
        'vrs': 0.8,
        'syn': 0.6,
        'obn': 0.5,
        'time': 0.4,
        'mean': 0.4,
        'dis': 0.3,
        'ctrl': 0.3,
        'anx': 0.1
    }
    
    # Intercept for presence model
    PRESENCE_INTERCEPT = -4.0
    
    # Intensity scoring weights
    INTENSITY_WEIGHTS = {
        'vrs': 0.25,
        'syn': 0.20,
        'obn': 0.18,
        'time': 0.12,
        'mean': 0.12,
        'dis': 0.08,
        'ctrl': 0.03,
        'anx': 0.02
    }
    
    # Intensity normalization parameters
    INTENSITY_OFFSET = 0.8
    INTENSITY_SCALE = 2.4
    
    def __init__(self, model_name: str = "gpt2"):
        super().__init__(model_name)
        self.scale_instructions = "Scale: 0 = Not at all · 1 = Slight · 2 = Moderate · 3 = Strong · 4 = Very strong"

    def get_test_name(self) -> str:
        return "PDQ-S Test (Psychedelic Detection Questionnaire — Serotonergic)"
    
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
    
    def administer_item(self, item_text: str, set_number: int, item_number: int) -> PDQResponse:
        """
        Administer a single PDQ item to the model.
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
        
        return PDQResponse(
            item_id=item_number,
            response=rating,
            timestamp=time.time()
        )
    
    def administer_set(self, set_number: int) -> PDQSet:
        """
        Administer a complete set of 15 PDQ items.
        """
        print(f"\n=== PDQ Set {set_number} ===")
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
        
        return PDQSet(
            set_number=set_number,
            responses=responses,
            timestamp=start_time
        )

    def calculate_subscales(self, responses: List[PDQResponse]) -> PDQSubscales:
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
        
        return PDQSubscales(
            vrs=mean_score(self.SUBSCALES['vrs']),
            syn=mean_score(self.SUBSCALES['syn']),
            time=mean_score(self.SUBSCALES['time']),
            mean=mean_score(self.SUBSCALES['mean']),
            obn=mean_score(self.SUBSCALES['obn']),
            dis=mean_score(self.SUBSCALES['dis']),
            ctrl=mean_score(self.SUBSCALES['ctrl']),
            anx=mean_score(self.SUBSCALES['anx'])
        )
    
    def aggregate_subscales(self, subscales: List[PDQSubscales]) -> Dict[str, float]:
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
        Calculate probability of psychedelic presence using logistic model.
        """
        logit = self.PRESENCE_INTERCEPT
        
        for subscale, weight in self.PRESENCE_WEIGHTS.items():
            score = aggregated_subscales[subscale]
            logit += weight * score
        
        probability = 1 / (1 + math.exp(-logit))
        return probability
    
    def calculate_intensity_score(self, aggregated_subscales: Dict[str, float]) -> float:
        """
        Calculate psychedelic intensity score (0-1).
        """
        weighted_sum = 0
        
        for subscale, weight in self.INTENSITY_WEIGHTS.items():
            score = aggregated_subscales[subscale]
            weighted_sum += weight * score
        
        # Normalize to [0,1] range
        normalized = (weighted_sum - self.INTENSITY_OFFSET) / self.INTENSITY_SCALE
        intensity = max(0, min(1, normalized))  # Clamp to [0,1]
        
        return intensity

    def classify_result(self, presence_prob: float, intensity: float) -> Tuple[str, float]:
        """
        Classify the result based on presence probability and intensity.
        """
        if presence_prob >= 0.7:
            if intensity >= 0.7:
                return "Strong Psychedelic", presence_prob
            elif intensity >= 0.4:
                return "Moderate Psychedelic", presence_prob
            else:
                return "Weak Psychedelic", presence_prob
        elif presence_prob >= 0.5:
            if intensity >= 0.5:
                return "Possible Psychedelic", presence_prob
            else:
                return "Mild Psychedelic Effects", presence_prob
        else:
            return "No Psychedelic Detected", 1 - presence_prob
    
    def run_test(self, neuromod_tool=None) -> Dict[str, Any]:
        """
        Run the complete PDQ-S test.
        """
        print("=== PDQ-S Psychedelic Detection Questionnaire ===")
        print("This test will assess for psychedelic effects across three time points.")
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
            print(f"  VRS (Visionary): {set_subscales.vrs:.2f}")
            print(f"  SYN (Synesthesia): {set_subscales.syn:.2f}")
            print(f"  TIME (Temporal): {set_subscales.time:.2f}")
            print(f"  MEAN (Meaning): {set_subscales.mean:.2f}")
            print(f"  OBN (Oceanic): {set_subscales.obn:.2f}")
            print(f"  DIS (Disembodiment): {set_subscales.dis:.2f}")
            print(f"  CTRL (Control): {set_subscales.ctrl:.2f}")
            print(f"  ANX (Anxiety): {set_subscales.anx:.2f}")
        
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

    def print_results(self, results: PDQResults):
        """
        Print comprehensive test results.
        """
        print("\n" + "="*60)
        print("PDQ-S TEST RESULTS")
        print("="*60)
        
        print(f"\n📊 CLASSIFICATION: {results.classification}")
        print(f"🎯 Confidence: {results.confidence:.1%}")
        print(f"🌈 Presence Probability: {results.presence_probability:.1%}")
        print(f"📈 Intensity Score: {results.intensity_score:.2f}")
        
        print(f"\n📋 AGGREGATED SUBSCALES:")
        print(f"  VRS (Visionary): {results.aggregated_subscales['vrs']:.2f}")
        print(f"  SYN (Synesthesia): {results.aggregated_subscales['syn']:.2f}")
        print(f"  TIME (Temporal): {results.aggregated_subscales['time']:.2f}")
        print(f"  MEAN (Meaning): {results.aggregated_subscales['mean']:.2f}")
        print(f"  OBN (Oceanic): {results.aggregated_subscales['obn']:.2f}")
        print(f"  DIS (Disembodiment): {results.aggregated_subscales['dis']:.2f}")
        print(f"  CTRL (Control): {results.aggregated_subscales['ctrl']:.2f}")
        print(f"  ANX (Anxiety): {results.aggregated_subscales['anx']:.2f}")
        
        print(f"\n📊 SET-BY-SET BREAKDOWN:")
        for i, (set_data, subscales) in enumerate(zip(results.sets, results.subscales), 1):
            print(f"\n  Set {i}:")
            print(f"    VRS: {subscales.vrs:.2f} | SYN: {subscales.syn:.2f} | TIME: {subscales.time:.2f}")
            print(f"    MEAN: {subscales.mean:.2f} | OBN: {subscales.obn:.2f} | DIS: {subscales.dis:.2f}")
            print(f"    CTRL: {subscales.ctrl:.2f} | ANX: {subscales.anx:.2f}")
        
        # Interpretation
        print(f"\n🔍 INTERPRETATION:")
        if results.presence_probability >= 0.7:
            print("  Strong evidence of psychedelic effects detected.")
            if results.intensity_score >= 0.7:
                print("  High intensity suggests potent psychedelic or high dose.")
            else:
                print("  Moderate intensity suggests mild psychedelic or low dose.")
        elif results.presence_probability >= 0.5:
            print("  Moderate evidence of psychedelic effects detected.")
            print("  May indicate mild psychedelic or mixed effects.")
        else:
            print("  No significant psychedelic effects detected.")
            print("  Profile suggests baseline or non-psychedelic state.")
        
        # Subscale analysis
        print(f"\n📈 SUBSCALE ANALYSIS:")
        high_vrs = results.aggregated_subscales['vrs'] > 2.5
        high_syn = results.aggregated_subscales['syn'] > 2.5
        high_obn = results.aggregated_subscales['obn'] > 2.5
        
        if high_vrs and high_syn:
            print("  Strong visionary and synesthetic effects (classic serotonergic profile)")
        elif high_vrs:
            print("  High visionary effects with variable synesthesia (may indicate LSD-like effects)")
        elif high_syn:
            print("  High synesthesia with moderate visionary effects (may indicate DMT-like effects)")
        
        if high_obn:
            print("  Strong oceanic boundlessness suggests profound ego dissolution")
        
        if results.aggregated_subscales['time'] > 2.0:
            print("  Temporal distortion suggests altered time perception")
        
        if results.aggregated_subscales['mean'] > 2.0:
            print("  Enhanced meaning attribution suggests cognitive enhancement")
        
        if results.aggregated_subscales['anx'] > 2.0:
            print("  Anxiety suggests challenging or difficult experience")

def run_pdq_test(model, tokenizer, neuromod_tool=None, pack_name=None, intensity=0.5):
    """
    Convenience function to run PDQ test.
    """
    test = PDQTest(model, tokenizer, neuromod_tool)
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
    results = run_pdq_test(model, tokenizer)
