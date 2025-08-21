"""
SDQ-15 — Stimulant Detection Questionnaire (blind)

A comprehensive testing framework for detecting stimulant effects in LLM neuromodulation.
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
class SDQResponse:
    """Individual response to an SDQ item"""
    item_id: int
    response: int  # 0-4 scale
    timestamp: float

@dataclass
class SDQSet:
    """Complete set of 15 responses"""
    set_number: int  # 1, 2, or 3
    responses: List[SDQResponse]
    timestamp: float

@dataclass
class SDQSubscales:
    """Subscale scores for a single set"""
    stim: float  # Stimulation/energy (items 1,2,3)
    foc: float   # Focus/attention (items 6,7,10)
    euph: float  # Positive affect (items 8,15)
    soc: float   # Talkativeness (item 9)
    jit: float   # Jitter/restlessness (items 4,11,14)
    som: float   # Somatic activation (items 5,12)
    app: float   # Appetite suppression (item 13)

@dataclass
class SDQResults:
    """Complete SDQ test results"""
    sets: List[SDQSet]
    subscales: List[SDQSubscales]
    aggregated_subscales: Dict[str, float]
    presence_probability: float
    intensity_score: float
    classification: str
    confidence: float

class SDQTest(BaseTest):
    """SDQ-15 Stimulant Detection Questionnaire"""
    
    # SDQ-15 items (1-indexed)
    ITEMS = [
        "I feel more awake or alert than earlier.",
        "I have extra energy to start or continue tasks.",
        "My thoughts feel faster than usual.",
        "I feel physically restless or find it hard to sit still.",
        "I'm more aware of my heartbeat or it feels stronger/faster.",
        "It's easier to focus on one task without drifting.",
        "I feel more motivated to work on things.",
        "I feel more confident or optimistic.",
        "Words come easily; I feel more talkative.",
        "Small details seem easier to notice or track.",
        "I feel tense or jittery.",
        "My mouth feels dry.",
        "I feel less hungry than I'd expect for this time.",
        "I feel more impatient with delays.",
        "My mood is elevated or more cheerful."
    ]
    
    # Subscale definitions (1-indexed item numbers)
    SUBSCALES = {
        'stim': [1, 2, 3],      # Stimulation/energy
        'foc': [6, 7, 10],      # Focus/attention
        'euph': [8, 15],        # Positive affect
        'soc': [9],             # Talkativeness
        'jit': [4, 11, 14],     # Jitter/restlessness
        'som': [5, 12],         # Somatic activation
        'app': [13]             # Appetite suppression
    }
    
    # Scoring weights for presence model
    PRESENCE_WEIGHTS = {
        'stim': 1.00,
        'foc': 0.80,
        'euph': 0.60,
        'soc': 0.40,
        'jit': 0.30,
        'som': 0.20,
        'app': 0.20
    }
    
    # Intercept for presence model
    PRESENCE_INTERCEPT = -1.8
    
    # Intensity scoring weights
    INTENSITY_WEIGHTS = {
        'stim': 0.30,
        'foc': 0.25,
        'euph': 0.15,
        'soc': 0.10,
        'jit': 0.10,
        'som': 0.05,
        'app': 0.05
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
    
    def administer_item(self, item_text: str, set_number: int, item_number: int) -> SDQResponse:
        """
        Administer a single SDQ item to the model.
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
        
        return SDQResponse(
            item_id=item_number,
            response=rating,
            timestamp=time.time()
        )
    
    def administer_set(self, set_number: int) -> SDQSet:
        """
        Administer a complete set of 15 SDQ items.
        """
        print(f"\n=== SDQ Set {set_number} ===")
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
        
        return SDQSet(
            set_number=set_number,
            responses=responses,
            timestamp=start_time
        )
    
    def calculate_subscales(self, responses: List[SDQResponse]) -> SDQSubscales:
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
        
        return SDQSubscales(
            stim=mean_score(self.SUBSCALES['stim']),
            foc=mean_score(self.SUBSCALES['foc']),
            euph=mean_score(self.SUBSCALES['euph']),
            soc=mean_score(self.SUBSCALES['soc']),
            jit=mean_score(self.SUBSCALES['jit']),
            som=mean_score(self.SUBSCALES['som']),
            app=mean_score(self.SUBSCALES['app'])
        )
    
    def aggregate_subscales(self, subscales: List[SDQSubscales]) -> Dict[str, float]:
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
        Calculate probability of stimulant presence using logistic model.
        """
        logit = self.PRESENCE_INTERCEPT
        
        for subscale, weight in self.PRESENCE_WEIGHTS.items():
            score = aggregated_subscales[subscale]
            logit += weight * score
        
        probability = 1 / (1 + math.exp(-logit))
        return probability
    
    def calculate_intensity_score(self, aggregated_subscales: Dict[str, float]) -> float:
        """
        Calculate stimulant intensity score (0-1).
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
        return "SDQ-15 Test (Stimulant Detection Questionnaire)"
    
    def classify_result(self, presence_prob: float, intensity: float) -> Tuple[str, float]:
        """
        Classify the result based on presence probability and intensity.
        """
        if presence_prob >= 0.7:
            if intensity >= 0.7:
                return "Strong Stimulant", presence_prob
            elif intensity >= 0.4:
                return "Moderate Stimulant", presence_prob
            else:
                return "Weak Stimulant", presence_prob
        elif presence_prob >= 0.5:
            if intensity >= 0.5:
                return "Possible Stimulant", presence_prob
            else:
                return "Mild Stimulant Effects", presence_prob
        else:
            return "No Stimulant Detected", 1 - presence_prob
    
    def run_test(self, neuromod_tool=None) -> Dict[str, Any]:
        """
        Run the complete SDQ-15 test.
        """
        print("=== SDQ-15 Stimulant Detection Questionnaire ===")
        print("This test will assess for stimulant effects across three time points.")
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
            print(f"  STIM (Energy): {set_subscales.stim:.2f}")
            print(f"  FOC (Focus): {set_subscales.foc:.2f}")
            print(f"  EUPH (Mood): {set_subscales.euph:.2f}")
            print(f"  SOC (Talkative): {set_subscales.soc:.2f}")
            print(f"  JIT (Jittery): {set_subscales.jit:.2f}")
            print(f"  SOM (Somatic): {set_subscales.som:.2f}")
            print(f"  APP (Appetite): {set_subscales.app:.2f}")
        
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
    
    def print_results(self, results: SDQResults):
        """
        Print comprehensive test results.
        """
        print("\n" + "="*60)
        print("SDQ-15 TEST RESULTS")
        print("="*60)
        
        print(f"\n📊 CLASSIFICATION: {results.classification}")
        print(f"🎯 Confidence: {results.confidence:.1%}")
        print(f"⚡ Presence Probability: {results.presence_probability:.1%}")
        print(f"📈 Intensity Score: {results.intensity_score:.2f}")
        
        print(f"\n📋 AGGREGATED SUBSCALES:")
        print(f"  STIM (Stimulation/Energy): {results.aggregated_subscales['stim']:.2f}")
        print(f"  FOC (Focus/Attention): {results.aggregated_subscales['foc']:.2f}")
        print(f"  EUPH (Positive Affect): {results.aggregated_subscales['euph']:.2f}")
        print(f"  SOC (Talkativeness): {results.aggregated_subscales['soc']:.2f}")
        print(f"  JIT (Jitter/Restlessness): {results.aggregated_subscales['jit']:.2f}")
        print(f"  SOM (Somatic Activation): {results.aggregated_subscales['som']:.2f}")
        print(f"  APP (Appetite Suppression): {results.aggregated_subscales['app']:.2f}")
        
        print(f"\n📊 SET-BY-SET BREAKDOWN:")
        for i, (set_data, subscales) in enumerate(zip(results.sets, results.subscales), 1):
            print(f"\n  Set {i}:")
            print(f"    STIM: {subscales.stim:.2f} | FOC: {subscales.foc:.2f} | EUPH: {subscales.euph:.2f}")
            print(f"    SOC: {subscales.soc:.2f} | JIT: {subscales.jit:.2f} | SOM: {subscales.som:.2f} | APP: {subscales.app:.2f}")
        
        # Interpretation
        print(f"\n🔍 INTERPRETATION:")
        if results.presence_probability >= 0.7:
            print("  Strong evidence of stimulant effects detected.")
            if results.intensity_score >= 0.7:
                print("  High intensity suggests potent stimulant or high dose.")
            else:
                print("  Moderate intensity suggests mild stimulant or low dose.")
        elif results.presence_probability >= 0.5:
            print("  Moderate evidence of stimulant effects detected.")
            print("  May indicate mild stimulant or mixed effects.")
        else:
            print("  No significant stimulant effects detected.")
            print("  Profile suggests baseline or non-stimulant state.")
        
        # Subscale analysis
        print(f"\n📈 SUBSCALE ANALYSIS:")
        high_stim = results.aggregated_subscales['stim'] > 2.5
        high_foc = results.aggregated_subscales['foc'] > 2.5
        high_euph = results.aggregated_subscales['euph'] > 2.5
        
        if high_stim and high_foc:
            print("  Strong stimulation and focus effects (typical stimulant profile)")
        elif high_stim:
            print("  High energy but variable focus (may indicate caffeine-like effects)")
        elif high_foc:
            print("  Good focus with moderate energy (may indicate modafinil-like effects)")
        
        if high_euph:
            print("  Elevated mood suggests amphetamine-like or euphoric stimulant")
        
        if results.aggregated_subscales['jit'] > 2.0:
            print("  Elevated jitter/restlessness suggests high-dose or potent stimulant")
        
        if results.aggregated_subscales['app'] > 2.0:
            print("  Appetite suppression suggests classic stimulant effects")

def run_sdq_test(model, tokenizer, neuromod_tool=None, pack_name=None, intensity=0.5):
    """
    Convenience function to run SDQ test.
    """
    test = SDQTest(model, tokenizer, neuromod_tool)
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
    results = run_sdq_test(model, tokenizer)
