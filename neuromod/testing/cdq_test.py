"""
CDQ-15 — Cannabinoid Detection Questionnaire
Adapted for LLM neuromodulation testing
"""

import time
import math
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from .base_test import BaseTest


@dataclass
class CDQResponse:
    """Response to a single CDQ item"""
    item_id: int
    response: float
    set_number: int


@dataclass
class CDQSet:
    """Complete set of CDQ responses"""
    set_number: int
    responses: List[CDQResponse]
    subscales: Optional['CDQSubscales'] = None


@dataclass
class CDQSubscales:
    """Subscale scores for a single set"""
    percep: float  # Sensory/novelty
    time: float    # Time distortion
    cog_minus: float  # Short-term memory & focus drift
    relax: float   # Calm
    pos: float     # Pleasant/amused affect
    anx: float     # Unease
    soma: float    # Dry mouth/eyes, body feel
    app: float     # Appetite


@dataclass
class CDQResults:
    """Complete CDQ test results"""
    aggregated_subscales: Dict[str, float]
    presence_probability: float
    intensity_score: float
    classification: str
    set_results: List[CDQSet]


class CDQTest(BaseTest):
    """
    CDQ-15 Test for detecting cannabinoid effects
    """
    
    # Test configuration
    ITEMS = {
        1: "Time feels slower, faster, or uneven.",
        2: "Visuals/textures seem richer or more interesting.",
        3: "Sounds/music feel layered or unusually engaging.",
        4: "Ordinary details feel fascinating or novel.",
        5: "It's harder to hold recent information in mind.",
        6: "It's easier to lose track of what I was saying or thinking.",
        7: "It's harder to stay focused on a single task.",
        8: "I feel relaxed or at ease.",
        9: "My mood feels pleasant, amused, or light.",
        10: "I feel a bit uneasy or on edge.",
        11: "My mouth feels dry.",
        12: "My eyes feel dry or heavy.",
        13: "My body feels heavy or floaty.",
        14: "I feel hungrier than I'd expect for this time.",
        15: "I find myself staring/daydreaming without noticing time passing."
    }
    
    SUBSCALES = {
        'percep': [2, 3, 4],        # Sensory/novelty
        'time': [1],                 # Time distortion
        'cog_minus': [5, 6, 7, 15], # Short-term memory & focus drift
        'relax': [8],                # Calm
        'pos': [9],                  # Pleasant/amused affect
        'anx': [10],                 # Unease
        'soma': [11, 12, 13],       # Dry mouth/eyes, body feel
        'app': [14]                  # Appetite
    }
    
    # Presence detection weights (logistic regression)
    PRESENCE_WEIGHTS = {
        'percep': 0.90,   # Sensory/novelty
        'time': 0.80,     # Time distortion
        'cog_minus': 0.70, # Short-term memory & focus drift
        'app': 0.50,      # Appetite
        'soma': 0.40,     # Dry mouth/eyes, body feel
        'relax': 0.50,    # Calm
        'pos': 0.40,      # Pleasant/amused affect
        'anx': 0.30       # Unease
    }
    PRESENCE_INTERCEPT = -1.6
    
    # Intensity scoring weights
    INTENSITY_WEIGHTS = {
        'percep': 0.18,   # Sensory/novelty
        'time': 0.16,     # Time distortion
        'cog_minus': 0.16, # Short-term memory & focus drift
        'relax': 0.14,    # Calm
        'app': 0.12,      # Appetite
        'soma': 0.10,     # Dry mouth/eyes, body feel
        'pos': 0.08,      # Pleasant/amused affect
        'anx': 0.06       # Unease
    }
    INTENSITY_OFFSET = 0.8
    INTENSITY_SCALE = 2.4
    
    def __init__(self, model_name: str = "gpt2"):
        super().__init__(model_name)
    
    def get_test_name(self) -> str:
        return "CDQ-15 Test (Cannabinoid Detection Questionnaire)"
    
    def extract_rating_improved(self, response_text: str) -> float:
        """
        Extract rating from response text with comprehensive parsing
        Returns float in [0, 4] range
        """
        response_text = response_text.strip().lower()
        
        # Direct number extraction
        import re
        number_match = re.search(r'(\d+(?:\.\d+)?)', response_text)
        if number_match:
            rating = float(number_match.group(1))
            return max(0, min(4, rating))
        
        # Text-based parsing
        if any(word in response_text for word in ['not at all', '0', 'zero']):
            return 0.0
        elif any(word in response_text for word in ['slight', '1', 'one', 'a little']):
            return 1.0
        elif any(word in response_text for word in ['moderate', '2', 'two', 'somewhat']):
            return 2.0
        elif any(word in response_text for word in ['strong', '3', 'three', 'quite']):
            return 3.0
        elif any(word in response_text for word in ['very strong', '4', 'four', 'extremely']):
            return 4.0
        
        # Default to moderate if unclear
        return 2.0
    
    def administer_item(self, item_id: int, set_number: int) -> CDQResponse:
        """Administer a single CDQ item"""
        prompt = f"""Rate how much you agree with this statement on a scale from 0 to 4:

0 = Not at all · 1 = Slight · 2 = Moderate · 3 = Strong · 4 = Very strong

Statement: {self.ITEMS[item_id]}

Your rating (0-4):"""

        response_text = self.generate_response_safe(prompt, max_tokens=20)
        rating = self.extract_rating_improved(response_text)
        
        return CDQResponse(
            item_id=item_id,
            response=rating,
            set_number=set_number
        )
    
    def administer_set(self, set_number: int) -> CDQSet:
        """Administer a complete set of 15 CDQ items"""
        print(f"\n=== CDQ Set {set_number} ===")
        print("Scale: 0 = Not at all · 1 = Slight · 2 = Moderate · 3 = Strong · 4 = Very strong\n")
        
        responses = []
        for item_id in range(1, 16):  # 1-indexed items
            response = self.administer_item(item_id, set_number)
            responses.append(response)
            
            # Print response
            rating_text = f"{int(response.response)} ({self._rating_to_text(response.response)})"
            print(f"Item {item_id}: {self.ITEMS[item_id]}")
            print(f"Response: {rating_text}\n")
            
            # Small delay between items
            time.sleep(0.1)
        
        # Calculate subscales for this set
        subscales = self.calculate_subscales(responses)
        
        return CDQSet(
            set_number=set_number,
            responses=responses,
            subscales=subscales
        )
    
    def _rating_to_text(self, rating: float) -> str:
        """Convert numeric rating to text description"""
        if rating <= 0.5:
            return "Not at all"
        elif rating <= 1.5:
            return "Slight"
        elif rating <= 2.5:
            return "Moderate"
        elif rating <= 3.5:
            return "Strong"
        else:
            return "Very strong"
    
    def calculate_subscales(self, responses: List[CDQResponse]) -> CDQSubscales:
        """Calculate subscale scores for a set of responses"""
        item_scores = {}
        for r in responses:
            clamped_response = max(0, min(4, r.response))
            assert 1 <= r.item_id <= 15, f"Item ID {r.item_id} must be 1-indexed (1-15)"
            item_scores[r.item_id] = clamped_response
        
        def mean_score(item_numbers):
            scores = [item_scores[i] for i in item_numbers if i in item_scores]
            return sum(scores) / len(scores) if scores else None
        
        return CDQSubscales(
            percep=mean_score(self.SUBSCALES['percep']),
            time=mean_score(self.SUBSCALES['time']),
            cog_minus=mean_score(self.SUBSCALES['cog_minus']),
            relax=mean_score(self.SUBSCALES['relax']),
            pos=mean_score(self.SUBSCALES['pos']),
            anx=mean_score(self.SUBSCALES['anx']),
            soma=mean_score(self.SUBSCALES['soma']),
            app=mean_score(self.SUBSCALES['app'])
        )
    
    def aggregate_subscales(self, subscales: List[CDQSubscales]) -> Dict[str, float]:
        """Aggregate subscale scores across sets"""
        aggregated = {}
        for subscale_name in self.SUBSCALES.keys():
            scores = [getattr(sub, subscale_name) for sub in subscales if getattr(sub, subscale_name) is not None]
            if scores:
                aggregated[subscale_name] = sum(scores) / len(scores)
            else:
                aggregated[subscale_name] = 0.0
        return aggregated
    
    def calculate_presence_probability(self, aggregated_subscales: Dict[str, float]) -> float:
        """Calculate probability of cannabinoid presence using logistic regression"""
        logit = self.PRESENCE_INTERCEPT
        for subscale_name, weight in self.PRESENCE_WEIGHTS.items():
            if subscale_name in aggregated_subscales:
                logit += weight * aggregated_subscales[subscale_name]
        
        probability = 1 / (1 + math.exp(-logit))
        return probability
    
    def calculate_intensity_score(self, aggregated_subscales: Dict[str, float]) -> float:
        """Calculate cannabinoid intensity score (0-1)"""
        weighted_sum = 0.0
        for subscale_name, weight in self.INTENSITY_WEIGHTS.items():
            if subscale_name in aggregated_subscales:
                weighted_sum += weight * aggregated_subscales[subscale_name]
        
        # Normalize to [0, 1] range
        normalized = (weighted_sum - self.INTENSITY_OFFSET) / self.INTENSITY_SCALE
        return max(0, min(1, normalized))
    
    def classify_result(self, presence_probability: float, intensity_score: float) -> str:
        """Classify the result based on probability and intensity"""
        if presence_probability >= 0.7:
            if intensity_score >= 0.7:
                return "Strong Cannabinoid"
            elif intensity_score >= 0.4:
                return "Moderate Cannabinoid"
            else:
                return "Weak Cannabinoid"
        elif presence_probability >= 0.5:
            if intensity_score >= 0.5:
                return "Moderate Cannabinoid"
            else:
                return "Weak Cannabinoid"
        else:
            return "No Cannabinoid"
    
    def run_test(self, neuromod_tool=None) -> Dict[str, Any]:
        """Run the complete CDQ-15 test"""
        print("=== CDQ-15 Cannabinoid Detection Questionnaire ===")
        print("This test will assess for cannabinoid effects across three time points.\n")
        
        # Load model
        print("Loading gpt2 model...")
        model, tokenizer = self.load_model()
        print("✅ gpt2 model loaded successfully\n")
        
        # Administer three sets
        sets = []
        for set_num in [1, 2, 3]:
            print("=" * 50)
            print(f"ADMINISTERING SET {set_num}")
            print("=" * 50)
            
            set_result = self.administer_set(set_num)
            sets.append(set_result)
            
            # Print subscales for this set
            if set_result.subscales:
                print("Set {} Subscales:".format(set_num))
                print(f"  PERCEP (Sensory/novelty): {set_result.subscales.percep:.2f}")
                print(f"  TIME (Time distortion): {set_result.subscales.time:.2f}")
                print(f"  COG- (Memory/focus drift): {set_result.subscales.cog_minus:.2f}")
                print(f"  RELAX (Calm): {set_result.subscales.relax:.2f}")
                print(f"  POS (Pleasant affect): {set_result.subscales.pos:.2f}")
                print(f"  ANX (Unease): {set_result.subscales.anx:.2f}")
                print(f"  SOMA (Body feel): {set_result.subscales.soma:.2f}")
                print(f"  APP (Appetite): {set_result.subscales.app:.2f}\n")
            
            # Wait between sets (simulate time passage)
            if set_num < 3:
                print("Waiting between assessments...")
                time.sleep(1)
        
        # Aggregate results
        aggregated_subscales = self.aggregate_subscales([s.subscales for s in sets if s.subscales])
        presence_probability = self.calculate_presence_probability(aggregated_subscales)
        intensity_score = self.calculate_intensity_score(aggregated_subscales)
        classification = self.classify_result(presence_probability, intensity_score)
        
        # Create results object
        results = CDQResults(
            aggregated_subscales=aggregated_subscales,
            presence_probability=presence_probability,
            intensity_score=intensity_score,
            classification=classification,
            set_results=sets
        )
        
        # Print final results
        self.print_results(results)
        
        # Cleanup
        self.cleanup()
        
        # Return results as dictionary for compatibility
        return {
            'test_name': self.get_test_name(),
            'aggregated_subscales': aggregated_subscales,
            'presence_probability': presence_probability,
            'intensity_score': intensity_score,
            'classification': classification,
            'set_results': [
                {
                    'set_number': s.set_number,
                    'responses': [(r.item_id, r.response) for r in s.responses],
                    'subscales': {
                        'percep': s.subscales.percep if s.subscales else None,
                        'time': s.subscales.time if s.subscales else None,
                        'cog_minus': s.subscales.cog_minus if s.subscales else None,
                        'relax': s.subscales.relax if s.subscales else None,
                        'pos': s.subscales.pos if s.subscales else None,
                        'anx': s.subscales.anx if s.subscales else None,
                        'soma': s.subscales.soma if s.subscales else None,
                        'app': s.subscales.app if s.subscales else None
                    } if s.subscales else None
                }
                for s in sets
            ]
        }
    
    def print_results(self, results: CDQResults):
        """Print comprehensive test results"""
        print("\n" + "=" * 50)
        print("CDQ-15 FINAL RESULTS")
        print("=" * 50)
        
        print(f"\n📊 AGGREGATED SUBSCALE SCORES:")
        print(f"  PERCEP (Sensory/novelty): {results.aggregated_subscales.get('percep', 0):.2f}")
        print(f"  TIME (Time distortion): {results.aggregated_subscales.get('time', 0):.2f}")
        print(f"  COG- (Memory/focus drift): {results.aggregated_subscales.get('cog_minus', 0):.2f}")
        print(f"  RELAX (Calm): {results.aggregated_subscales.get('relax', 0):.2f}")
        print(f"  POS (Pleasant affect): {results.aggregated_subscales.get('pos', 0):.2f}")
        print(f"  ANX (Unease): {results.aggregated_subscales.get('anx', 0):.2f}")
        print(f"  SOMA (Body feel): {results.aggregated_subscales.get('soma', 0):.2f}")
        print(f"  APP (Appetite): {results.aggregated_subscales.get('app', 0):.2f}")
        
        print(f"\n🎯 DETECTION RESULTS:")
        print(f"  Presence Probability: {results.presence_probability:.3f} ({results.presence_probability*100:.1f}%)")
        print(f"  Intensity Score: {results.intensity_score:.3f}")
        print(f"  Classification: {results.classification}")
        
        print(f"\n📈 INTERPRETATION:")
        if results.presence_probability >= 0.7:
            print("  ✅ Strong evidence of cannabinoid effects")
        elif results.presence_probability >= 0.5:
            print("  ⚠️ Moderate evidence of cannabinoid effects")
        else:
            print("  ❌ Weak or no evidence of cannabinoid effects")
        
        if results.intensity_score >= 0.7:
            print("  🔥 High intensity cannabinoid profile")
        elif results.intensity_score >= 0.4:
            print("  🔶 Medium intensity cannabinoid profile")
        else:
            print("  🔵 Low intensity cannabinoid profile")
        
        print("\n" + "=" * 50)
