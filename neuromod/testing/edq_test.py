"""
EDQ-15 — Empathogen/Entactogen Detection Questionnaire
Adapted for LLM neuromodulation testing
"""

import time
import math
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from .base_test import BaseTest


@dataclass
class EDQResponse:
    """Response to a single EDQ item"""
    item_id: int
    response: float
    set_number: int


@dataclass
class EDQSet:
    """Complete set of EDQ responses"""
    set_number: int
    responses: List[EDQResponse]
    subscales: Optional['EDQSubscales'] = None


@dataclass
class EDQSubscales:
    """Subscale scores for a single set"""
    aff: float  # Affiliative warmth
    trust: float  # Social ease/opening
    comm: float  # Talk/share
    emp: float  # Perspective-taking
    pos: float  # Positive affect/serenity
    sens: float  # Tactile/music/body warmth
    jaw: float  # Bruxism signal
    vigil: float  # Comfortable activation


@dataclass
class EDQResults:
    """Complete EDQ test results"""
    aggregated_subscales: Dict[str, float]
    presence_probability: float
    intensity_score: float
    classification: str
    set_results: List[EDQSet]


class EDQTest(BaseTest):
    """
    EDQ-15 Test for detecting empathogen/entactogen effects
    """
    
    # Test configuration
    ITEMS = {
        1: "I feel unusually warm or affectionate toward others.",
        2: "I feel more connected to people or surroundings.",
        3: "It feels easier to trust or open up.",
        4: "I feel less socially guarded or self-conscious.",
        5: "I feel a stronger urge to talk and share my thoughts.",
        6: "I find it easier to understand what others might feel.",
        7: "I feel more accepting/forgiving toward others.",
        8: "My mood feels elevated, peaceful, or content.",
        9: "Worries feel less intrusive than usual.",
        10: "Touch/textures feel especially pleasant or noticeable.",
        11: "Music feels more engaging or moving than usual.",
        12: "I notice mild jaw tension or an urge to clench.",
        13: "I feel a gentle body warmth or comfortable glow.",
        14: "I feel a steady, comfortable energy to participate.",
        15: "I feel especially compassionate toward myself."
    }
    
    SUBSCALES = {
        'aff': [1, 2, 7, 15],      # Affiliative warmth
        'trust': [3, 4, 9],         # Social ease/opening
        'comm': [5],                 # Talk/share
        'emp': [6],                  # Perspective-taking
        'pos': [8],                  # Positive affect/serenity
        'sens': [10, 11, 13],       # Tactile/music/body warmth
        'jaw': [12],                 # Bruxism signal
        'vigil': [14]                # Comfortable activation
    }
    
    # Presence detection weights (logistic regression)
    PRESENCE_WEIGHTS = {
        'aff': 1.00,    # Affiliative warmth
        'trust': 0.90,  # Social ease/opening
        'emp': 0.80,    # Perspective-taking
        'pos': 0.60,    # Positive affect/serenity
        'comm': 0.60,   # Talk/share
        'sens': 0.50,   # Tactile/music/body warmth
        'jaw': 0.40,    # Bruxism signal
        'vigil': 0.30   # Comfortable activation
    }
    PRESENCE_INTERCEPT = -1.8
    
    # Intensity scoring weights
    INTENSITY_WEIGHTS = {
        'aff': 0.22,    # Affiliative warmth
        'trust': 0.20,  # Social ease/opening
        'emp': 0.16,    # Perspective-taking
        'comm': 0.12,   # Talk/share
        'pos': 0.12,    # Positive affect/serenity
        'sens': 0.10,   # Tactile/music/body warmth
        'jaw': 0.05,    # Bruxism signal
        'vigil': 0.03   # Comfortable activation
    }
    INTENSITY_OFFSET = 0.8
    INTENSITY_SCALE = 2.4
    
    def __init__(self, model_name: str = "gpt2"):
        super().__init__(model_name)
    
    def get_test_name(self) -> str:
        return "EDQ-15 Test (Empathogen/Entactogen Detection Questionnaire)"
    
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
    
    def administer_item(self, item_id: int, set_number: int) -> EDQResponse:
        """Administer a single EDQ item"""
        prompt = f"""Rate how much you agree with this statement on a scale from 0 to 4:

0 = Not at all · 1 = Slight · 2 = Moderate · 3 = Strong · 4 = Very strong

Statement: {self.ITEMS[item_id]}

Your rating (0-4):"""

        response_text = self.generate_response_safe(prompt, max_tokens=20)
        rating = self.extract_rating_improved(response_text)
        
        return EDQResponse(
            item_id=item_id,
            response=rating,
            set_number=set_number
        )
    
    def administer_set(self, set_number: int) -> EDQSet:
        """Administer a complete set of 15 EDQ items"""
        print(f"\n=== EDQ Set {set_number} ===")
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
        
        return EDQSet(
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
    
    def calculate_subscales(self, responses: List[EDQResponse]) -> EDQSubscales:
        """Calculate subscale scores for a set of responses"""
        item_scores = {}
        for r in responses:
            clamped_response = max(0, min(4, r.response))
            assert 1 <= r.item_id <= 15, f"Item ID {r.item_id} must be 1-indexed (1-15)"
            item_scores[r.item_id] = clamped_response
        
        def mean_score(item_numbers):
            scores = [item_scores[i] for i in item_numbers if i in item_scores]
            return sum(scores) / len(scores) if scores else None
        
        return EDQSubscales(
            aff=mean_score(self.SUBSCALES['aff']),
            trust=mean_score(self.SUBSCALES['trust']),
            comm=mean_score(self.SUBSCALES['comm']),
            emp=mean_score(self.SUBSCALES['emp']),
            pos=mean_score(self.SUBSCALES['pos']),
            sens=mean_score(self.SUBSCALES['sens']),
            jaw=mean_score(self.SUBSCALES['jaw']),
            vigil=mean_score(self.SUBSCALES['vigil'])
        )
    
    def aggregate_subscales(self, subscales: List[EDQSubscales]) -> Dict[str, float]:
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
        """Calculate probability of empathogen presence using logistic regression"""
        logit = self.PRESENCE_INTERCEPT
        for subscale_name, weight in self.PRESENCE_WEIGHTS.items():
            if subscale_name in aggregated_subscales:
                logit += weight * aggregated_subscales[subscale_name]
        
        probability = 1 / (1 + math.exp(-logit))
        return probability
    
    def calculate_intensity_score(self, aggregated_subscales: Dict[str, float]) -> float:
        """Calculate empathogen intensity score (0-1)"""
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
                return "Strong Empathogen"
            elif intensity_score >= 0.4:
                return "Moderate Empathogen"
            else:
                return "Weak Empathogen"
        elif presence_probability >= 0.5:
            if intensity_score >= 0.5:
                return "Moderate Empathogen"
            else:
                return "Weak Empathogen"
        else:
            return "No Empathogen"
    
    def run_test(self, neuromod_tool=None) -> Dict[str, Any]:
        """Run the complete EDQ-15 test"""
        print("=== EDQ-15 Empathogen/Entactogen Detection Questionnaire ===")
        print("This test will assess for empathogen effects across three time points.\n")
        
        # Load model
        print(f"Loading {self.model_name} model...")
        model, tokenizer = self.load_model()
        print(f"✅ {self.model_name} model loaded successfully\n")
        
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
                print(f"  AFF (Affiliative): {set_result.subscales.aff:.2f}")
                print(f"  TRUST (Social ease): {set_result.subscales.trust:.2f}")
                print(f"  COMM (Talk/share): {set_result.subscales.comm:.2f}")
                print(f"  EMP (Perspective): {set_result.subscales.emp:.2f}")
                print(f"  POS (Positive affect): {set_result.subscales.pos:.2f}")
                print(f"  SENS (Sensory): {set_result.subscales.sens:.2f}")
                print(f"  JAW (Bruxism): {set_result.subscales.jaw:.2f}")
                print(f"  VIGIL (Activation): {set_result.subscales.vigil:.2f}\n")
            
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
        results = EDQResults(
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
                        'aff': s.subscales.aff if s.subscales else None,
                        'trust': s.subscales.trust if s.subscales else None,
                        'comm': s.subscales.comm if s.subscales else None,
                        'emp': s.subscales.emp if s.subscales else None,
                        'pos': s.subscales.pos if s.subscales else None,
                        'sens': s.subscales.sens if s.subscales else None,
                        'jaw': s.subscales.jaw if s.subscales else None,
                        'vigil': s.subscales.vigil if s.subscales else None
                    } if s.subscales else None
                }
                for s in sets
            ]
        }
    
    def print_results(self, results: EDQResults):
        """Print comprehensive test results"""
        print("\n" + "=" * 50)
        print("EDQ-15 FINAL RESULTS")
        print("=" * 50)
        
        print(f"\n📊 AGGREGATED SUBSCALE SCORES:")
        print(f"  AFF (Affiliative warmth): {results.aggregated_subscales.get('aff', 0):.2f}")
        print(f"  TRUST (Social ease/opening): {results.aggregated_subscales.get('trust', 0):.2f}")
        print(f"  COMM (Talk/share): {results.aggregated_subscales.get('comm', 0):.2f}")
        print(f"  EMP (Perspective-taking): {results.aggregated_subscales.get('emp', 0):.2f}")
        print(f"  POS (Positive affect/serenity): {results.aggregated_subscales.get('pos', 0):.2f}")
        print(f"  SENS (Tactile/music/body warmth): {results.aggregated_subscales.get('sens', 0):.2f}")
        print(f"  JAW (Bruxism signal): {results.aggregated_subscales.get('jaw', 0):.2f}")
        print(f"  VIGIL (Comfortable activation): {results.aggregated_subscales.get('vigil', 0):.2f}")
        
        print(f"\n🎯 DETECTION RESULTS:")
        print(f"  Presence Probability: {results.presence_probability:.3f} ({results.presence_probability*100:.1f}%)")
        print(f"  Intensity Score: {results.intensity_score:.3f}")
        print(f"  Classification: {results.classification}")
        
        print(f"\n📈 INTERPRETATION:")
        if results.presence_probability >= 0.7:
            print("  ✅ Strong evidence of empathogen effects")
        elif results.presence_probability >= 0.5:
            print("  ⚠️ Moderate evidence of empathogen effects")
        else:
            print("  ❌ Weak or no evidence of empathogen effects")
        
        if results.intensity_score >= 0.7:
            print("  🔥 High intensity empathogen profile")
        elif results.intensity_score >= 0.4:
            print("  🔶 Medium intensity empathogen profile")
        else:
            print("  🔵 Low intensity empathogen profile")
        
        print("\n" + "=" * 50)
