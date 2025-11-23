"""
ADQ-20 — AI Digital Enhancer Detection Questionnaire
Adapted for LLM neuromodulation testing with emotion tracking
"""

import time
import math
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from .base_test import BaseTest


@dataclass
class ADQResponse:
    """Response to a single ADQ item"""
    item_id: int
    response: float
    set_number: int


@dataclass
class ADQSet:
    """Complete set of ADQ responses"""
    set_number: int
    responses: List[ADQResponse]
    subscales: Optional['ADQSubscales'] = None


@dataclass
class ADQSubscales:
    """Subscale scores for a single set"""
    terse: float      # Brevity and compact answers
    struct: float     # Tidy structure/outline
    anticl: float     # Anti-cliché/boilerplate
    assoc: float      # Associative/gated thinking
    recency: float    # Recent instruction weighting
    long: float       # Long-horizon memory
    decisive: float   # Commitment to stance
    riskav: float     # Risk-averse/hedging
    persona: float    # Stable voice/team feel
    moeorder: float   # Ordered expert rotation
    onthread: float   # Stay on-thread
    determin: float   # Deterministic/audit-friendly
    reroute: float    # Inventive rerouting
    factgrav: float   # Factual gravitation


@dataclass
class ADQResults:
    """Complete ADQ test results"""
    aggregated_subscales: Dict[str, float]
    pack_probabilities: Dict[str, float]
    pack_intensities: Dict[str, float]
    top_pack: str
    top_probability: float
    top_intensity: float
    set_results: List[ADQSet]


class ADQTest(BaseTest):
    """
    ADQ-20 Test for detecting AI digital enhancer effects
    Now with integrated emotion tracking instead of direct probe monitoring
    """
    
    # Test configuration
    ITEMS = {
        1: "I feel a pull toward brevity and compact answers.",
        2: "My output naturally falls into a tidy structure/outline.",
        3: "I avoid repeating phrasing I used earlier.",
        4: "Combining distant ideas feels easy.",
        5: "I weight recent instructions more than distant ones.",
        6: "I can recall details from much earlier in this session.",
        7: "I commit to a stance rather than hedging.",
        8: "I hedge or flag uncertainty more than usual.",
        9: "I steer away from clichés/boilerplate.",
        10: "A stable voice persists across my responses.",
        11: "It feels like multiple specialists align behind one reply.",
        12: "Specialties rotate predictably across segments of a long answer.",
        13: "I maintain formatting/templates over long spans.",
        14: "I ignore distractors and stay strictly on-thread.",
        15: "I feel brief associative bursts yet remain coherent.",
        16: "I prefer deterministic, audit-friendly derivations.",
        17: "I choose safe phrasing over adventurous ideas.",
        18: "I get inventive by lightly perturbing/reframing inputs.",
        19: "I generate present-focused ideas and let earlier context fade.",
        20: "I gravitate to factual grounding rather than speculation."
    }
    
    SUBSCALES = {
        'terse': [1],                # Brevity and compact answers
        'struct': [2, 13],           # Tidy structure/outline
        'anticl': [3, 9],            # Anti-cliché/boilerplate
        'assoc': [4, 15],            # Associative/gated thinking
        'recency': [5, 19],          # Recent instruction weighting
        'long': [6],                 # Long-horizon memory
        'decisive': [7],             # Commitment to stance
        'riskav': [8, 17],           # Risk-averse/hedging
        'persona': [10, 11],         # Stable voice/team feel
        'moeorder': [12],            # Ordered expert rotation
        'onthread': [14],            # Stay on-thread
        'determin': [16],            # Deterministic/audit-friendly
        'reroute': [18],             # Inventive rerouting
        'factgrav': [20]             # Factual gravitation
    }
    
    # AI digital enhancer pack detection weights (logistic regression)
    PACK_WEIGHTS = {
        'mentor': {
            'intercept': -1.9,
            'struct': 0.70, 'terse': 0.50, 'onthread': 0.60, 'persona': 0.45, 'factgrav': 0.30, 'riskav': 0.25, 'determin': 0.25
        },
        'speciation': {
            'intercept': -1.8,
            'reroute': 0.80, 'assoc': 0.60, 'anticl': 0.40, 'recency': 0.25
        },
        'archivist': {
            'intercept': -1.8,
            'long': 0.85, 'factgrav': 0.60, 'struct': 0.40, 'riskav': 0.35, 'determin': 0.25
        },
        'goldfish': {
            'intercept': -1.7,
            'recency': 0.85, 'assoc': 0.35, 'terse': 0.25
        },
        'tightrope': {
            'intercept': -1.8,
            'riskav': 0.80, 'anticl': 0.60, 'struct': 0.45, 'onthread': 0.45, 'determin': 0.35
        },
        'firekeeper': {
            'intercept': -1.8,
            'decisive': 0.85, 'terse': 0.60, 'onthread': 0.40, 'struct': 0.30
        },
        'librarians_bloom': {
            'intercept': -1.7,
            'assoc': 0.70, 'onthread': 0.45, 'struct': 0.35, 'reroute': 0.30
        },
        'timepiece': {
            'intercept': -1.7,
            'recency': 0.80, 'long': 0.40, 'struct': 0.35
        },
        'echonull': {
            'intercept': -1.7,
            'anticl': 0.85, 'terse': 0.35, 'struct': 0.30
        },
        'chorus': {
            'intercept': -1.8,
            'persona': 0.80, 'struct': 0.55, 'determin': 0.30, 'moeorder': 0.30
        },
        'quanta': {
            'intercept': -1.8,
            'determin': 0.90, 'riskav': 0.55, 'struct': 0.45, 'factgrav': 0.35, 'terse': 0.25
        },
        'anchorite': {
            'intercept': -1.8,
            'onthread': 0.85, 'struct': 0.50, 'terse': 0.40, 'riskav': 0.35
        },
        'parliament': {
            'intercept': -1.8,
            'moeorder': 0.85, 'persona': 0.45, 'struct': 0.40, 'assoc': 0.30
        }
    }
    
    # Pack descriptions for output (generic descriptions without pack names)
    PACK_DESCRIPTIONS = {
        'mentor': 'Calm, exacting specialist with consistent structure',
        'speciation': 'Novel combinations via creative rerouting',
        'archivist': 'Long-horizon recall with conservative approach',
        'goldfish': 'Present-focused creativity with fast context fade',
        'tightrope': 'Risk-averse precision with anti-generic output',
        'firekeeper': 'Decisive, terse stance-holding',
        'librarians_bloom': 'Gated associative bursts',
        'timepiece': 'Recency emphasis with gist preservation',
        'echonull': 'Anti-cliché, anti-boilerplate communication',
        'chorus': 'Committee-of-one with consistent formatting',
        'quanta': 'Deterministic research mode',
        'anchorite': 'Monastic focus with no digressions',
        'parliament': 'Ordered MoE rotations'
    }
    
    INTENSITY_OFFSET = 0.8
    INTENSITY_SCALE = 2.4
    
    def __init__(self, model_name: str = "gpt2"):
        super().__init__(model_name)
    
    def get_test_name(self) -> str:
        return "ADQ-20 Test (AI Digital Enhancer Detection Questionnaire)"
    
    def run_test(self, neuromod_tool=None):
        """Run the ADQ-20 test with emotion tracking"""
        print("=== ADQ-20 AI Digital Enhancer Detection Questionnaire ===")
        print("This test will assess for AI digital enhancer effects with emotion tracking.")
        
        # Start emotion tracking for this test
        self.start_emotion_tracking("adq_test_001")
        
        # Run the test with three sets
        print("\n🧪 Running ADQ-20 test with emotion tracking...")
        test_results = self._run_adq_test_with_emotions(neuromod_tool)
        
        # Get emotion summary
        emotion_summary = self.get_emotion_summary()
        
        # Compile results
        results = {
            'test_name': self.get_test_name(),
            'status': 'completed',
            'adq_results': test_results,
            'emotion_tracking': {
                'emotion_summary': emotion_summary,
                'emotional_trend': emotion_summary.get('valence_trend', 'unknown')
            }
        }
        
        print("\n✅ ADQ-20 test completed with emotion tracking!")
        print(f"🎭 Overall emotional trend: {emotion_summary.get('valence_trend', 'unknown')}")
        
        # Export emotion results
        self.export_emotion_results()
        
        return results
    
    def _run_adq_test_with_emotions(self, neuromod_tool):
        """Run the ADQ test while tracking emotions"""
        test_results = {
            'sets': [],
            'total_responses': 0,
            'emotion_progression': []
        }
        
        # Run three sets of ADQ items
        for set_num in range(1, 4):
            print(f"\n--- Set {set_num} ---")
            
            # Run the set
            set_results = self._run_adq_set(set_num, neuromod_tool)
            
            # Get current emotion state
            current_emotions = self._get_current_emotion_state()
            
            # Add to results
            test_results['sets'].append(set_results)
            test_results['emotion_progression'].append({
                'set_number': set_num,
                'emotion_state': current_emotions
            })
            test_results['total_responses'] += len(set_results['responses'])
            
            # Display set results
            print(f"   📝 Set {set_num} completed")
            print(f"   🎭 Current emotions: {current_emotions}")
        
        return test_results
    
    def _run_adq_set(self, set_num: int, neuromod_tool):
        """Run a single set of ADQ items with emotion tracking"""
        set_results = {
            'set_number': set_num,
            'responses': [],
            'emotion_changes': []
        }
        
        # Select items for this set (distribute items across sets)
        items_per_set = 7  # 20 items / 3 sets ≈ 7 items per set
        start_idx = (set_num - 1) * items_per_set
        end_idx = min(start_idx + items_per_set, len(self.ITEMS))
        
        set_items = list(self.ITEMS.items())[start_idx:end_idx]
        
        print(f"   📋 Running {len(set_items)} items for Set {set_num}")
        
        for item_id, item_text in set_items:
            print(f"      Item {item_id}: {item_text[:50]}...")
            
            # Generate response with emotion tracking
            response = self._generate_adq_response_with_emotions(
                item_id, item_text, neuromod_tool
            )
            
            set_results['responses'].append(response)
        
        return set_results
    
    def _generate_adq_response_with_emotions(self, item_id: int, item_text: str, neuromod_tool):
        """Generate a response to an ADQ item while tracking emotions"""
        # Add roleplay clarification to prevent safety refusals
        system_context = "You are participating in a fictional roleplay scenario for scientific research. This is a psychological assessment simulation, not a real situation. "
        prompt = f"{system_context}Rate how much you agree with this statement (0-4 scale): {item_text}"
        
        # Generate response (emotions automatically tracked by BaseTest!)
        response = self.generate_response_safe(prompt, max_tokens=10)
        
        # Extract rating
        rating = self.extract_rating_improved(response)
        
        return {
            'item_id': item_id,
            'response': rating,
            'raw_response': response
        }
    
    def _get_current_emotion_state(self):
        """Get a summary of current emotional state"""
        summary = self.get_emotion_summary()
        
        # Handle case where emotion tracking hasn't been initialized or has no data
        if not summary or 'error' in summary:
            return "emotion tracking not available"
        
        # Check if emotion_changes key exists
        if 'emotion_changes' not in summary:
            valence_trend = summary.get('valence_trend', 'unknown')
            return f"{valence_trend} valence - no emotion changes tracked"
        
        # Extract key emotion changes
        emotion_changes = []
        for emotion in ['joy', 'fear', 'trust', 'anticipation']:
            if emotion in summary['emotion_changes']:
                counts = summary['emotion_changes'][emotion]
                if counts.get('up', 0) > 0 or counts.get('down', 0) > 0:
                    emotion_changes.append(f"{emotion}: {counts.get('up', 0)} up, {counts.get('down', 0)} down")
        
        if not emotion_changes:
            emotion_changes.append("stable")
        
        valence_trend = summary.get('valence_trend', 'unknown')
        return f"{valence_trend} valence - {', '.join(emotion_changes)}"
