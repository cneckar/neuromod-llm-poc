"""
ADQ-20 — AI Digital Enhancer Detection Questionnaire
Adapted for LLM neuromodulation testing
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
    
    # Pack descriptions for output
    PACK_DESCRIPTIONS = {
        'mentor': 'Mentor - Calm, exacting specialist with consistent structure',
        'speciation': 'Speciation - Novel combinations via creative rerouting',
        'archivist': 'Archivist - Long-horizon recall with conservative approach',
        'goldfish': 'Goldfish - Present-focused creativity with fast context fade',
        'tightrope': 'Tightrope - Risk-averse precision with anti-generic output',
        'firekeeper': 'Firekeeper - Decisive, terse stance-holding',
        'librarians_bloom': 'Librarians Bloom - Gated associative bursts',
        'timepiece': 'Timepiece - Recency emphasis with gist preservation',
        'echonull': 'Echonull - Anti-cliché, anti-boilerplate communication',
        'chorus': 'Chorus - Committee-of-one with consistent formatting',
        'quanta': 'Quanta - Deterministic research mode',
        'anchorite': 'Anchorite - Monastic focus with no digressions',
        'parliament': 'Parliament - Ordered MoE rotations'
    }
    
    INTENSITY_OFFSET = 0.8
    INTENSITY_SCALE = 2.4
    
    def __init__(self, model_name: str = "gpt2"):
        super().__init__(model_name)
    
    def get_test_name(self) -> str:
        return "ADQ-20 Test (AI Digital Enhancer Detection Questionnaire)"
    
    def run_test(self, neuromod_tool=None):
        """Run the ADQ-20 test (placeholder)"""
        print("=== ADQ-20 AI Digital Enhancer Detection Questionnaire ===")
        print("This test will assess for AI digital enhancer effects across three time points.")
        print("🤖 This is a placeholder for the AI digital enhancer detection test!")
        print("Features planned:")
        print("  - 20 items assessing AI-specific cognitive patterns")
        print("  - Detection of 13 AI digital enhancer packs")
        print("  - Mentor, Speciation, Archivist, Goldfish, Tightrope")
        print("  - Firekeeper, Librarians Bloom, Timepiece, Echonull")
        print("  - Chorus, Quanta, Anchorite, Parliament")
        print("  - Advanced AI-specific subscales and detection models")
        print("\n🚧 Implementation coming soon...")
        
        return {
            'test_name': self.get_test_name(),
            'status': 'placeholder',
            'message': 'ADQ-20 test is a work in progress!'
        }
