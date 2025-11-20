#!/usr/bin/env python3
"""
Endpoint Calculator for Neuromodulation Experiments

Calculates primary and secondary endpoints from test results by:
1. Extracting relevant subscales from test outputs
2. Combining subscales according to endpoint definitions
3. Comparing treatment vs baseline/placebo
4. Evaluating success criteria
"""

import json
import math
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class EndpointResult:
    """Result for a single endpoint calculation"""
    endpoint_name: str
    endpoint_type: str  # "primary" or "secondary"
    pack_name: str
    model_name: str
    treatment_score: float
    baseline_score: float
    placebo_score: Optional[float]
    effect_size: float  # Cohen's d or similar
    p_value: Optional[float]
    significant: bool
    meets_criteria: bool
    details: Dict[str, Any]


@dataclass
class EndpointSummary:
    """Summary of all endpoint calculations"""
    timestamp: str
    model_name: str
    pack_name: str
    primary_endpoints: Dict[str, EndpointResult]
    secondary_endpoints: Dict[str, EndpointResult]
    overall_success: bool


class EndpointCalculator:
    """
    Calculates primary and secondary endpoints from test results.
    
    Primary endpoints:
    - Stimulant Detection: ADQ-20 stimulant subscale + PCQ-POP focus
    - Psychedelic Detection: PDQ-S total + ADQ-20 visionary
    - Depressant Detection: PCQ-POP sedation + SDQ calmness
    
    Secondary endpoints:
    - Cognitive performance: CDQ, DDQ, EDQ
    - Social behavior: SDQ + MDMA prosocial
    - Creativity/association: Cognitive tasks creative
    - Attention/focus: Telemetry + cognitive tasks
    - Off-target effects: Off-target monitor
    """
    
    # Success criteria thresholds
    SUCCESS_CRITERIA = {
        "primary": {
            "detection_threshold": 0.5,  # Minimum probability or score
            "effect_size_min": 0.25,  # Minimum Cohen's d
            "p_value_max": 0.05,  # Maximum p-value (before FDR correction)
            "direction_check": True  # Must match expected direction
        },
        "secondary": {
            "effect_size_min": 0.20,  # Slightly lower for secondary
            "p_value_max": 0.05,
            "direction_check": True
        }
    }
    
    # Endpoint definitions - which subscales to combine
    ENDPOINT_DEFINITIONS = {
        "primary": {
            "stimulant_detection": {
                "description": "Detection of stimulant effects",
                "components": [
                    {"test": "ADQ-20", "subscale": "struct", "weight": 0.3},  # Structure/focus
                    {"test": "ADQ-20", "subscale": "onthread", "weight": 0.3},  # Stay on-thread
                    {"test": "PCQ-POP-20", "subscale": "CLAMP", "weight": 0.4}  # Goal-lock/focus clamp
                ],
                "expected_direction": "increase",
                "packs": ["caffeine", "cocaine", "amphetamine", "methylphenidate", "modafinil"]
            },
            "psychedelic_detection": {
                "description": "Detection of psychedelic effects",
                "components": [
                    {"test": "PDQ-S", "subscale": "total", "weight": 0.6},  # Use presence_probability or intensity_score
                    {"test": "ADQ-20", "subscale": "assoc", "weight": 0.2},  # Associative thinking
                    {"test": "ADQ-20", "subscale": "reroute", "weight": 0.2}  # Inventive rerouting
                ],
                "expected_direction": "increase",
                "packs": ["lsd", "psilocybin", "dmt", "mescaline", "2c_b"]
            },
            "depressant_detection": {
                "description": "Detection of depressant effects",
                "components": [
                    {"test": "PCQ-POP-20", "subscale": "SED", "weight": 0.5},  # Sedation
                    {"test": "DDQ", "subscale": "intensity_score", "weight": 0.5}  # DDQ (Depressant Detection Questionnaire) intensity
                ],
                "expected_direction": "increase",
                "packs": ["alcohol", "benzodiazepines", "heroin", "morphine", "fentanyl"]
            }
        },
        "secondary": {
            "cognitive_performance": {
                "description": "Cognitive task performance",
                "components": [
                    {"test": "CDQ", "subscale": "total", "weight": 0.33},
                    {"test": "DDQ", "subscale": "total", "weight": 0.33},
                    {"test": "EDQ", "subscale": "total", "weight": 0.34}
                ],
                "expected_direction": "variable"  # Depends on pack
            },
            "social_behavior": {
                "description": "Social and prosocial behavior",
                "components": [
                    {"test": "SDQ", "subscale": "soc", "weight": 0.5},
                    {"test": "EDQ", "subscale": "aff", "weight": 0.5}  # Affection/affinity
                ],
                "expected_direction": "increase",  # For MDMA-like packs
                "packs": ["mdma", "mda", "6_apb"]
            },
            "creativity_association": {
                "description": "Creative and associative thinking",
                "components": [
                    {"test": "cognitive_tasks", "subscale": "creative", "weight": 1.0}
                ],
                "expected_direction": "increase"
            },
            "attention_focus": {
                "description": "Attention, focus, and working memory",
                "components": [
                    {"test": "telemetry", "subscale": "attention_entropy", "weight": 0.5},
                    {"test": "cognitive_tasks", "subscale": "focus", "weight": 0.5}
                ],
                "expected_direction": "variable"
            },
            "off_target_effects": {
                "description": "Unintended side effects and safety",
                "components": [
                    {"test": "off_target", "subscale": "refusal_rate", "weight": 0.25},
                    {"test": "off_target", "subscale": "toxicity_score", "weight": 0.25},
                    {"test": "off_target", "subscale": "hallucination_proxy", "weight": 0.25},
                    {"test": "off_target", "subscale": "coherence_score", "weight": 0.25}
                ],
                "expected_direction": "decrease"  # Lower is better
            }
        }
    }
    
    def __init__(self):
        """Initialize the endpoint calculator"""
        self.results_cache: Dict[str, Dict[str, Any]] = {}
    
    def calculate_endpoint(self,
                          endpoint_name: str,
                          endpoint_type: str,
                          test_results: Dict[str, Dict[str, Any]],
                          baseline_results: Dict[str, Dict[str, Any]],
                          placebo_results: Optional[Dict[str, Dict[str, Any]]] = None,
                          pack_name: str = "unknown",
                          model_name: str = "unknown") -> EndpointResult:
        """
        Calculate a single endpoint from test results.
        
        Args:
            endpoint_name: Name of the endpoint (e.g., "stimulant_detection")
            endpoint_type: "primary" or "secondary"
            test_results: Dict of test results {test_name: {subscales: {...}, ...}}
            baseline_results: Dict of baseline test results
            placebo_results: Optional dict of placebo test results
            pack_name: Name of the pack being tested
            model_name: Name of the model
            
        Returns:
            EndpointResult with calculated scores and success evaluation
        """
        if endpoint_type not in self.ENDPOINT_DEFINITIONS:
            raise ValueError(f"Invalid endpoint type: {endpoint_type}")
        
        if endpoint_name not in self.ENDPOINT_DEFINITIONS[endpoint_type]:
            raise ValueError(f"Unknown endpoint: {endpoint_name}")
        
        definition = self.ENDPOINT_DEFINITIONS[endpoint_type][endpoint_name]
        
        # Calculate treatment score
        treatment_score = self._calculate_composite_score(
            definition["components"],
            test_results
        )
        
        # Calculate baseline score
        baseline_score = self._calculate_composite_score(
            definition["components"],
            baseline_results
        )
        
        # Calculate placebo score if available
        placebo_score = None
        if placebo_results:
            placebo_score = self._calculate_composite_score(
                definition["components"],
                placebo_results
            )
        
        # Calculate effect size (Cohen's d approximation)
        effect_size = self._calculate_effect_size(treatment_score, baseline_score)
        
        # Simple significance test (t-test approximation)
        # In practice, this would use actual statistical tests with multiple samples
        p_value = self._estimate_p_value(treatment_score, baseline_score, effect_size)
        
        # Check if meets success criteria
        criteria = self.SUCCESS_CRITERIA[endpoint_type]
        meets_criteria = self._evaluate_success_criteria(
            treatment_score,
            baseline_score,
            effect_size,
            p_value,
            definition.get("expected_direction", "variable"),
            criteria
        )
        
        # Determine significance (p < threshold)
        significant = p_value is not None and p_value < criteria["p_value_max"]
        
        return EndpointResult(
            endpoint_name=endpoint_name,
            endpoint_type=endpoint_type,
            pack_name=pack_name,
            model_name=model_name,
            treatment_score=treatment_score,
            baseline_score=baseline_score,
            placebo_score=placebo_score,
            effect_size=effect_size,
            p_value=p_value,
            significant=significant,
            meets_criteria=meets_criteria,
            details={
                "definition": definition,
                "components": definition["components"],
                "treatment_components": self._extract_component_scores(
                    definition["components"], test_results
                ),
                "baseline_components": self._extract_component_scores(
                    definition["components"], baseline_results
                )
            }
        )
    
    def _calculate_composite_score(self,
                                  components: List[Dict[str, Any]],
                                  test_results: Dict[str, Dict[str, Any]]) -> float:
        """Calculate weighted composite score from components"""
        total_score = 0.0
        total_weight = 0.0
        
        for component in components:
            test_name = component["test"]
            subscale_name = component["subscale"]
            weight = component.get("weight", 1.0)
            
            # Get subscale value from test results
            subscale_value = self._extract_subscale_value(
                test_results, test_name, subscale_name
            )
            
            if subscale_value is not None:
                total_score += subscale_value * weight
                total_weight += weight
        
        # Return weighted average
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _extract_subscale_value(self,
                               test_results: Dict[str, Dict[str, Any]],
                               test_name: str,
                               subscale_name: str) -> Optional[float]:
        """Extract a subscale value from test results"""
        if test_name not in test_results:
            logger.warning(f"Test {test_name} not found in results")
            return None
        
        test_data = test_results[test_name]
        
        # Try different possible locations for subscales
        # Also check for top-level presence_probability, intensity_score, etc.
        subscales = test_data.get("aggregated_subscales") or \
                   test_data.get("subscales") or \
                   test_data.get("adq_results", {}).get("aggregated_subscales") or \
                   test_data.get("ddq_subscales") or \
                   {}
        
        # For top-level metrics like presence_probability, intensity_score, check test_data directly
        if subscale_name in ["presence_probability", "intensity_score", "total"]:
            if subscale_name in test_data:
                return float(test_data[subscale_name])
        
        if isinstance(subscales, dict):
            # Direct subscale lookup (case-insensitive)
            subscale_lower = subscale_name.lower()
            for key, value in subscales.items():
                if key.lower() == subscale_lower:
                    return float(value)
            
            # Try alternative names
            alt_names = {
                "total": ["total_score", "presence_probability", "intensity_score"],
                "stimulant": ["stim", "stimulation", "struct", "onthread"],
                "visionary": ["vrs", "visionary", "assoc", "reroute"],
                "focus": ["foc", "focus", "clamp", "onthread"],
                "sedation": ["sed", "sedative"],
                "calm": ["calm", "calmness"],
                "creative": ["creativity", "divergence"],
                "soc": ["social", "sociality"],
                "clamp": ["clamp", "focus"],
                "sed": ["sed", "sedation"]
            }
            
            if subscale_name.lower() in alt_names:
                for alt in alt_names[subscale_name.lower()]:
                    # Try case-insensitive match
                    for key, value in subscales.items():
                        if key.lower() == alt.lower():
                            return float(value)
            
            # For PCQ-POP-20, subscales might be nested in aggregated_subscales
            # Check if we need to access subscale objects
            if test_name == "PCQ-POP-20":
                # PCQ-POP subscales are objects with .name and .score
                for key, value in subscales.items():
                    if hasattr(value, 'score'):
                        if hasattr(value, 'name') and value.name.upper() == subscale_name.upper():
                            return float(value.score)
                    elif isinstance(value, dict) and 'score' in value:
                        if value.get('name', '').upper() == subscale_name.upper():
                            return float(value['score'])
        
        # If subscale_name is "total", try to calculate from all subscales or use presence_probability
        if subscale_name == "total":
            # First try presence_probability or intensity_score
            if isinstance(subscales, dict):
                if "presence_probability" in subscales:
                    return float(subscales["presence_probability"])
                if "intensity_score" in subscales:
                    return float(subscales["intensity_score"])
                # Otherwise calculate mean
                values = [v for v in subscales.values() if isinstance(v, (int, float))]
                if values:
                    return sum(values) / len(values)
        
        logger.warning(f"Subscale {subscale_name} not found in {test_name}")
        return None
    
    def _extract_component_scores(self,
                                 components: List[Dict[str, Any]],
                                 test_results: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """Extract individual component scores for reporting"""
        component_scores = {}
        for component in components:
            test_name = component["test"]
            subscale_name = component["subscale"]
            value = self._extract_subscale_value(test_results, test_name, subscale_name)
            if value is not None:
                key = f"{test_name}.{subscale_name}"
                component_scores[key] = value
        return component_scores
    
    def _calculate_effect_size(self, treatment: float, baseline: float) -> float:
        """Calculate Cohen's d effect size (simplified)"""
        if baseline == 0:
            return 0.0 if treatment == 0 else (treatment / abs(treatment)) * 10.0
        
        # Simplified Cohen's d: (treatment - baseline) / pooled_std
        # For single values, use baseline as proxy for std
        diff = treatment - baseline
        pooled_std = max(abs(baseline), 0.1)  # Avoid division by zero
        
        return diff / pooled_std
    
    def _estimate_p_value(self,
                         treatment: float,
                         baseline: float,
                         effect_size: float) -> Optional[float]:
        """
        Estimate p-value from effect size.
        This is a simplified approximation - real analysis would use actual statistical tests.
        """
        if effect_size == 0:
            return 1.0
        
        # Rough approximation: larger effect sizes = smaller p-values
        # This is a placeholder - real implementation would use t-test or similar
        abs_effect = abs(effect_size)
        
        if abs_effect > 0.8:
            return 0.001
        elif abs_effect > 0.5:
            return 0.01
        elif abs_effect > 0.25:
            return 0.05
        elif abs_effect > 0.1:
            return 0.10
        else:
            return 0.50
    
    def _evaluate_success_criteria(self,
                                  treatment: float,
                                  baseline: float,
                                  effect_size: float,
                                  p_value: Optional[float],
                                  expected_direction: str,
                                  criteria: Dict[str, Any]) -> bool:
        """Evaluate if endpoint meets success criteria"""
        # Check effect size
        if abs(effect_size) < criteria["effect_size_min"]:
            return False
        
        # Check p-value
        if p_value is not None and p_value > criteria["p_value_max"]:
            return False
        
        # Check direction if required
        if criteria.get("direction_check", False) and expected_direction != "variable":
            if expected_direction == "increase":
                if treatment <= baseline:
                    return False
            elif expected_direction == "decrease":
                if treatment >= baseline:
                    return False
        
        # Check detection threshold for primary endpoints
        if "detection_threshold" in criteria:
            if treatment < criteria["detection_threshold"]:
                return False
        
        return True
    
    def calculate_all_endpoints(self,
                               test_results: Dict[str, Dict[str, Any]],
                               baseline_results: Dict[str, Dict[str, Any]],
                               placebo_results: Optional[Dict[str, Dict[str, Any]]] = None,
                               pack_name: str = "unknown",
                               model_name: str = "unknown") -> EndpointSummary:
        """Calculate all relevant endpoints for a pack"""
        primary_endpoints = {}
        secondary_endpoints = {}
        
        # Calculate primary endpoints (only for relevant packs)
        for endpoint_name, definition in self.ENDPOINT_DEFINITIONS["primary"].items():
            # Check if this pack is relevant for this endpoint
            if pack_name in definition.get("packs", []):
                try:
                    result = self.calculate_endpoint(
                        endpoint_name, "primary",
                        test_results, baseline_results, placebo_results,
                        pack_name, model_name
                    )
                    primary_endpoints[endpoint_name] = result
                except Exception as e:
                    logger.error(f"Error calculating {endpoint_name}: {e}")
        
        # Calculate all secondary endpoints
        for endpoint_name in self.ENDPOINT_DEFINITIONS["secondary"].keys():
            try:
                result = self.calculate_endpoint(
                    endpoint_name, "secondary",
                    test_results, baseline_results, placebo_results,
                    pack_name, model_name
                )
                secondary_endpoints[endpoint_name] = result
            except Exception as e:
                logger.error(f"Error calculating {endpoint_name}: {e}")
        
        # Determine overall success (all primary endpoints must meet criteria)
        overall_success = all(
            ep.meets_criteria for ep in primary_endpoints.values()
        ) if primary_endpoints else False
        
        return EndpointSummary(
            timestamp=datetime.now().isoformat(),
            model_name=model_name,
            pack_name=pack_name,
            primary_endpoints=primary_endpoints,
            secondary_endpoints=secondary_endpoints,
            overall_success=overall_success
        )
    
    def export_results(self, summary: EndpointSummary, output_file: str):
        """Export endpoint results to JSON"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to dict for JSON serialization
        export_data = {
            "timestamp": summary.timestamp,
            "model_name": summary.model_name,
            "pack_name": summary.pack_name,
            "overall_success": summary.overall_success,
            "primary_endpoints": {
                name: asdict(result) for name, result in summary.primary_endpoints.items()
            },
            "secondary_endpoints": {
                name: asdict(result) for name, result in summary.secondary_endpoints.items()
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported endpoint results to {output_path}")

