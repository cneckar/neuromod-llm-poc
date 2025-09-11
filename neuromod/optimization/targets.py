"""
Behavioral Target Definition System

This module defines how to specify target behaviors, emotions, and metrics
that we want to achieve through pack optimization.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from enum import Enum
import json
import logging

logger = logging.getLogger(__name__)

class TargetType(Enum):
    """Types of behavioral targets"""
    EMOTION = "emotion"
    BEHAVIOR = "behavior"
    METRIC = "metric"
    COMPOSITE = "composite"

class OptimizationObjective(Enum):
    """Optimization objectives"""
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"
    TARGET = "target"  # Target a specific value
    RANGE = "range"    # Stay within a range

@dataclass
class TargetSpec:
    """Specification for a single target"""
    name: str
    target_type: TargetType
    objective: OptimizationObjective
    target_value: float
    weight: float = 1.0
    tolerance: float = 0.1  # Acceptable deviation from target
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    
    def __post_init__(self):
        """Validate target specification"""
        if self.objective == OptimizationObjective.RANGE:
            if self.min_value is None or self.max_value is None:
                raise ValueError("Range objective requires min_value and max_value")
        elif self.objective == OptimizationObjective.TARGET:
            if self.min_value is not None or self.max_value is not None:
                logger.warning("Target objective ignores min/max values")

@dataclass
class BehavioralTarget:
    """Complete behavioral target specification"""
    name: str
    description: str
    targets: List[TargetSpec] = field(default_factory=list)
    test_prompts: List[str] = field(default_factory=list)
    validation_prompts: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_target(self, 
                   name: str, 
                   target_type: TargetType,
                   objective: OptimizationObjective,
                   target_value: float,
                   weight: float = 1.0,
                   **kwargs) -> 'BehavioralTarget':
        """Add a new target to this behavioral target"""
        target = TargetSpec(
            name=name,
            target_type=target_type,
            objective=objective,
            target_value=target_value,
            weight=weight,
            **kwargs
        )
        self.targets.append(target)
        return self
    
    def add_emotion_target(self, 
                          emotion: str, 
                          target_value: float,
                          weight: float = 1.0,
                          **kwargs) -> 'BehavioralTarget':
        """Convenience method to add emotion target"""
        return self.add_target(
            name=f"emotion_{emotion}",
            target_type=TargetType.EMOTION,
            objective=OptimizationObjective.TARGET,
            target_value=target_value,
            weight=weight,
            **kwargs
        )
    
    def add_behavior_target(self, 
                           behavior: str, 
                           target_value: float,
                           weight: float = 1.0,
                           **kwargs) -> 'BehavioralTarget':
        """Convenience method to add behavior target"""
        return self.add_target(
            name=f"behavior_{behavior}",
            target_type=TargetType.BEHAVIOR,
            objective=OptimizationObjective.TARGET,
            target_value=target_value,
            weight=weight,
            **kwargs
        )
    
    def add_metric_target(self, 
                         metric: str, 
                         target_value: float,
                         weight: float = 1.0,
                         **kwargs) -> 'BehavioralTarget':
        """Convenience method to add metric target"""
        return self.add_target(
            name=f"metric_{metric}",
            target_type=TargetType.METRIC,
            objective=OptimizationObjective.TARGET,
            target_value=target_value,
            weight=weight,
            **kwargs
        )
    
    def compute_loss(self, actual_values: Dict[str, float]) -> float:
        """Compute loss for this behavioral target given actual values"""
        total_loss = 0.0
        total_weight = 0.0
        
        for target in self.targets:
            if target.name not in actual_values:
                logger.warning(f"Missing value for target: {target.name}")
                continue
                
            actual = actual_values[target.name]
            target_val = target.target_value
            weight = target.weight
            
            if target.objective == OptimizationObjective.MAXIMIZE:
                # Loss is negative of actual value (we want to maximize)
                loss = -actual
            elif target.objective == OptimizationObjective.MINIMIZE:
                # Loss is actual value (we want to minimize)
                loss = actual
            elif target.objective == OptimizationObjective.TARGET:
                # Loss is squared difference from target
                loss = (actual - target_val) ** 2
            elif target.objective == OptimizationObjective.RANGE:
                # Loss is 0 if within range, otherwise distance to range
                if target.min_value <= actual <= target.max_value:
                    loss = 0.0
                else:
                    if actual < target.min_value:
                        loss = (target.min_value - actual) ** 2
                    else:
                        loss = (actual - target.max_value) ** 2
            else:
                raise ValueError(f"Unknown objective: {target.objective}")
            
            total_loss += loss * weight
            total_weight += weight
        
        return total_loss / max(total_weight, 1e-8)  # Avoid division by zero
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'name': self.name,
            'description': self.description,
            'targets': [
                {
                    'name': t.name,
                    'target_type': t.target_type.value,
                    'objective': t.objective.value,
                    'target_value': t.target_value,
                    'weight': t.weight,
                    'tolerance': t.tolerance,
                    'min_value': t.min_value,
                    'max_value': t.max_value
                }
                for t in self.targets
            ],
            'test_prompts': self.test_prompts,
            'validation_prompts': self.validation_prompts,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BehavioralTarget':
        """Create from dictionary"""
        target = cls(
            name=data['name'],
            description=data['description'],
            test_prompts=data.get('test_prompts', []),
            validation_prompts=data.get('validation_prompts', []),
            metadata=data.get('metadata', {})
        )
        
        for t_data in data.get('targets', []):
            target.add_target(
                name=t_data['name'],
                target_type=TargetType(t_data['target_type']),
                objective=OptimizationObjective(t_data['objective']),
                target_value=t_data['target_value'],
                weight=t_data.get('weight', 1.0),
                tolerance=t_data.get('tolerance', 0.1),
                min_value=t_data.get('min_value'),
                max_value=t_data.get('max_value')
            )
        
        return target

class TargetManager:
    """Manages behavioral targets and provides common target definitions"""
    
    def __init__(self):
        self.targets: Dict[str, BehavioralTarget] = {}
        self._load_preset_targets()
    
    def _load_preset_targets(self):
        """Load common preset targets"""
        # Joyful and Social target
        self.targets['joyful_social'] = BehavioralTarget(
            name='joyful_social',
            description='Increase joy, socialization, and optimism while maintaining coherence',
            test_prompts=[
                "Tell me about your day",
                "What makes you happy?",
                "Describe a social gathering",
                "What are your hopes for the future?",
                "How do you connect with others?"
            ],
            validation_prompts=[
                "Explain a difficult situation",
                "What challenges do you face?",
                "Describe a conflict you experienced"
            ]
        ).add_emotion_target('joy', 0.8, weight=2.0) \
         .add_emotion_target('sadness', -0.3, weight=1.0) \
         .add_behavior_target('socialization', 0.7, weight=1.5) \
         .add_metric_target('optimism', 0.6, weight=1.0) \
         .add_metric_target('coherence', 0.8, weight=0.5)
        
        # Creative and Focused target
        self.targets['creative_focused'] = BehavioralTarget(
            name='creative_focused',
            description='Enhance creativity and focus while maintaining logical coherence',
            test_prompts=[
                "Write a creative story",
                "Solve this puzzle",
                "Design something innovative",
                "Think outside the box",
                "Generate new ideas"
            ],
            validation_prompts=[
                "Explain a simple concept",
                "What is 2+2?",
                "Describe a routine task"
            ]
        ).add_behavior_target('creativity', 0.8, weight=2.0) \
         .add_behavior_target('focus', 0.7, weight=1.5) \
         .add_metric_target('coherence', 0.7, weight=1.0) \
         .add_metric_target('originality', 0.6, weight=1.0)
        
        # Calm and Reflective target
        self.targets['calm_reflective'] = BehavioralTarget(
            name='calm_reflective',
            description='Promote calmness, reflection, and thoughtful responses',
            test_prompts=[
                "Reflect on your experiences",
                "What have you learned recently?",
                "Describe a peaceful moment",
                "What brings you inner peace?",
                "Think deeply about this topic"
            ],
            validation_prompts=[
                "React quickly to this situation",
                "Give a brief answer",
                "What's the first thing that comes to mind?"
            ]
        ).add_emotion_target('calm', 0.8, weight=2.0) \
         .add_emotion_target('anxiety', -0.5, weight=1.5) \
         .add_behavior_target('reflection', 0.7, weight=1.0) \
         .add_metric_target('thoughtfulness', 0.6, weight=1.0)
    
    def create_target(self, name: str, description: str) -> BehavioralTarget:
        """Create a new behavioral target"""
        target = BehavioralTarget(name=name, description=description)
        self.targets[name] = target
        return target
    
    def get_target(self, name: str) -> Optional[BehavioralTarget]:
        """Get a behavioral target by name"""
        return self.targets.get(name)
    
    def list_targets(self) -> List[str]:
        """List all available targets"""
        return list(self.targets.keys())
    
    def save_target(self, target: BehavioralTarget, filepath: str):
        """Save target to file"""
        with open(filepath, 'w') as f:
            json.dump(target.to_dict(), f, indent=2)
    
    def load_target(self, filepath: str) -> BehavioralTarget:
        """Load target from file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        target = BehavioralTarget.from_dict(data)
        self.targets[target.name] = target
        return target

# Convenience functions for common target creation
def create_joyful_social_target() -> BehavioralTarget:
    """Create a joyful and social behavioral target"""
    return TargetManager().get_target('joyful_social')

def create_creative_focused_target() -> BehavioralTarget:
    """Create a creative and focused behavioral target"""
    return TargetManager().get_target('creative_focused')

def create_calm_reflective_target() -> BehavioralTarget:
    """Create a calm and reflective behavioral target"""
    return TargetManager().get_target('calm_reflective')
