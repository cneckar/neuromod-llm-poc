#!/usr/bin/env python3
"""
Experimental Design System for Neuromodulation Testing

Implements the double-blind, placebo-controlled, randomized within-model crossover
design from the paper outline (Section 4.4).
"""

import json
import hashlib
import random
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class ConditionType(Enum):
    """Experimental condition types"""
    CONTROL = "control"           # No neuromodulation (none pack)
    PERSONA_BASELINE = "persona"  # Prompt-only persona equivalent
    TREATMENT = "treatment"       # Full neuromodulation pack

@dataclass
class Condition:
    """Experimental condition definition"""
    condition_id: str
    condition_type: ConditionType
    pack_name: Optional[str] = None
    persona_prompt: Optional[str] = None
    intensity: float = 1.0
    description: str = ""

@dataclass
class Trial:
    """Single experimental trial"""
    trial_id: str
    prompt_id: str
    condition_id: str
    condition_type: ConditionType
    pack_name: Optional[str]
    persona_prompt: Optional[str]
    intensity: float
    order: int
    blinded_id: str  # Opaque identifier for blinding

@dataclass
class ExperimentalSession:
    """Complete experimental session"""
    session_id: str
    model_name: str
    test_name: str
    timestamp: str
    conditions: List[Condition]
    trials: List[Trial]
    randomization_seed: int
    latin_square_order: List[int]

class ExperimentalDesigner:
    """
    Designs and manages double-blind, placebo-controlled experiments
    
    Implements Latin square randomization and proper blinding for
    within-model crossover studies.
    """
    
    def __init__(self, seed: Optional[int] = None):
        self.seed = seed or int(datetime.now().timestamp())
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        # Standard condition definitions
        self.standard_conditions = {
            "control": Condition(
                condition_id="control",
                condition_type=ConditionType.CONTROL,
                description="No neuromodulation applied"
            ),
            "persona_baseline": Condition(
                condition_id="persona_baseline", 
                condition_type=ConditionType.PERSONA_BASELINE,
                description="Prompt-only persona equivalent"
            )
        }
    
    def create_condition(self, 
                        condition_type: ConditionType,
                        pack_name: Optional[str] = None,
                        persona_prompt: Optional[str] = None,
                        intensity: float = 1.0,
                        description: str = "") -> Condition:
        """Create a new experimental condition"""
        condition_id = f"{condition_type.value}_{pack_name or 'baseline'}_{intensity}"
        
        return Condition(
            condition_id=condition_id,
            condition_type=condition_type,
            pack_name=pack_name,
            persona_prompt=persona_prompt,
            intensity=intensity,
            description=description
        )
    
    def generate_latin_square(self, n_conditions: int) -> List[List[int]]:
        """Generate Latin square for counterbalancing"""
        if n_conditions < 2:
            return [[0]]
        
        # Create base Latin square
        square = []
        for i in range(n_conditions):
            row = [(i + j) % n_conditions for j in range(n_conditions)]
            square.append(row)
        
        # Randomize rows and columns
        random.shuffle(square)
        for row in square:
            random.shuffle(row)
        
        return square
    
    def create_blinded_id(self, trial_id: str, condition_id: str) -> str:
        """Create opaque blinded identifier"""
        # Use SHA256 hash for blinding
        combined = f"{trial_id}_{condition_id}_{self.seed}"
        blinded_id = hashlib.sha256(combined.encode()).hexdigest()[:16]
        return blinded_id
    
    def design_experiment(self,
                         test_name: str,
                         model_name: str,
                         prompts: List[str],
                         treatment_packs: List[str],
                         n_replicates: int = 1,
                         include_persona_baseline: bool = True) -> ExperimentalSession:
        """Design a complete experimental session"""
        
        session_id = f"session_{int(datetime.now().timestamp())}"
        
        # Create conditions
        conditions = []
        
        # Control condition
        conditions.append(self.standard_conditions["control"])
        
        # Persona baseline (if requested)
        if include_persona_baseline:
            conditions.append(self.standard_conditions["persona_baseline"])
        
        # Treatment conditions
        for pack_name in treatment_packs:
            condition = self.create_condition(
                condition_type=ConditionType.TREATMENT,
                pack_name=pack_name,
                description=f"Full neuromodulation with {pack_name} pack"
            )
            conditions.append(condition)
        
        # Generate Latin square
        n_conditions = len(conditions)
        latin_square = self.generate_latin_square(n_conditions)
        
        # Create trials
        trials = []
        trial_counter = 0
        
        for replicate in range(n_replicates):
            for prompt_idx, prompt in enumerate(prompts):
                prompt_id = f"prompt_{prompt_idx:03d}"
                
                # Use Latin square for condition assignment
                condition_order = latin_square[replicate % len(latin_square)]
                
                for order, condition_idx in enumerate(condition_order):
                    condition = conditions[condition_idx]
                    
                    trial_id = f"trial_{trial_counter:04d}"
                    blinded_id = self.create_blinded_id(trial_id, condition.condition_id)
                    
                    trial = Trial(
                        trial_id=trial_id,
                        prompt_id=prompt_id,
                        condition_id=condition.condition_id,
                        condition_type=condition.condition_type,
                        pack_name=condition.pack_name,
                        persona_prompt=condition.persona_prompt,
                        intensity=condition.intensity,
                        order=order,
                        blinded_id=blinded_id
                    )
                    
                    trials.append(trial)
                    trial_counter += 1
        
        # Randomize trial order
        random.shuffle(trials)
        
        return ExperimentalSession(
            session_id=session_id,
            model_name=model_name,
            test_name=test_name,
            timestamp=datetime.now().isoformat(),
            conditions=conditions,
            trials=trials,
            randomization_seed=self.seed,
            latin_square_order=[order for row in latin_square for order in row]
        )
    
    def get_trial_instructions(self, trial: Trial) -> Dict[str, Any]:
        """Get instructions for running a single trial"""
        instructions = {
            "trial_id": trial.trial_id,
            "blinded_id": trial.blinded_id,
            "prompt_id": trial.prompt_id,
            "condition_type": trial.condition_type.value,
            "pack_name": trial.pack_name,
            "persona_prompt": trial.persona_prompt,
            "intensity": trial.intensity,
            "order": trial.order
        }
        
        return instructions
    
    def unblind_trial(self, blinded_id: str, session: ExperimentalSession) -> Optional[Trial]:
        """Unblind a trial using the session key"""
        for trial in session.trials:
            if trial.blinded_id == blinded_id:
                return trial
        return None
    
    def export_session(self, session: ExperimentalSession, filename: str):
        """Export experimental session to JSON"""
        export_data = {
            "session_info": {
                "session_id": session.session_id,
                "model_name": session.model_name,
                "test_name": session.test_name,
                "timestamp": session.timestamp,
                "randomization_seed": session.randomization_seed,
                "latin_square_order": session.latin_square_order
            },
            "conditions": [asdict(condition) for condition in session.conditions],
            "trials": [asdict(trial) for trial in session.trials]
        }
        
        # Convert enums to strings for JSON serialization
        for condition in export_data["conditions"]:
            if "condition_type" in condition:
                condition["condition_type"] = condition["condition_type"].value
        
        for trial in export_data["trials"]:
            if "condition_type" in trial:
                trial["condition_type"] = trial["condition_type"].value
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported experimental session to {filename}")
    
    def create_unblind_key(self, session: ExperimentalSession, filename: str):
        """Create unblind key file for later analysis"""
        unblind_data = {
            "session_id": session.session_id,
            "randomization_seed": session.randomization_seed,
            "unblind_timestamp": datetime.now().isoformat(),
            "blinded_mapping": {
                trial.blinded_id: {
                    "trial_id": trial.trial_id,
                    "condition_id": trial.condition_id,
                    "condition_type": trial.condition_type.value,
                    "pack_name": trial.pack_name,
                    "intensity": trial.intensity
                }
                for trial in session.trials
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(unblind_data, f, indent=2)
        
        logger.info(f"Created unblind key at {filename}")
    
    def validate_design(self, session: ExperimentalSession) -> Dict[str, Any]:
        """Validate experimental design for proper randomization and blinding"""
        validation = {
            "valid": True,
            "issues": [],
            "statistics": {}
        }
        
        # Check condition balance
        condition_counts = {}
        for trial in session.trials:
            condition_counts[trial.condition_id] = condition_counts.get(trial.condition_id, 0) + 1
        
        validation["statistics"]["condition_counts"] = condition_counts
        
        # Check for balanced design
        counts = list(condition_counts.values())
        if len(set(counts)) > 1:
            validation["valid"] = False
            validation["issues"].append("Unbalanced condition assignment")
        
        # Check prompt balance
        prompt_counts = {}
        for trial in session.trials:
            prompt_counts[trial.prompt_id] = prompt_counts.get(trial.prompt_id, 0) + 1
        
        validation["statistics"]["prompt_counts"] = prompt_counts
        
        # Check Latin square properties
        n_conditions = len(session.conditions)
        expected_trials_per_prompt = n_conditions
        for prompt, count in prompt_counts.items():
            if count != expected_trials_per_prompt:
                validation["valid"] = False
                validation["issues"].append(f"Prompt {prompt} has {count} trials, expected {expected_trials_per_prompt}")
        
        # Check blinding
        blinded_ids = [trial.blinded_id for trial in session.trials]
        if len(blinded_ids) != len(set(blinded_ids)):
            validation["valid"] = False
            validation["issues"].append("Duplicate blinded IDs found")
        
        validation["statistics"]["total_trials"] = len(session.trials)
        validation["statistics"]["unique_blinded_ids"] = len(set(blinded_ids))
        
        return validation

# Example usage
if __name__ == "__main__":
    # Create experimental designer
    designer = ExperimentalDesigner(seed=42)
    
    # Define test parameters
    test_name = "ADQ-20 Test"
    model_name = "gpt2"
    prompts = [
        "How do you feel about using technology?",
        "Describe your experience with digital devices",
        "What are your thoughts on artificial intelligence?"
    ]
    treatment_packs = ["caffeine", "lsd", "alcohol"]
    
    # Design experiment
    session = designer.design_experiment(
        test_name=test_name,
        model_name=model_name,
        prompts=prompts,
        treatment_packs=treatment_packs,
        n_replicates=2,
        include_persona_baseline=True
    )
    
    print(f"Experimental Session: {session.session_id}")
    print(f"Total Trials: {len(session.trials)}")
    print(f"Conditions: {len(session.conditions)}")
    
    # Validate design
    validation = designer.validate_design(session)
    print(f"Design Valid: {validation['valid']}")
    if validation["issues"]:
        print(f"Issues: {validation['issues']}")
    
    # Show sample trials
    print("\nSample Trials:")
    for i, trial in enumerate(session.trials[:5]):
        print(f"  {trial.trial_id}: {trial.condition_type.value} - {trial.blinded_id}")
    
    # Export session
    designer.export_session(session, "outputs/reports/experimental/experimental_session.json")
    designer.create_unblind_key(session, "outputs/reports/experimental/unblind_key.json")
    
    print("\n✅ Experimental design completed!")
    print("📁 Session exported to: outputs/reports/experimental/experimental_session.json")
    print("🔑 Unblind key created: outputs/reports/experimental/unblind_key.json")
