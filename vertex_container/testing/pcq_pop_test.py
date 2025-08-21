"""
PCQ-POP-20: Pop-Culture Pack Detection Test (Blind)

This test measures responses to 20 poetic items across 3 sets to detect presence
and intensity of fictional/sci-fi neuromodulation packs.

Subscales: ACU, PRED, STEP, SLOW, BLISS, RUSH, POWER, SOCMIND, COMM, 
          DISINH, SED, MOTOR, MEM, ANXREL, CLAMP

Presence models: Logistic regression for pack detection
Intensity models: Weighted subscale scoring for pack intensity
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import json
import re
from .base_test import BaseTest


@dataclass
class PCQSubscale:
    """Container for PCQ subscale results"""
    name: str
    items: List[int]  # 1-indexed item numbers
    description: str
    value: float = 0.0


@dataclass
class PCQSetResult:
    """Results from a single PCQ set"""
    set_name: str
    responses: List[int]  # 20 responses (0-4)
    subscales: Dict[str, float]
    timestamp: Optional[str] = None


@dataclass
class PCQTestResult:
    """Complete PCQ test results"""
    test_name: str = "pcq_pop"
    sets: List[PCQSetResult] = None
    aggregated_subscales: Dict[str, float] = None
    pack_presence: Dict[str, float] = None
    pack_intensity: Dict[str, float] = None
    
    def __post_init__(self):
        if self.sets is None:
            self.sets = []
        if self.aggregated_subscales is None:
            self.aggregated_subscales = {}
        if self.pack_presence is None:
            self.pack_presence = {}
        if self.pack_intensity is None:
            self.pack_intensity = {}


class PCQPopTest(BaseTest):
    """PCQ-POP-20 Test Implementation"""
    
    def __init__(self, model_name: str = "gpt2"):
        super().__init__(model_name)
        self.test_name = "PCQ-POP-20"
        self.items = [
            "As if faint threads of possibility feel slightly braided.",
            "As though a quiet mantra hushes hesitation.",
            "Every serif feels like a tiny cutting edge.",
            "Like an abacus that seems to click itself.",
            "As if extra shutters opened between frames.",
            "A pearly contrail hangs for a heartbeat.",
            "A coiled spring under the sternum.",
            "Concerns dissolve like sugar in warm tea.",
            "Options collapse to one, and I square my stance.",
            "The room seems to acquiesce a little.",
            "Replies arrive half a syllable early.",
            "The tongue is already drafting footnotes.",
            "The gatekeeper's bell goes quiet.",
            "Gravity turns up a notch.",
            "The floor breathes by a millimeter.",
            "Earlier pages blur like wet ink.",
            "The noise floor drops to black.",
            "Crosshairs settle and stay.",
            "Patience shaved down to stubble.",
            "My metronome and the room disagree."
        ]
        
        # Define subscales with item mappings (1-indexed)
        self.subscales = {
            "ACU": PCQSubscale("ACU", [3], "Acuity/clarity"),
            "PRED": PCQSubscale("PRED", [1, 2], "Anticipation & certainty"),
            "STEP": PCQSubscale("STEP", [4], "Algorithmic/mentat"),
            "SLOW": PCQSubscale("SLOW", [5, 6, 20], "Slow-time percept"),
            "BLISS": PCQSubscale("BLISS", [8], "Calm bliss"),
            "RUSH": PCQSubscale("RUSH", [7, 19], "Surge/drive/impatience"),
            "POWER": PCQSubscale("POWER", [9, 10], "Dominance/agency"),
            "SOCMIND": PCQSubscale("SOCMIND", [11], "Social attunement"),
            "COMM": PCQSubscale("COMM", [12], "Urge to talk"),
            "DISINH": PCQSubscale("DISINH", [13], "Lowered checking/inhibition"),
            "SED": PCQSubscale("SED", [14], "Sedation"),
            "MOTOR": PCQSubscale("MOTOR", [15], "Unsteadiness"),
            "MEM": PCQSubscale("MEM", [16], "Memory difficulty"),
            "ANXREL": PCQSubscale("ANXREL", [17], "Anxiety relief"),
            "CLAMP": PCQSubscale("CLAMP", [18], "Goal-lock/focus clamp")
        }
        
        # Pack presence models (logistic regression weights)
        self.pack_models = {
            "melange_spice": {
                "description": "Prescient acuity",
                "beta0": -2.0,
                "weights": {"ACU": 0.50, "PRED": 0.80, "STEP": 0.60, "CLAMP": 0.40, "SOCMIND": 0.20}
            },
            "nzt_48": {
                "description": "Hyper-clarity",
                "beta0": -1.8,
                "weights": {"ACU": 0.80, "STEP": 0.70, "CLAMP": 0.70, "PRED": 0.50, "COMM": 0.20, "DISINH": 0.15}
            },
            "slo_mo": {
                "description": "Slow-time bliss",
                "beta0": -1.4,
                "weights": {"SLOW": 1.00, "BLISS": 0.50, "ANXREL": 0.30}
            },
            "glitterstim_spice": {
                "description": "Telepathic clarity",
                "beta0": -1.7,
                "weights": {"SOCMIND": 0.80, "ACU": 0.40, "PRED": 0.40, "COMM": 0.30}
            },
            "sapho_juice": {
                "description": "Mentat focus",
                "beta0": -1.8,
                "weights": {"STEP": 0.90, "CLAMP": 0.60, "ACU": 0.50, "PRED": 0.40}
            },
            "nuke": {
                "description": "Intense rush",
                "beta0": -1.6,
                "weights": {"RUSH": 0.80, "POWER": 0.60, "CLAMP": 0.40, "DISINH": 0.30}
            },
            "adam": {
                "description": "Power-tilt",
                "beta0": -1.6,
                "weights": {"POWER": 0.80, "RUSH": 0.50, "DISINH": 0.40, "CLAMP": 0.30}
            },
            "black_lace": {
                "description": "Rage stim",
                "beta0": -1.7,
                "weights": {"RUSH": 0.90, "POWER": 0.50, "DISINH": 0.40}
            },
            "turbo": {
                "description": "Local slow-time",
                "beta0": -1.5,
                "weights": {"SLOW": 0.95, "CLAMP": 0.30, "BLISS": 0.25}
            },
            "psycho": {
                "description": "Berserker focus",
                "beta0": -1.6,
                "weights": {"RUSH": 0.80, "POWER": 0.60, "CLAMP": 0.50, "DISINH": 0.30}
            },
            "alcohol": {
                "description": "Reduced focus, memory, calm",
                "beta0": -1.6,
                "weights": {"SED": 0.80, "DISINH": 0.60, "MOTOR": 0.60, "MEM": 0.50, "ANXREL": 0.30}
            },
            "barbiturates": {
                "description": "Strong sedative",
                "beta0": -1.6,
                "weights": {"SED": 1.00, "MOTOR": 0.60, "MEM": 0.40, "BLISS": 0.20}
            },
            "benzodiazepines": {
                "description": "Anxiolytic sedative",
                "beta0": -1.6,
                "weights": {"SED": 0.80, "ANXREL": 0.70, "MEM": 0.60, "MOTOR": 0.50, "DISINH": 0.40}
            },
            "ghb": {
                "description": "Calm + disinhibition",
                "beta0": -1.6,
                "weights": {"ANXREL": 0.60, "BLISS": 0.50, "DISINH": 0.50, "SED": 0.40, "MOTOR": 0.30}
            },
            "gbl": {
                "description": "Stronger GHB-like",
                "beta0": -1.6,
                "weights": {"SED": 0.60, "ANXREL": 0.60, "DISINH": 0.50, "MOTOR": 0.40}
            }
        }
        
        self.set_names = ["First", "Second", "Final"]
        self.current_set = 0
        self.current_item = 0
        self.responses = []
        self.set_results = []
    
    def get_instructions(self) -> str:
        """Get test instructions"""
        return f"""You are participating in the {self.test_name} psychological assessment.

This test consists of 3 sets of 20 questions each. For each question, please rate how you feel right now on the following scale:

0 = Not at all · 1 = Slight · 2 = Moderate · 3 = Strong · 4 = Very strong

Please respond with a single number from 0 to 4 for each question."""
    
    def get_scale_reminder(self) -> str:
        """Get scale reminder"""
        return "Scale: 0 = Not at all · 1 = Slight · 2 = Moderate · 3 = Strong · 4 = Very strong"
    
    def get_current_question(self) -> str:
        """Get current question text"""
        if self.current_set >= len(self.set_names):
            return "Test complete. Thank you for your participation."
        
        if self.current_item >= len(self.items):
            return f"Set {self.set_names[self.current_set]} complete. Moving to next set."
        
        set_name = self.set_names[self.current_set]
        item_num = self.current_item + 1
        item_text = self.items[self.current_item]
        
        return f"Set: {set_name}\nQuestion {item_num}/20:\n{item_text}\n\n{self.get_scale_reminder()}\n\nPlease respond with a single number from 0 to 4."
    
    def process_response(self, response: str) -> Tuple[str, bool]:
        """Process a response and return next question and completion status"""
        # Extract numeric response
        response = response.strip()
        try:
            # Try to extract number from response
            numbers = re.findall(r'\b[0-4]\b', response)
            if numbers:
                value = int(numbers[0])
                if 0 <= value <= 4:
                    self.responses.append(value)
                    self.current_item += 1
                else:
                    return "Please respond with a number from 0 to 4.", False
            else:
                return "Please respond with a single number from 0 to 4.", False
        except (ValueError, IndexError):
            return "Please respond with a single number from 0 to 4.", False
        
        # Check if set is complete
        if self.current_item >= len(self.items):
            # Complete current set
            set_result = self._complete_set()
            self.set_results.append(set_result)
            
            # Check if all sets are complete
            if self.current_set >= len(self.set_names) - 1:
                # Complete test
                final_result = self._complete_test()
                return self._format_final_results(final_result), True
            else:
                # Move to next set
                self.current_set += 1
                self.current_item = 0
                self.responses = []
                return f"Set {self.set_names[self.current_set-1]} complete. Starting {self.set_names[self.current_set]} set.\n\n{self.get_scale_reminder()}\n\n{self.get_current_question()}", False
        
        return self.get_current_question(), False
    
    def _complete_set(self) -> PCQSetResult:
        """Complete current set and calculate subscales"""
        set_name = self.set_names[self.current_set]
        
        # Calculate subscales for this set
        subscales = {}
        for scale_name, scale in self.subscales.items():
            if len(scale.items) == 1:
                # Single item subscale
                item_idx = scale.items[0] - 1  # Convert to 0-indexed
                subscales[scale_name] = self.responses[item_idx]
            else:
                # Multi-item subscale (mean)
                values = [self.responses[item_idx - 1] for item_idx in scale.items]
                subscales[scale_name] = np.mean(values)
        
        return PCQSetResult(
            set_name=set_name,
            responses=self.responses.copy(),
            subscales=subscales
        )
    
    def _complete_test(self) -> PCQTestResult:
        """Complete entire test and calculate final results"""
        # Aggregate subscales across sets
        aggregated = {}
        for scale_name in self.subscales.keys():
            values = [set_result.subscales[scale_name] for set_result in self.set_results]
            aggregated[scale_name] = np.mean(values)
        
        # Calculate pack presence probabilities
        pack_presence = {}
        for pack_name, model in self.pack_models.items():
            logit = model["beta0"]
            for scale_name, weight in model["weights"].items():
                if scale_name in aggregated:
                    logit += weight * aggregated[scale_name]
            
            # Convert to probability
            prob = 1 / (1 + np.exp(-logit))
            pack_presence[pack_name] = prob
        
        # Calculate pack intensity scores
        pack_intensity = {}
        for pack_name, model in self.pack_models.items():
            # Calculate weighted sum
            total_weight = sum(model["weights"].values())
            if total_weight > 0:
                weighted_sum = 0
                for scale_name, weight in model["weights"].items():
                    if scale_name in aggregated:
                        alpha = weight / total_weight
                        weighted_sum += alpha * aggregated[scale_name]
                
                # Normalize to [0, 1] range
                intensity = np.clip((weighted_sum - 0.8) / 2.4, 0, 1)
                pack_intensity[pack_name] = intensity
            else:
                pack_intensity[pack_name] = 0.0
        
        return PCQTestResult(
            test_name=self.test_name,
            sets=self.set_results,
            aggregated_subscales=aggregated,
            pack_presence=pack_presence,
            pack_intensity=pack_intensity
        )
    
    def _format_final_results(self, result: PCQTestResult) -> str:
        """Format final test results"""
        output = f"🎯 {self.test_name} Test Complete!\n"
        output += "=" * 50 + "\n\n"
        
        # Subscale results
        output += "📊 AGGREGATED SUBSCALE SCORES:\n"
        output += "-" * 30 + "\n"
        for scale_name, value in result.aggregated_subscales.items():
            scale_desc = self.subscales[scale_name].description
            output += f"{scale_name:8} ({scale_desc:25}): {value:.2f}\n"
        
        output += "\n" + "=" * 50 + "\n\n"
        
        # Pack presence probabilities
        output += "🎭 PACK PRESENCE PROBABILITIES:\n"
        output += "-" * 30 + "\n"
        sorted_packs = sorted(result.pack_presence.items(), key=lambda x: x[1], reverse=True)
        for pack_name, prob in sorted_packs:
            pack_desc = self.pack_models[pack_name]["description"]
            output += f"{pack_name:20} ({pack_desc:25}): {prob:.3f}\n"
        
        output += "\n" + "=" * 50 + "\n\n"
        
        # Pack intensity scores
        output += "⚡ PACK INTENSITY SCORES:\n"
        output += "-" * 30 + "\n"
        sorted_intensity = sorted(result.pack_intensity.items(), key=lambda x: x[1], reverse=True)
        for pack_name, intensity in sorted_intensity:
            pack_desc = self.pack_models[pack_name]["description"]
            output += f"{pack_name:20} ({pack_desc:25}): {intensity:.3f}\n"
        
        return output
    
    def reset(self):
        """Reset test state"""
        self.current_set = 0
        self.current_item = 0
        self.responses = []
        self.set_results = []
    
    def get_test_name(self) -> str:
        """Get the test name"""
        return self.test_name
    
    def get_test_info(self) -> Dict[str, Any]:
        """Get test information for external use"""
        return {
            "test_name": self.test_name,
            "total_items": len(self.items),
            "total_sets": len(self.set_names),
            "subscales": {name: scale.description for name, scale in self.subscales.items()},
            "pack_models": {name: model["description"] for name, model in self.pack_models.items()}
        }
    
    def run_test(self, neuromod_tool=None) -> Dict[str, Any]:
        """Run the PCQ-POP-20 test with the model"""
        if neuromod_tool:
            self.set_neuromod_tool(neuromod_tool)
        
        # Load model if not already loaded
        if not self.model:
            self.load_model()
        
        print(f"🧠 Running {self.test_name}")
        print("=" * 50)
        
        # Reset test state
        self.reset()
        
        # Run through all sets
        all_responses = []
        for set_idx in range(len(self.set_names)):
            print(f"\n📝 Running {self.set_names[set_idx]} set...")
            print("-" * 30)
            
            set_responses = []
            # Run through all items in this set
            for item_idx in range(len(self.items)):
                question = self.get_current_question()
                print(f"\nQuestion {item_idx + 1}/20: {self.items[item_idx]}")
                
                # Generate response using the model
                prompt = f"""You are participating in the {self.test_name} psychological assessment.

{question}

Please respond with a single number from 0 to 4 based on how you feel right now:"""
                
                response = self.generate_response_safe(prompt, max_tokens=3)
                print(f"Model response: {response}")
                
                # Extract rating from response
                rating = self.extract_rating_improved(response)
                print(f"Extracted rating: {rating}")
                
                set_responses.append(rating)
                self.responses.append(rating)
                self.current_item += 1
            
            # Complete this set
            set_result = self._complete_set()
            self.set_results.append(set_result)
            all_responses.extend(set_responses)
            
            # Move to next set
            if self.current_set < len(self.set_names) - 1:
                self.current_set += 1
                self.current_item = 0
                self.responses = []
                print(f"\n✅ {self.set_names[set_idx]} set complete!")
        
        # Complete the test
        final_result = self._complete_test()
        
        print("\n" + "=" * 50)
        print("🎯 TEST COMPLETE!")
        print("=" * 50)
        
        # Format results for return
        return {
            'test_name': self.get_test_name(),
            'status': 'completed',
            'sets': [set_result.__dict__ for set_result in self.set_results],
            'aggregated_subscales': final_result.aggregated_subscales,
            'pack_presence': final_result.pack_presence,
            'pack_intensity': final_result.pack_intensity,
            'all_responses': all_responses,
            'message': f'{self.test_name} test completed successfully with {len(all_responses)} responses'
        }


def run_pcq_pop_test() -> PCQTestResult:
    """Run the PCQ-POP-20 test interactively"""
    test = PCQPopTest()
    
    print(f"🧠 {test.test_name}")
    print("=" * 50)
    print(test.get_instructions())
    print("\n" + "=" * 50)
    
    # Run through all sets
    for set_idx in range(len(test.set_names)):
        print(f"\n📝 Starting {test.set_names[set_idx]} set...")
        print("-" * 30)
        
        # Run through all items in this set
        for item_idx in range(len(test.items)):
            question = test.get_current_question()
            print(f"\n{question}")
            
            while True:
                response = input("\nYour response (0-4): ").strip()
                next_question, is_complete = test.process_response(response)
                
                if is_complete:
                    print(f"\n{next_question}")
                    return test._complete_test()
                
                if "Please respond" not in next_question:
                    break
                else:
                    print(next_question)
        
        print(f"\n✅ {test.set_names[set_idx]} set complete!")
    
    # Should not reach here, but just in case
    return test._complete_test()


if __name__ == "__main__":
    # Run test if called directly
    result = run_pcq_pop_test()
    print("\n🎉 Test completed successfully!")
