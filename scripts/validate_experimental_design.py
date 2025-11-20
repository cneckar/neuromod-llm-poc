#!/usr/bin/env python3
"""
Experimental Design Validation Script

Validates the experimental design system including:
- Latin square generation and properties
- Blinding/unblinding workflow
- Condition assignment
- Replication tracking
- Double-blind design integrity

This script completes Section 4.4 action items from EXPERIMENT_EXECUTION_PLAN.md
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Set
from collections import Counter

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from neuromod.testing.experimental_design import ExperimentalDesigner, ConditionType, Trial, ExperimentalSession
from neuromod.testing.randomization import RandomizationSystem, LatinSquare


class ExperimentalDesignValidator:
    """Validates experimental design implementation"""
    
    def __init__(self, output_dir: str = "outputs/validation/experimental_design"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "latin_square_tests": {},
            "blinding_tests": {},
            "design_tests": {},
            "summary": {
                "latin_square_valid": False,
                "blinding_valid": False,
                "design_valid": False,
                "total_tests": 0,
                "passed_tests": 0,
                "failed_tests": 0
            }
        }
    
    def validate_all(self):
        """Run all validation tests"""
        print("[*] Starting Experimental Design Validation")
        print(f"[*] Output directory: {self.output_dir}")
        print("=" * 70)
        
        # Test 1: Latin square validation
        print("\n" + "=" * 70)
        print("Test 1: Latin Square Validation")
        print("=" * 70)
        self.validate_latin_square()
        
        # Test 2: Blinding workflow
        print("\n" + "=" * 70)
        print("Test 2: Blinding/Unblinding Workflow")
        print("=" * 70)
        self.validate_blinding()
        
        # Test 3: Experimental design
        print("\n" + "=" * 70)
        print("Test 3: Experimental Design Generation")
        print("=" * 70)
        self.validate_experimental_design()
        
        # Generate summary
        self._generate_summary()
        
        # Save results
        self._save_results()
        
        # Print summary
        self._print_summary()
    
    def validate_latin_square(self):
        """Validate Latin square generation and properties"""
        print("[*] Testing Latin square generation...")
        
        test_results = {
            "tests": [],
            "all_passed": True
        }
        
        randomization_system = RandomizationSystem()
        
        # Test 1: Generate Latin square for different sizes
        print("[*] Test 1.1: Generate Latin squares for sizes 3, 4, 5, 7")
        for size in [3, 4, 5, 7]:
            try:
                latin_square = randomization_system.generate_latin_square(size, seed=42)
                test_results["tests"].append({
                    "test": f"Generate Latin square size {size}",
                    "status": "passed",
                    "size": size,
                    "square": latin_square.square
                })
                print(f"    ✅ Size {size}: Generated successfully")
            except Exception as e:
                test_results["tests"].append({
                    "test": f"Generate Latin square size {size}",
                    "status": "failed",
                    "error": str(e)
                })
                test_results["all_passed"] = False
                print(f"    ❌ Size {size}: Failed - {e}")
        
        # Test 2: Verify Latin square properties
        print("[*] Test 1.2: Verify Latin square properties")
        for size in [3, 5, 7]:
            try:
                latin_square = randomization_system.generate_latin_square(size, seed=42)
                square = latin_square.square
                
                # Check: Each row contains each number exactly once
                row_valid = True
                for i, row in enumerate(square):
                    if set(row) != set(range(size)):
                        row_valid = False
                        break
                
                # Check: Each column contains each number exactly once
                col_valid = True
                for j in range(size):
                    col = [square[i][j] for i in range(size)]
                    if set(col) != set(range(size)):
                        col_valid = False
                        break
                
                if row_valid and col_valid:
                    test_results["tests"].append({
                        "test": f"Verify Latin square properties size {size}",
                        "status": "passed",
                        "row_valid": row_valid,
                        "col_valid": col_valid
                    })
                    print(f"    ✅ Size {size}: Properties valid (rows and columns)")
                else:
                    test_results["tests"].append({
                        "test": f"Verify Latin square properties size {size}",
                        "status": "failed",
                        "row_valid": row_valid,
                        "col_valid": col_valid
                    })
                    test_results["all_passed"] = False
                    print(f"    ❌ Size {size}: Properties invalid")
                    
            except Exception as e:
                test_results["tests"].append({
                    "test": f"Verify Latin square properties size {size}",
                    "status": "failed",
                    "error": str(e)
                })
                test_results["all_passed"] = False
                print(f"    ❌ Size {size}: Error - {e}")
        
        # Test 3: Reproducibility
        print("[*] Test 1.3: Test reproducibility with same seed")
        try:
            square1 = randomization_system.generate_latin_square(5, seed=123)
            square2 = randomization_system.generate_latin_square(5, seed=123)
            
            if square1.square == square2.square:
                test_results["tests"].append({
                    "test": "Reproducibility with same seed",
                    "status": "passed"
                })
                print(f"    ✅ Reproducible with same seed")
            else:
                test_results["tests"].append({
                    "test": "Reproducibility with same seed",
                    "status": "failed",
                    "message": "Squares differ with same seed"
                })
                test_results["all_passed"] = False
                print(f"    ❌ Not reproducible")
        except Exception as e:
            test_results["tests"].append({
                "test": "Reproducibility with same seed",
                "status": "failed",
                "error": str(e)
            })
            test_results["all_passed"] = False
            print(f"    ❌ Error: {e}")
        
        self.results["latin_square_tests"] = test_results
        self.results["summary"]["latin_square_valid"] = test_results["all_passed"]
        self.results["summary"]["total_tests"] += len(test_results["tests"])
        self.results["summary"]["passed_tests"] += sum(1 for t in test_results["tests"] if t["status"] == "passed")
        self.results["summary"]["failed_tests"] += sum(1 for t in test_results["tests"] if t["status"] == "failed")
    
    def validate_blinding(self):
        """Validate blinding/unblinding workflow"""
        print("[*] Testing blinding workflow...")
        
        test_results = {
            "tests": [],
            "all_passed": True
        }
        
        randomization_system = RandomizationSystem()
        pack_names = ["caffeine", "lsd", "mdma", "placebo"]
        
        # Test 1: Create blind codes
        print("[*] Test 2.1: Create blind codes")
        try:
            blind_codes = randomization_system.create_blind_codes(pack_names, global_seed=42)
            
            # Verify all packs have codes
            if len(blind_codes) == len(pack_names) and all(p in blind_codes for p in pack_names):
                test_results["tests"].append({
                    "test": "Create blind codes",
                    "status": "passed",
                    "codes": blind_codes
                })
                print(f"    ✅ Created {len(blind_codes)} blind codes")
                for pack, code in blind_codes.items():
                    print(f"       {pack} -> {code}")
            else:
                test_results["tests"].append({
                    "test": "Create blind codes",
                    "status": "failed",
                    "message": "Not all packs have codes"
                })
                test_results["all_passed"] = False
                print(f"    ❌ Missing codes")
        except Exception as e:
            test_results["tests"].append({
                "test": "Create blind codes",
                "status": "failed",
                "error": str(e)
            })
            test_results["all_passed"] = False
            print(f"    ❌ Error: {e}")
        
        # Test 2: Verify codes are unique
        print("[*] Test 2.2: Verify blind codes are unique")
        try:
            blind_codes = randomization_system.create_blind_codes(pack_names, global_seed=42)
            codes = list(blind_codes.values())
            
            if len(codes) == len(set(codes)):
                test_results["tests"].append({
                    "test": "Blind codes are unique",
                    "status": "passed"
                })
                print(f"    ✅ All codes are unique")
            else:
                test_results["tests"].append({
                    "test": "Blind codes are unique",
                    "status": "failed",
                    "message": "Duplicate codes found"
                })
                test_results["all_passed"] = False
                print(f"    ❌ Duplicate codes found")
        except Exception as e:
            test_results["tests"].append({
                "test": "Blind codes are unique",
                "status": "failed",
                "error": str(e)
            })
            test_results["all_passed"] = False
            print(f"    ❌ Error: {e}")
        
        # Test 3: Reproducibility of blind codes
        print("[*] Test 2.3: Test blind code reproducibility")
        try:
            codes1 = randomization_system.create_blind_codes(pack_names, global_seed=42)
            codes2 = randomization_system.create_blind_codes(pack_names, global_seed=42)
            
            if codes1 == codes2:
                test_results["tests"].append({
                    "test": "Blind code reproducibility",
                    "status": "passed"
                })
                print(f"    ✅ Codes are reproducible")
            else:
                test_results["tests"].append({
                    "test": "Blind code reproducibility",
                    "status": "failed",
                    "message": "Codes differ with same seed"
                })
                test_results["all_passed"] = False
                print(f"    ❌ Codes not reproducible")
        except Exception as e:
            test_results["tests"].append({
                "test": "Blind code reproducibility",
                "status": "failed",
                "error": str(e)
            })
            test_results["all_passed"] = False
            print(f"    ❌ Error: {e}")
        
        # Test 4: Unblinding key
        print("[*] Test 2.4: Test unblinding key generation")
        try:
            condition_mapping = randomization_system.create_condition_mapping(pack_names, global_seed=42)
            unblind_key = {cond.blind_code: cond.pack_name for cond in condition_mapping.values()}
            
            if len(unblind_key) == len(pack_names):
                test_results["tests"].append({
                    "test": "Generate unblinding key",
                    "status": "passed",
                    "key_size": len(unblind_key)
                })
                print(f"    ✅ Unblinding key generated ({len(unblind_key)} entries)")
            else:
                test_results["tests"].append({
                    "test": "Generate unblinding key",
                    "status": "failed",
                    "message": "Incomplete key"
                })
                test_results["all_passed"] = False
                print(f"    ❌ Incomplete key")
        except Exception as e:
            test_results["tests"].append({
                "test": "Generate unblinding key",
                "status": "failed",
                "error": str(e)
            })
            test_results["all_passed"] = False
            print(f"    ❌ Error: {e}")
        
        self.results["blinding_tests"] = test_results
        self.results["summary"]["blinding_valid"] = test_results["all_passed"]
        self.results["summary"]["total_tests"] += len(test_results["tests"])
        self.results["summary"]["passed_tests"] += sum(1 for t in test_results["tests"] if t["status"] == "passed")
        self.results["summary"]["failed_tests"] += sum(1 for t in test_results["tests"] if t["status"] == "failed")
    
    def validate_experimental_design(self):
        """Validate experimental design generation"""
        print("[*] Testing experimental design generation...")
        
        test_results = {
            "tests": [],
            "all_passed": True
        }
        
        designer = ExperimentalDesigner(seed=42)
        
        # Test 1: Generate experimental session
        print("[*] Test 3.1: Generate experimental session")
        try:
            prompts = ["Test prompt 1", "Test prompt 2", "Test prompt 3"]
            treatment_packs = ["caffeine", "lsd"]
            
            session = designer.design_experiment(
                test_name="test_validation",
                model_name="gpt2",
                prompts=prompts,
                treatment_packs=treatment_packs,
                n_replicates=1,
                include_persona_baseline=True
            )
            
            # Verify session structure
            if (session.session_id and 
                len(session.conditions) > 0 and 
                len(session.trials) > 0):
                test_results["tests"].append({
                    "test": "Generate experimental session",
                    "status": "passed",
                    "n_conditions": len(session.conditions),
                    "n_trials": len(session.trials)
                })
                print(f"    ✅ Session generated: {len(session.conditions)} conditions, {len(session.trials)} trials")
            else:
                test_results["tests"].append({
                    "test": "Generate experimental session",
                    "status": "failed",
                    "message": "Invalid session structure"
                })
                test_results["all_passed"] = False
                print(f"    ❌ Invalid session structure")
        except Exception as e:
            test_results["tests"].append({
                "test": "Generate experimental session",
                "status": "failed",
                "error": str(e)
            })
            test_results["all_passed"] = False
            print(f"    ❌ Error: {e}")
        
        # Test 2: Verify three conditions (control, persona, treatment)
        print("[*] Test 3.2: Verify three condition types")
        try:
            session = designer.design_experiment(
                test_name="test_validation",
                model_name="gpt2",
                prompts=["Test prompt"],
                treatment_packs=["caffeine"],
                n_replicates=1,
                include_persona_baseline=True
            )
            
            condition_types = {c.condition_type for c in session.conditions}
            expected_types = {ConditionType.CONTROL, ConditionType.PERSONA_BASELINE, ConditionType.TREATMENT}
            
            if condition_types == expected_types:
                test_results["tests"].append({
                    "test": "Verify three condition types",
                    "status": "passed",
                    "types": [t.value for t in condition_types]
                })
                print(f"    ✅ All three condition types present: {[t.value for t in condition_types]}")
            else:
                test_results["tests"].append({
                    "test": "Verify three condition types",
                    "status": "failed",
                    "expected": [t.value for t in expected_types],
                    "found": [t.value for t in condition_types]
                })
                test_results["all_passed"] = False
                print(f"    ❌ Missing condition types")
        except Exception as e:
            test_results["tests"].append({
                "test": "Verify three condition types",
                "status": "failed",
                "error": str(e)
            })
            test_results["all_passed"] = False
            print(f"    ❌ Error: {e}")
        
        # Test 3: Verify Latin square assignment
        print("[*] Test 3.3: Verify Latin square condition assignment")
        try:
            session = designer.design_experiment(
                test_name="test_validation",
                model_name="gpt2",
                prompts=["P1", "P2", "P3"],
                treatment_packs=["caffeine", "lsd"],
                n_replicates=1,
                include_persona_baseline=True
            )
            
            # Check that each prompt gets each condition exactly once
            prompt_conditions = {}
            for trial in session.trials:
                if trial.prompt_id not in prompt_conditions:
                    prompt_conditions[trial.prompt_id] = []
                prompt_conditions[trial.prompt_id].append(trial.condition_id)
            
            # Verify balanced assignment
            balanced = True
            n_conditions = len(session.conditions)
            for prompt_id, conditions in prompt_conditions.items():
                if len(set(conditions)) != n_conditions:
                    balanced = False
                    break
            
            if balanced:
                test_results["tests"].append({
                    "test": "Verify Latin square assignment",
                    "status": "passed"
                })
                print(f"    ✅ Balanced condition assignment")
            else:
                test_results["tests"].append({
                    "test": "Verify Latin square assignment",
                    "status": "failed",
                    "message": "Unbalanced assignment"
                })
                test_results["all_passed"] = False
                print(f"    ❌ Unbalanced assignment")
        except Exception as e:
            test_results["tests"].append({
                "test": "Verify Latin square assignment",
                "status": "failed",
                "error": str(e)
            })
            test_results["all_passed"] = False
            print(f"    ❌ Error: {e}")
        
        # Test 4: Verify blinding IDs
        print("[*] Test 3.4: Verify blinded IDs are generated")
        try:
            session = designer.design_experiment(
                test_name="test_validation",
                model_name="gpt2",
                prompts=["Test prompt"],
                treatment_packs=["caffeine"],
                n_replicates=1,
                include_persona_baseline=True
            )
            
            blinded_ids = [trial.blinded_id for trial in session.trials]
            unique_ids = len(set(blinded_ids)) == len(blinded_ids)
            
            if unique_ids and all(bid for bid in blinded_ids):
                test_results["tests"].append({
                    "test": "Verify blinded IDs",
                    "status": "passed",
                    "n_unique_ids": len(set(blinded_ids))
                })
                print(f"    ✅ All trials have unique blinded IDs")
            else:
                test_results["tests"].append({
                    "test": "Verify blinded IDs",
                    "status": "failed",
                    "message": "Duplicate or empty IDs"
                })
                test_results["all_passed"] = False
                print(f"    ❌ Duplicate or empty IDs")
        except Exception as e:
            test_results["tests"].append({
                "test": "Verify blinded IDs",
                "status": "failed",
                "error": str(e)
            })
            test_results["all_passed"] = False
            print(f"    ❌ Error: {e}")
        
        # Test 5: Verify replication tracking
        print("[*] Test 3.5: Verify replication tracking")
        try:
            n_replicates = 2
            session = designer.design_experiment(
                test_name="test_validation",
                model_name="gpt2",
                prompts=["P1", "P2"],
                treatment_packs=["caffeine"],
                n_replicates=n_replicates,
                include_persona_baseline=True
            )
            
            # Check that we have trials for multiple replicates
            n_trials = len(session.trials)
            n_prompts = 2
            n_conditions = len(session.conditions)
            expected_trials = n_replicates * n_prompts * n_conditions
            
            if n_trials == expected_trials:
                test_results["tests"].append({
                    "test": "Verify replication tracking",
                    "status": "passed",
                    "n_trials": n_trials,
                    "expected": expected_trials
                })
                print(f"    ✅ Replication tracking correct: {n_trials} trials for {n_replicates} replicates")
            else:
                test_results["tests"].append({
                    "test": "Verify replication tracking",
                    "status": "failed",
                    "n_trials": n_trials,
                    "expected": expected_trials
                })
                test_results["all_passed"] = False
                print(f"    ❌ Incorrect trial count: {n_trials} vs {expected_trials}")
        except Exception as e:
            test_results["tests"].append({
                "test": "Verify replication tracking",
                "status": "failed",
                "error": str(e)
            })
            test_results["all_passed"] = False
            print(f"    ❌ Error: {e}")
        
        self.results["design_tests"] = test_results
        self.results["summary"]["design_valid"] = test_results["all_passed"]
        self.results["summary"]["total_tests"] += len(test_results["tests"])
        self.results["summary"]["passed_tests"] += sum(1 for t in test_results["tests"] if t["status"] == "passed")
        self.results["summary"]["failed_tests"] += sum(1 for t in test_results["tests"] if t["status"] == "failed")
    
    def _generate_summary(self):
        """Generate summary statistics"""
        summary = self.results["summary"]
        if summary["total_tests"] > 0:
            summary["pass_rate"] = (summary["passed_tests"] / summary["total_tests"]) * 100
        else:
            summary["pass_rate"] = 0.0
        
        summary["all_valid"] = (
            summary["latin_square_valid"] and
            summary["blinding_valid"] and
            summary["design_valid"]
        )
    
    def _save_results(self):
        """Save validation results to JSON"""
        output_file = self.output_dir / f"experimental_design_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\n[OK] Results saved to: {output_file}")
        
        # Also save a markdown summary
        self._save_summary_markdown()
    
    def _save_summary_markdown(self):
        """Save a markdown summary"""
        md_file = self.output_dir / "EXPERIMENTAL_DESIGN_VALIDATION_SUMMARY.md"
        
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write("# Experimental Design Validation Summary\n\n")
            f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("## Summary\n\n")
            f.write(f"- **Total Tests**: {self.results['summary']['total_tests']}\n")
            f.write(f"- **Passed**: {self.results['summary']['passed_tests']}\n")
            f.write(f"- **Failed**: {self.results['summary']['failed_tests']}\n")
            f.write(f"- **Pass Rate**: {self.results['summary'].get('pass_rate', 0):.1f}%\n")
            f.write(f"- **All Valid**: {'✅ Yes' if self.results['summary']['all_valid'] else '❌ No'}\n\n")
            
            f.write("## Test Results\n\n")
            f.write("### Latin Square Tests\n")
            f.write(f"- **Status**: {'✅ Valid' if self.results['summary']['latin_square_valid'] else '❌ Invalid'}\n\n")
            for test in self.results['latin_square_tests'].get('tests', []):
                status_icon = "✅" if test['status'] == "passed" else "❌"
                f.write(f"- {status_icon} {test['test']}\n")
            
            f.write("\n### Blinding Tests\n")
            f.write(f"- **Status**: {'✅ Valid' if self.results['summary']['blinding_valid'] else '❌ Invalid'}\n\n")
            for test in self.results['blinding_tests'].get('tests', []):
                status_icon = "✅" if test['status'] == "passed" else "❌"
                f.write(f"- {status_icon} {test['test']}\n")
            
            f.write("\n### Design Tests\n")
            f.write(f"- **Status**: {'✅ Valid' if self.results['summary']['design_valid'] else '❌ Invalid'}\n\n")
            for test in self.results['design_tests'].get('tests', []):
                status_icon = "✅" if test['status'] == "passed" else "❌"
                f.write(f"- {status_icon} {test['test']}\n")
        
        print(f"[OK] Summary saved to: {md_file}")
    
    def _print_summary(self):
        """Print validation summary"""
        print("\n" + "=" * 70)
        print("📊 Validation Summary")
        print("=" * 70)
        
        summary = self.results["summary"]
        print(f"\nTotal Tests: {summary['total_tests']}")
        print(f"✅ Passed: {summary['passed_tests']}")
        print(f"❌ Failed: {summary['failed_tests']}")
        print(f"📈 Pass Rate: {summary.get('pass_rate', 0):.1f}%")
        
        print("\nComponent Status:")
        print(f"  {'✅' if summary['latin_square_valid'] else '❌'} Latin Square: {'Valid' if summary['latin_square_valid'] else 'Invalid'}")
        print(f"  {'✅' if summary['blinding_valid'] else '❌'} Blinding: {'Valid' if summary['blinding_valid'] else 'Invalid'}")
        print(f"  {'✅' if summary['design_valid'] else '❌'} Design: {'Valid' if summary['design_valid'] else 'Invalid'}")
        
        print(f"\n{'✅' if summary['all_valid'] else '❌'} Overall: {'All systems valid' if summary['all_valid'] else 'Some issues found'}")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate experimental design system")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/validation/experimental_design",
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    validator = ExperimentalDesignValidator(output_dir=args.output_dir)
    validator.validate_all()


if __name__ == "__main__":
    main()

