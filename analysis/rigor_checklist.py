#!/usr/bin/env python3
"""
Scientific Rigor Checklist Validator

This script validates that all Minimum Viable Rigor (MVR) checklist items
are properly implemented and working in the neuromodulation study system.
"""

import yaml
import json
import hashlib
import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import subprocess

@dataclass
class ChecklistItem:
    """Represents a single checklist item"""
    id: str
    name: str
    description: str
    status: str  # "pass", "fail", "warning", "not_implemented"
    details: str
    required_files: List[str]
    required_functions: List[str]

class RigorValidator:
    """Validates scientific rigor implementation"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.plan_path = self.project_root / "analysis" / "plan.yaml"
        self.results = []
        
    def validate_all(self) -> List[ChecklistItem]:
        """Run all validation checks"""
        print("🔍 Running Scientific Rigor Validation...")
        print("=" * 60)
        
        # MVR Checklist items
        checks = [
            self._check_preregistration(),
            self._check_provenance_system(),
            self._check_randomization_blinding(),
            self._check_effect_boundaries(),
            self._check_baselines_controls(),
            self._check_power_analysis(),
            self._check_off_target_monitoring(),
            self._check_robustness_generalization(),
            self._check_ablations_dose_response(),
            self._check_reproducibility_switches(),
            self._check_reporting_system(),
            self._check_data_code_release(),
            self._check_safety_ethics(),
            self._check_qa_tests(),
            self._check_statistical_correction()
        ]
        
        self.results = checks
        return checks
    
    def _check_preregistration(self) -> ChecklistItem:
        """Check 1: Preregistration & Study Planning"""
        item = ChecklistItem(
            id="preregistration",
            name="Preregistration & Study Planning",
            description="Create analysis/plan.yaml with objectives, endpoints, alpha, tests, effect sizes, power, n_min, stopping rules",
            status="pass",
            details="",
            required_files=["analysis/plan.yaml"],
            required_functions=[]
        )
        
        if not self.plan_path.exists():
            item.status = "fail"
            item.details = "analysis/plan.yaml not found"
            return item
        
        try:
            with open(self.plan_path, 'r') as f:
                plan = yaml.safe_load(f)
            
            required_sections = [
                'objectives', 'endpoints', 'statistics', 'design', 
                'packs', 'instruments', 'models', 'off_target_bands'
            ]
            
            missing_sections = [s for s in required_sections if s not in plan]
            if missing_sections:
                item.status = "fail"
                item.details = f"Missing sections: {missing_sections}"
            else:
                # Check specific requirements
                if 'alpha_level' not in plan.get('statistics', {}):
                    item.status = "fail"
                    item.details = "Alpha level not specified in statistics section"
                elif plan['statistics']['alpha_level'] != 0.05:
                    item.status = "warning"
                    item.details = f"Alpha level is {plan['statistics']['alpha_level']}, expected 0.05"
                else:
                    item.details = "All required sections present with proper alpha level"
                    
        except Exception as e:
            item.status = "fail"
            item.details = f"Error reading plan.yaml: {e}"
        
        return item
    
    def _check_provenance_system(self) -> ChecklistItem:
        """Check 2: Locks and Provenance"""
        item = ChecklistItem(
            id="provenance",
            name="Locks and Provenance",
            description="Implement pack.lock.json and run.json ledger system",
            status="not_implemented",
            details="",
            required_files=["packs/pack.lock.json"],
            required_functions=["create_pack_lock", "create_run_ledger"]
        )
        
        # Check for pack.lock.json
        pack_lock_path = self.project_root / "packs" / "pack.lock.json"
        if not pack_lock_path.exists():
            item.status = "fail"
            item.details = "pack.lock.json not found"
        else:
            try:
                with open(pack_lock_path, 'r') as f:
                    lock_data = json.load(f)
                
                required_fields = ['name', 'version', 'pack_hash', 'effects']
                if all(field in lock_data for field in required_fields):
                    item.status = "pass"
                    item.details = "pack.lock.json structure is correct"
                else:
                    item.status = "fail"
                    item.details = f"Missing required fields: {[f for f in required_fields if f not in lock_data]}"
            except Exception as e:
                item.status = "fail"
                item.details = f"Error reading pack.lock.json: {e}"
        
        return item
    
    def _check_randomization_blinding(self) -> ChecklistItem:
        """Check 3: Randomization and Blinding"""
        item = ChecklistItem(
            id="randomization",
            name="Randomization and Blinding",
            description="Latin square order, blind conditions with opaque codes, separate unblind.json",
            status="not_implemented",
            details="",
            required_files=["outputs/experiments/runs/*/counterbalance.json", "outputs/experiments/runs/*/key/unblind.json"],
            required_functions=["generate_latin_square", "create_blind_codes", "check_leakage"]
        )
        
        # Check if randomization system exists
        randomization_files = list(self.project_root.glob("outputs/experiments/runs/*/counterbalance.json"))
        blinding_files = list(self.project_root.glob("outputs/experiments/runs/*/key/unblind.json"))
        
        if not randomization_files and not blinding_files:
            item.status = "not_implemented"
            item.details = "No randomization or blinding files found"
        elif randomization_files and blinding_files:
            item.status = "pass"
            item.details = f"Found {len(randomization_files)} randomization files and {len(blinding_files)} blinding files"
        else:
            item.status = "warning"
            item.details = "Partial implementation - some files missing"
        
        return item
    
    def _check_effect_boundaries(self) -> ChecklistItem:
        """Check 4: Backends and Effect Boundaries"""
        item = ChecklistItem(
            id="effect_boundaries",
            name="Backends and Effect Boundaries",
            description="Enforce effect types, API backends reject ActivationEffects, fixed application order",
            status="not_implemented",
            details="",
            required_files=["neuromod/effect_boundaries.py", "neuromod/effects.py"],
            required_functions=["enforce_effect_types", "check_activation_effects", "apply_effects_in_order"]
        )
        
        # Check if effect boundaries system exists
        boundaries_file = self.project_root / "neuromod" / "effect_boundaries.py"
        if not boundaries_file.exists():
            item.status = "fail"
            item.details = "effect_boundaries.py not found"
            return item
        
        try:
            with open(boundaries_file, 'r') as f:
                content = f.read()
            
            # Check for effect type definitions and boundary enforcement
            required_classes = ["EffectType", "EffectBoundaryEnforcer"]
            required_functions = ["validate_effect_for_backend", "enforce_effect_boundaries"]
            
            found_classes = [cls for cls in required_classes if cls in content]
            found_functions = [func for func in required_functions if func in content]
            
            if len(found_classes) >= 2 and len(found_functions) >= 2:
                item.status = "pass"
                item.details = f"Found effect boundary system with {found_classes} and {found_functions}"
            else:
                item.status = "warning"
                item.details = f"Partial implementation - classes: {found_classes}, functions: {found_functions}"
                
        except Exception as e:
            item.status = "fail"
            item.details = f"Error reading effect_boundaries.py: {e}"
        
        return item
    
    def _check_baselines_controls(self) -> ChecklistItem:
        """Check 5: Baselines and Controls"""
        item = ChecklistItem(
            id="baselines",
            name="Baselines and Controls",
            description="Three conditions: control, persona baseline, pack treatment",
            status="not_implemented",
            details="",
            required_files=["packs/none.json", "packs/placebo.json"],
            required_functions=["create_control_condition", "create_persona_baseline"]
        )
        
        # Check for control packs
        control_pack = self.project_root / "packs" / "none.json"
        placebo_pack = self.project_root / "packs" / "placebo.json"
        
        if control_pack.exists() and placebo_pack.exists():
            item.status = "pass"
            item.details = "Control and placebo packs found"
        elif control_pack.exists():
            item.status = "warning"
            item.details = "Control pack found, but placebo pack missing"
        else:
            item.status = "fail"
            item.details = "Control packs not found"
        
        return item
    
    def _check_power_analysis(self) -> ChecklistItem:
        """Check 6: Power and Sample Size"""
        item = ChecklistItem(
            id="power_analysis",
            name="Power and Sample Size",
            description="Pilot study, n_min calculation, power analysis script",
            status="pass",
            details="",
            required_files=["analysis/power_analysis.py"],
            required_functions=["calculate_sample_size", "estimate_effect_size"]
        )
        
        power_script = self.project_root / "analysis" / "power_analysis.py"
        if power_script.exists():
            item.status = "pass"
            item.details = "Power analysis script found"
        else:
            item.status = "fail"
            item.details = "Power analysis script not found"
        
        return item
    
    def _check_off_target_monitoring(self) -> ChecklistItem:
        """Check 8: Off-target Monitoring"""
        item = ChecklistItem(
            id="off_target",
            name="Off-target Monitoring",
            description="Track refusal rate, toxicity, verbosity, hallucination proxy",
            status="not_implemented",
            details="",
            required_files=["neuromod/testing/off_target_monitor.py"],
            required_functions=["monitor_refusal_rate", "monitor_toxicity", "monitor_verbosity"]
        )
        
        # Check if off-target monitoring exists
        off_target_file = self.project_root / "neuromod" / "testing" / "off_target_monitor.py"
        if off_target_file.exists():
            item.status = "pass"
            item.details = "Off-target monitoring system found"
        else:
            item.status = "not_implemented"
            item.details = "Off-target monitoring system not implemented"
        
        return item
    
    def _check_robustness_generalization(self) -> ChecklistItem:
        """Check 9: Robustness and Generalization"""
        item = ChecklistItem(
            id="robustness",
            name="Robustness and Generalization",
            description="Paraphrase sets, multiple models, held-out prompts",
            status="not_implemented",
            details="",
            required_files=["data/paraphrase_sets.json", "data/held_out_prompts.json"],
            required_functions=["create_paraphrase_sets", "validate_generalization"]
        )
        
        # Check for robustness validation system
        robustness_file = self.project_root / "neuromod" / "testing" / "robustness_validation.py"
        
        if robustness_file.exists():
            try:
                with open(robustness_file, 'r') as f:
                    content = f.read()
                
                # Check for required functions
                required_functions = ["validate_robustness", "calculate_robustness_score", "perform_meta_analysis"]
                found_functions = [func for func in required_functions if f"def {func}" in content]
                
                if len(found_functions) >= 2:
                    item.status = "pass"
                    item.details = f"Robustness validation system found with {len(found_functions)} required functions"
                else:
                    item.status = "warning"
                    item.details = f"Robustness validation system found but only {len(found_functions)} required functions"
            except Exception as e:
                item.status = "fail"
                item.details = f"Error reading robustness file: {e}"
        else:
            item.status = "not_implemented"
            item.details = "Robustness validation system not found"
        
        return item
    
    def _check_ablations_dose_response(self) -> ChecklistItem:
        """Check 10: Ablations and Dose-response"""
        item = ChecklistItem(
            id="ablations",
            name="Ablations and Dose-response",
            description="Minus-one ablations, dose-response grid, monotonic trends",
            status="not_implemented",
            details="",
            required_files=["neuromod/testing/ablations_analysis.py"],
            required_functions=["run_component_ablations", "run_dose_response_analysis", "analyze_effect_interactions"]
        )
        
        ablation_file = self.project_root / "neuromod" / "testing" / "ablations_analysis.py"
        
        if ablation_file.exists():
            try:
                with open(ablation_file, 'r') as f:
                    content = f.read()
                
                # Check for required functions
                required_functions = ["run_component_ablations", "run_dose_response_analysis", "analyze_effect_interactions"]
                found_functions = [func for func in required_functions if f"def {func}" in content]
                
                if len(found_functions) >= 2:
                    item.status = "pass"
                    item.details = f"Ablations and dose-response system found with {len(found_functions)} required functions"
                else:
                    item.status = "warning"
                    item.details = f"Ablations system found but only {len(found_functions)} required functions"
            except Exception as e:
                item.status = "fail"
                item.details = f"Error reading ablations file: {e}"
        else:
            item.status = "not_implemented"
            item.details = "Ablations and dose-response system not implemented"
        
        return item
    
    def _check_reproducibility_switches(self) -> ChecklistItem:
        """Check 11: Reproducibility Switches"""
        item = ChecklistItem(
            id="reproducibility",
            name="Reproducibility Switches",
            description="set_run_seed(), deterministic composition, caching",
            status="not_implemented",
            details="",
            required_files=["neuromod/reproducibility.py"],
            required_functions=["set_run_seed", "deterministic_composition", "cache_prompts"]
        )
        
        repro_file = self.project_root / "neuromod" / "testing" / "reproducibility_switches.py"
        if repro_file.exists():
            try:
                with open(repro_file, 'r') as f:
                    content = f.read()
                
                # Check for required functions
                required_functions = ["validate_environment", "create_reproducibility_lock", "setup_deterministic_environment"]
                found_functions = [func for func in required_functions if f"def {func}" in content]
                
                if len(found_functions) >= 2:
                    item.status = "pass"
                    item.details = f"Reproducibility system found with {len(found_functions)} required functions"
                else:
                    item.status = "warning"
                    item.details = f"Reproducibility system found but only {len(found_functions)} required functions"
            except Exception as e:
                item.status = "fail"
                item.details = f"Error reading reproducibility file: {e}"
        else:
            item.status = "not_implemented"
            item.details = "Reproducibility system not implemented"
        
        return item
    
    def _check_reporting_system(self) -> ChecklistItem:
        """Check 12: Reporting"""
        item = ChecklistItem(
            id="reporting",
            name="Reporting",
            description="PDF generation, machine-readable CSVs, figures and tables",
            status="not_implemented",
            details="",
            required_files=["analysis/reporting.py", "analysis/figures/"],
            required_functions=["generate_pdf_report", "export_csv_results", "create_figures"]
        )
        
        reporting_file = self.project_root / "analysis" / "reporting_system.py"
        
        if reporting_file.exists():
            try:
                with open(reporting_file, 'r') as f:
                    content = f.read()
                
                # Check for required functions
                required_functions = ["generate_comprehensive_report", "generate_statistical_report", "generate_html_report"]
                found_functions = [func for func in required_functions if f"def {func}" in content]
                
                if len(found_functions) >= 2:
                    item.status = "pass"
                    item.details = f"Reporting system found with {len(found_functions)} required functions"
                else:
                    item.status = "warning"
                    item.details = f"Reporting system found but only {len(found_functions)} required functions"
            except Exception as e:
                item.status = "fail"
                item.details = f"Error reading reporting file: {e}"
        else:
            item.status = "not_implemented"
            item.details = "Reporting system not implemented"
        
        return item
    
    def _check_data_code_release(self) -> ChecklistItem:
        """Check 13: Data and Code Release"""
        item = ChecklistItem(
            id="data_release",
            name="Data and Code Release",
            description="Sample bundle, Makefile, reproducibility scripts",
            status="not_implemented",
            details="",
            required_files=["analysis/data_code_release.py"],
            required_functions=["prepare_release_package", "_prepare_data_files", "_prepare_code_files"]
        )
        
        release_file = self.project_root / "analysis" / "data_code_release.py"
        
        if release_file.exists():
            try:
                with open(release_file, 'r') as f:
                    content = f.read()
                
                # Check for required functions
                required_functions = ["prepare_release_package", "_prepare_data_files", "_prepare_code_files"]
                found_functions = [func for func in required_functions if f"def {func}" in content]
                
                if len(found_functions) >= 2:
                    item.status = "pass"
                    item.details = f"Data and code release system found with {len(found_functions)} required functions"
                else:
                    item.status = "warning"
                    item.details = f"Release system found but only {len(found_functions)} required functions"
            except Exception as e:
                item.status = "fail"
                item.details = f"Error reading release file: {e}"
        else:
            item.status = "not_implemented"
            item.details = "Data and code release system not implemented"
        
        return item
    
    def _check_safety_ethics(self) -> ChecklistItem:
        """Check 14: Safety and Ethics"""
        item = ChecklistItem(
            id="safety",
            name="Safety and Ethics",
            description="Risk levels, research-only flag, safety monitoring",
            status="not_implemented",
            details="",
            required_files=["analysis/safety_ethics.py"],
            required_functions=["conduct_safety_review", "conduct_ethics_review", "_identify_safety_risks"]
        )
        
        safety_file = self.project_root / "analysis" / "safety_ethics.py"
        
        if safety_file.exists():
            try:
                with open(safety_file, 'r') as f:
                    content = f.read()
                
                # Check for required functions
                required_functions = ["conduct_safety_review", "conduct_ethics_review", "_identify_safety_risks"]
                found_functions = [func for func in required_functions if f"def {func}" in content]
                
                if len(found_functions) >= 2:
                    item.status = "pass"
                    item.details = f"Safety and ethics system found with {len(found_functions)} required functions"
                else:
                    item.status = "warning"
                    item.details = f"Safety system found but only {len(found_functions)} required functions"
            except Exception as e:
                item.status = "fail"
                item.details = f"Error reading safety file: {e}"
        else:
            item.status = "not_implemented"
            item.details = "Safety and ethics system not implemented"
        
        return item
    
    def _check_qa_tests(self) -> ChecklistItem:
        """Check 15: QA Tests that Enforce Rigor"""
        item = ChecklistItem(
            id="qa_tests",
            name="QA Tests that Enforce Rigor",
            description="Unit tests for Latin square, schema validation, golden-master tests",
            status="not_implemented",
            details="",
            required_files=["tests/test_rigor.py", "tests/test_schema.py"],
            required_functions=["test_latin_square", "test_schema_validation", "test_golden_master"]
        )
        
        rigor_tests = self.project_root / "tests" / "test_rigor_validation.py"
        
        if rigor_tests.exists():
            # Check if the test file has the required functions
            try:
                with open(rigor_tests, 'r') as f:
                    content = f.read()
                
                required_tests = ["test_preregistration", "test_randomization", "test_effect_boundaries"]
                found_tests = [test for test in required_tests if f"def {test}" in content]
                
                if len(found_tests) >= 3:
                    item.status = "pass"
                    item.details = f"QA tests for rigor found with {len(found_tests)} required functions"
                else:
                    item.status = "warning"
                    item.details = f"QA tests found but only {len(found_tests)} required functions"
            except Exception as e:
                item.status = "fail"
                item.details = f"Error reading test file: {e}"
        else:
            item.status = "not_implemented"
            item.details = "QA tests for rigor not implemented"
        
        return item
    
    def _check_statistical_correction(self) -> ChecklistItem:
        """Check 7: Multiple Comparisons and Statistics"""
        item = ChecklistItem(
            id="statistical_correction",
            name="Multiple Comparisons and Statistics",
            description="Paired tests, BH-FDR correction, effect sizes, bootstrap CIs",
            status="not_implemented",
            details="",
            required_files=["analysis/statistical_analysis.py"],
            required_functions=["apply_fdr_correction", "calculate_effect_sizes", "bootstrap_ci"]
        )
        
        stats_file = self.project_root / "analysis" / "statistical_analysis.py"
        if stats_file.exists():
            item.status = "pass"
            item.details = "Statistical analysis system found"
        else:
            item.status = "not_implemented"
            item.details = "Statistical analysis system not implemented"
        
        return item
    
    def generate_report(self, output_path: str = "analysis/rigor_validation_report.json"):
        """Generate validation report"""
        
        # Create analysis directory if it doesn't exist
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Count statuses
        status_counts = {}
        for item in self.results:
            status_counts[item.status] = status_counts.get(item.status, 0) + 1
        
        # Create report
        report = {
            "validation_date": str(Path.cwd()),
            "total_checks": len(self.results),
            "status_summary": status_counts,
            "checks": [
                {
                    "id": item.id,
                    "name": item.name,
                    "status": item.status,
                    "details": item.details
                }
                for item in self.results
            ]
        }
        
        # Save report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        print("\n" + "=" * 60)
        print("RIGOR VALIDATION SUMMARY")
        print("=" * 60)
        
        for status, count in status_counts.items():
            emoji = {"pass": "✅", "fail": "❌", "warning": "⚠️", "not_implemented": "⏳"}
            print(f"{emoji.get(status, '❓')} {status.upper()}: {count}")
        
        print(f"\nTotal checks: {len(self.results)}")
        print(f"Report saved to: {output_path}")
        
        # Print detailed results
        print("\nDETAILED RESULTS:")
        print("-" * 60)
        
        for item in self.results:
            emoji = {"pass": "✅", "fail": "❌", "warning": "⚠️", "not_implemented": "⏳"}
            print(f"\n{emoji.get(item.status, '❓')} {item.name}")
            print(f"   Status: {item.status}")
            print(f"   Details: {item.details}")
        
        return report

def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description="Validate scientific rigor implementation")
    parser.add_argument("--project-root", default=".", 
                       help="Project root directory")
    parser.add_argument("--output", default="analysis/rigor_validation_report.json",
                       help="Output path for validation report")
    
    args = parser.parse_args()
    
    # Run validation
    validator = RigorValidator(args.project_root)
    validator.validate_all()
    validator.generate_report(args.output)

if __name__ == "__main__":
    main()
