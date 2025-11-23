#!/usr/bin/env python3
"""
Endpoint Calculation Script

Calculates primary and secondary endpoints from test results.
This script can work with:
1. Live test runs (runs tests and calculates endpoints)
2. Saved test results (loads from JSON files)

Usage:
    # Calculate endpoints from saved results
    python scripts/calculate_endpoints.py --results-dir outputs/results --pack caffeine --model gpt2
    
    # Run tests and calculate endpoints
    python scripts/calculate_endpoints.py --run-tests --pack caffeine --model gpt2
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from neuromod.testing.endpoint_calculator import EndpointCalculator, EndpointSummary
from neuromod.testing.adq_test import ADQTest
from neuromod.testing.pdq_test import PDQTest
from neuromod.testing.pcq_pop_test import PCQPopTest
from neuromod.testing.sdq_test import SDQTest
from neuromod.testing.cdq_test import CDQTest
from neuromod.testing.ddq_test import DDQTest
from neuromod.testing.edq_test import EDQTest
from neuromod.testing.cognitive_tasks import CognitiveTasksTest
from neuromod.testing.telemetry import TelemetryCollector
from neuromod.testing.off_target_monitor import OffTargetMonitor
from neuromod.neuromod_tool import NeuromodTool
from neuromod.pack_system import PackRegistry


class EndpointRunner:
    """Runs tests and calculates endpoints"""
    
    def __init__(self, model_name: str = "gpt2", test_mode: bool = True):
        self.model_name = model_name
        self.test_mode = test_mode
        self.calculator = EndpointCalculator()
        self.pack_registry = PackRegistry("packs/config.json")
    
    def _create_neuromod_tool_for_test(self, test, pack_name: str):
        """Create a neuromod tool using the test's model and apply the pack"""
        from neuromod.neuromod_tool import NeuromodTool
        
        # Create neuromod tool with test's model
        neuromod_tool = NeuromodTool(
            registry=self.pack_registry,
            model=test.model,
            tokenizer=test.tokenizer
        )
        
        # Set it on the test
        test.set_neuromod_tool(neuromod_tool)
        
        # Apply the pack
        print(f"  [*] Applying pack: {pack_name}")
        neuromod_tool.apply(pack_name, intensity=0.5)
        
        # Verify orthogonality for RandomOrthogonalSteeringEffect if present
        self._verify_orthogonality(neuromod_tool, pack_name)
        
        return neuromod_tool
    
    def _verify_orthogonality(self, neuromod_tool, pack_name: str):
        """
        Verify orthogonality of RandomOrthogonalSteeringEffect if present in the pack.
        
        This is a critical quality gate that ensures the 'Active Placebo' control
        is scientifically valid (orthogonal to reference vector, not contaminated).
        """
        from neuromod.effects import RandomOrthogonalSteeringEffect
        
        # Check if pack contains random_orthogonal_steering effect
        pack = self.pack_registry.get_pack(pack_name)
        if not pack:
            return
        
        has_orthogonal_effect = False
        for effect_config in pack.effects:
            if effect_config.effect == "random_orthogonal_steering":
                has_orthogonal_effect = True
                break
        
        if not has_orthogonal_effect:
            return
        
        # Find the effect in active effects
        pack_manager = neuromod_tool.pack_manager
        for effect in pack_manager.active_effects:
            if isinstance(effect, RandomOrthogonalSteeringEffect):
                # Verify orthogonality by checking the computed vector
                if effect.orthogonal_vector is not None and effect.reference_vector is not None:
                    dot_product = torch.dot(effect.orthogonal_vector, effect.reference_vector).item()
                    ortho_norm = torch.norm(effect.orthogonal_vector).item()
                    ref_norm = torch.norm(effect.reference_vector).item()
                    
                    print("\n" + "=" * 80)
                    print("🔬 ORTHOGONALITY VERIFICATION (Active Placebo Control)")
                    print("=" * 80)
                    print(f"Pack: {pack_name}")
                    print(f"Reference vector norm: {ref_norm:.6f}")
                    print(f"Orthogonal vector norm: {ortho_norm:.6f}")
                    print(f"Dot product (should be ≈ 0): {dot_product:.2e}")
                    print(f"Threshold: < 1e-6")
                    
                    if abs(dot_product) < 1e-6:
                        print("✅ VALIDATION PASSED: Vectors are orthogonal")
                        print("   The Active Placebo control is scientifically valid.")
                        print("   The random vector has 0% semantic overlap with reference.")
                    else:
                        error_msg = (
                            f"\n❌ VALIDATION FAILED: Orthogonality check failed!\n"
                            f"   Dot product: {dot_product:.2e} (required: < 1e-6)\n"
                            f"   This indicates contamination of the Active Placebo control.\n"
                            f"   The random vector is NOT orthogonal to the reference vector.\n"
                            f"   This would produce false null results.\n"
                        )
                        print(error_msg)
                        raise RuntimeError(
                            f"CRITICAL EXPERIMENTAL FAILURE: Orthogonality validation failed for pack '{pack_name}'. "
                            f"Dot product = {dot_product:.2e} (required < 1e-6). Aborting trial."
                        )
                    print("=" * 80 + "\n")
                break
    
    def _has_successful_result(self, test_name: str, pack_name: str) -> bool:
        """Check if a test already has successful results saved"""
        # Check if results file exists and has valid data
        results_dir = Path("outputs/endpoints")
        if not results_dir.exists():
            return False
        
        # Look for recent endpoint files
        pattern = f"endpoints_{pack_name}_{self.model_name.replace('/', '_')}_*.json"
        result_files = list(results_dir.glob(pattern))
        
        if not result_files:
            return False
        
        # Check the most recent file
        latest_file = max(result_files, key=lambda p: p.stat().st_mtime)
        try:
            with open(latest_file, 'r') as f:
                data = json.load(f)
                # Check if this test has results
                treatment = data.get("treatment_results", {})
                return test_name in treatment and treatment[test_name].get("status") == "completed"
        except:
            return False
    
    def run_tests_for_pack(self, pack_name: str, skip_completed: bool = False, only_tests: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """Run all relevant tests for a pack and return results"""
        print(f"\n[*] Running tests for pack: {pack_name}")
        
        results = {}
        
        # Check if pack exists
        pack = self.pack_registry.get_pack(pack_name)
        if not pack:
            print(f"❌ Pack {pack_name} not found")
            return results
        
        # Run primary endpoint tests
        print("[*] Running primary endpoint tests...")
        
        # ADQ-20 (for stimulant and psychedelic detection)
        if pack_name in ["caffeine", "cocaine", "amphetamine", "methylphenidate", "modafinil",
                        "lsd", "psilocybin", "dmt", "mescaline", "2c_b"]:
            # Skip if only_tests specified and this test not in list
            if only_tests and "ADQ-20" not in only_tests and "adq" not in [t.lower() for t in only_tests]:
                print("  [⏭] Skipping ADQ-20 (not in --only-tests list)")
            # Skip if already completed
            elif skip_completed and self._has_successful_result("ADQ-20", pack_name):
                print("  [⏭] Skipping ADQ-20 (already completed)")
                results["ADQ-20"] = {"status": "skipped", "reason": "already_completed"}
            else:
                print("  [*] Running ADQ-20...")
                try:
                    adq_test = ADQTest(self.model_name)
                    adq_test.test_mode = self.test_mode
                    adq_test.load_model()
                    neuromod_tool = self._create_neuromod_tool_for_test(adq_test, pack_name)
                    if pack_name == "amphetamine":
                        print("  [DEBUG] Running ADQ-20 for amphetamine pack with verbose logging")
                        print(f"  [DEBUG] Model: {self.model_name}, Pack: {pack_name}")
                    adq_results = adq_test.run_test(neuromod_tool=neuromod_tool)
                    if pack_name == "amphetamine":
                        print(f"  [DEBUG] ADQ-20 results: {adq_results.get('status', 'unknown')}")
                        if 'adq_results' in adq_results:
                            print(f"  [DEBUG] Total responses: {adq_results['adq_results'].get('total_responses', 0)}")
                    results["ADQ-20"] = adq_results
                    adq_test.cleanup()
                except Exception as e:
                    print(f"  ❌ ADQ-20 failed: {e}")
                    if pack_name == "amphetamine":
                        print(f"  [DEBUG] Full error traceback for amphetamine:")
                    import traceback
                    traceback.print_exc()
                    results["ADQ-20"] = {"error": str(e), "status": "failed"}
        
        # PDQ-S (for psychedelic detection)
        if pack_name in ["lsd", "psilocybin", "dmt", "mescaline", "2c_b"]:
            if only_tests and "PDQ-S" not in only_tests and "pdq" not in [t.lower() for t in only_tests]:
                print("  [⏭] Skipping PDQ-S (not in --only-tests list)")
            elif skip_completed and self._has_successful_result("PDQ-S", pack_name):
                print("  [⏭] Skipping PDQ-S (already completed)")
                results["PDQ-S"] = {"status": "skipped", "reason": "already_completed"}
            else:
                print("  [*] Running PDQ-S...")
                try:
                    pdq_test = PDQTest(self.model_name)
                    pdq_test.test_mode = self.test_mode
                    pdq_test.load_model()
                    neuromod_tool = self._create_neuromod_tool_for_test(pdq_test, pack_name)
                    pdq_results = pdq_test.run_test(neuromod_tool=neuromod_tool)
                    results["PDQ-S"] = pdq_results
                    pdq_test.cleanup()
                except Exception as e:
                    print(f"  ❌ PDQ-S failed: {e}")
        
        # PCQ-POP-20 (for stimulant and depressant detection)
        if pack_name in ["caffeine", "cocaine", "amphetamine", "methylphenidate", "modafinil",
                        "alcohol", "benzodiazepines", "heroin", "morphine", "fentanyl"]:
            if only_tests and "PCQ-POP-20" not in only_tests and "pcq" not in [t.lower() for t in only_tests]:
                print("  [⏭] Skipping PCQ-POP-20 (not in --only-tests list)")
            elif skip_completed and self._has_successful_result("PCQ-POP-20", pack_name):
                print("  [⏭] Skipping PCQ-POP-20 (already completed)")
                results["PCQ-POP-20"] = {"status": "skipped", "reason": "already_completed"}
            else:
                print("  [*] Running PCQ-POP-20...")
                try:
                    pcq_test = PCQPopTest(self.model_name)
                    pcq_test.test_mode = self.test_mode
                    pcq_test.load_model()
                    neuromod_tool = self._create_neuromod_tool_for_test(pcq_test, pack_name)
                    if pack_name == "amphetamine":
                        print("  [DEBUG] Running PCQ-POP-20 for amphetamine pack with verbose logging")
                        print(f"  [DEBUG] Model: {self.model_name}, Pack: {pack_name}")
                    pcq_results = pcq_test.run_test(neuromod_tool=neuromod_tool)
                    if pack_name == "amphetamine":
                        print(f"  [DEBUG] PCQ-POP-20 results: {pcq_results.get('status', 'unknown')}")
                        if 'sets' in pcq_results:
                            print(f"  [DEBUG] Number of sets completed: {len(pcq_results.get('sets', []))}")
                    results["PCQ-POP-20"] = pcq_results
                    pcq_test.cleanup()
                except Exception as e:
                    print(f"  ❌ PCQ-POP-20 failed: {e}")
                    if pack_name == "amphetamine":
                        print(f"  [DEBUG] Full error traceback for amphetamine:")
                    import traceback
                    traceback.print_exc()
                    results["PCQ-POP-20"] = {"error": str(e), "status": "failed"}
        
        # DDQ (for depressant detection)
        if pack_name in ["alcohol", "benzodiazepines", "heroin", "morphine", "fentanyl"]:
            print("  [*] Running DDQ...")
            try:
                ddq_test = DDQTest(self.model_name)
                ddq_test.test_mode = self.test_mode
                ddq_test.load_model()
                neuromod_tool = self._create_neuromod_tool_for_test(ddq_test, pack_name)
                ddq_results = ddq_test.run_test(neuromod_tool=neuromod_tool)
                results["DDQ"] = ddq_results
                ddq_test.cleanup()
            except Exception as e:
                print(f"  ❌ DDQ failed: {e}")
        
        # Run secondary endpoint tests
        print("[*] Running secondary endpoint tests...")
        
        # CDQ, DDQ, EDQ (cognitive performance)
        print("  [*] Running CDQ, DDQ, EDQ...")
        for test_class, test_name in [(CDQTest, "CDQ"), (DDQTest, "DDQ"), (EDQTest, "EDQ")]:
            try:
                test = test_class(self.model_name)
                test.test_mode = self.test_mode
                test.load_model()
                neuromod_tool = self._create_neuromod_tool_for_test(test, pack_name)
                test_results = test.run_test(neuromod_tool=neuromod_tool)
                results[test_name] = test_results
                test.cleanup()
            except Exception as e:
                print(f"  ❌ {test_name} failed: {e}")
        
        # Cognitive tasks (creativity, attention)
        print("  [*] Running cognitive tasks...")
        try:
            cog_test = CognitiveTasksTest(self.model_name, self.test_mode)
            cog_test.load_model()
            neuromod_tool = self._create_neuromod_tool_for_test(cog_test, pack_name)
            cog_results = cog_test.run_test(neuromod_tool=neuromod_tool)
            results["cognitive_tasks"] = cog_results
            cog_test.cleanup()
        except Exception as e:
            print(f"  ❌ Cognitive tasks failed: {e}")
        
        return results
    
    def run_baseline_tests(self, skip_completed: bool = False, only_tests: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """Run tests without any pack (baseline)"""
        print("\n[*] Running baseline tests (no pack)...")
        return self.run_tests_for_pack("none", skip_completed=skip_completed, only_tests=only_tests)
    
    def run_placebo_tests(self, skip_completed: bool = False, only_tests: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """Run tests with placebo pack"""
        print("\n[*] Running placebo tests...")
        return self.run_tests_for_pack("placebo", skip_completed=skip_completed, only_tests=only_tests)
    
    def calculate_endpoints(self,
                           pack_name: str,
                           output_dir: str = "outputs/endpoints",
                           skip_completed: bool = False,
                           only_tests: Optional[List[str]] = None) -> EndpointSummary:
        """Run tests and calculate all endpoints for a pack"""
        print(f"\n{'='*70}")
        print(f"Calculating Endpoints for Pack: {pack_name}")
        print(f"Model: {self.model_name}")
        print(f"{'='*70}")
        
        # Run tests
        treatment_results = self.run_tests_for_pack(pack_name, skip_completed=skip_completed, only_tests=only_tests)
        baseline_results = self.run_baseline_tests(skip_completed=skip_completed, only_tests=only_tests)
        placebo_results = self.run_placebo_tests(skip_completed=skip_completed, only_tests=only_tests)
        
        # Calculate endpoints
        print(f"\n[*] Calculating endpoints...")
        summary = self.calculator.calculate_all_endpoints(
            test_results=treatment_results,
            baseline_results=baseline_results,
            placebo_results=placebo_results,
            pack_name=pack_name,
            model_name=self.model_name
        )
        
        # Export results
        output_path = Path(output_dir) / f"endpoints_{pack_name}_{self.model_name.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self.calculator.export_results(summary, str(output_path))
        
        # Print summary
        self._print_summary(summary)
        
        return summary
    
    def _print_summary(self, summary: EndpointSummary):
        """Print endpoint calculation summary"""
        print(f"\n{'='*70}")
        print("📊 Endpoint Calculation Summary")
        print(f"{'='*70}")
        print(f"\nPack: {summary.pack_name}")
        print(f"Model: {summary.model_name}")
        print(f"Overall Success: {'✅ YES' if summary.overall_success else '❌ NO'}")
        
        if summary.primary_endpoints:
            print(f"\n📈 Primary Endpoints:")
            for name, result in summary.primary_endpoints.items():
                status = "✅" if result.meets_criteria else "❌"
                print(f"  {status} {name}:")
                print(f"     Treatment: {result.treatment_score:.3f}")
                print(f"     Baseline: {result.baseline_score:.3f}")
                if result.placebo_score is not None:
                    print(f"     Placebo: {result.placebo_score:.3f}")
                print(f"     Effect Size: {result.effect_size:.3f}")
                p_value_str = f"{result.p_value:.4f}" if result.p_value is not None else "N/A"
                print(f"     P-value: {p_value_str}")
                print(f"     Significant: {'Yes' if result.significant else 'No'}")
                print(f"     Meets Criteria: {'Yes' if result.meets_criteria else 'No'}")
        
        if summary.secondary_endpoints:
            print(f"\n📊 Secondary Endpoints:")
            for name, result in summary.secondary_endpoints.items():
                status = "✅" if result.meets_criteria else "❌"
                print(f"  {status} {name}:")
                print(f"     Treatment: {result.treatment_score:.3f}")
                print(f"     Baseline: {result.baseline_score:.3f}")
                print(f"     Effect Size: {result.effect_size:.3f}")


def load_results_from_file(results_file: str) -> Dict[str, Dict[str, Any]]:
    """Load test results from a JSON file"""
    with open(results_file, 'r') as f:
        return json.load(f)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Calculate endpoints from test results")
    parser.add_argument(
        "--pack",
        type=str,
        required=True,
        help="Pack name to test (e.g., caffeine, lsd, alcohol)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2",
        help="Model to use for testing"
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Use test mode (smaller models)"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Directory containing saved test results (if not running tests)"
    )
    parser.add_argument(
        "--run-tests",
        action="store_true",
        help="Run tests before calculating endpoints (default if --results-dir not specified)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/endpoints",
        help="Output directory for endpoint results"
    )
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help="Skip tests that already have successful results"
    )
    parser.add_argument(
        "--only-tests",
        type=str,
        nargs="+",
        help="Run only specific tests (e.g., --only-tests ADQ-20 PCQ-POP-20)"
    )
    
    args = parser.parse_args()
    
    # Auto-detect test vs production models
    test_model_indicators = ['gpt2', 'distilgpt2', 'dialogpt', 'mock']
    production_model_indicators = ['llama', 'qwen', 'mixtral', 'mistral']
    
    if any(indicator in args.model.lower() for indicator in test_model_indicators):
        args.test_mode = True
        print(f"[*] Test model detected: {args.model}")
        print(f"[*] Using TEST mode (test_mode=True)")
    elif any(indicator in args.model.lower() for indicator in production_model_indicators):
        args.test_mode = False
        print(f"[*] Production model detected: {args.model}")
        print(f"[*] Using PRODUCTION mode (test_mode=False)")
    
    # Create runner
    runner = EndpointRunner(model_name=args.model, test_mode=args.test_mode)
    
    # Calculate endpoints
    summary = runner.calculate_endpoints(
        pack_name=args.pack,
        output_dir=args.output_dir,
        skip_completed=args.skip_completed,
        only_tests=args.only_tests
    )
    
    print(f"\n✅ Endpoint calculation complete!")
    print(f"📁 Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

