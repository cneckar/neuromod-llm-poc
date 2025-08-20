"""
Test Runner for Neuromodulation Testing Framework
"""

import argparse
from typing import Dict, List, Any, Optional
from .test_suite import TestSuite
from .pdq_test import PDQTest
from .sdq_test import SDQTest
from .ddq_test import DDQTest
from .didq_test import DiDQTest
from .edq_test import EDQTest
from .cdq_test import CDQTest
from .pcq_pop_test import PCQPopTest
from .adq_test import ADQTest
from ..pack_system import PackRegistry
from ..neuromod_tool import NeuromodTool

class TestRunner:
    """High-level test runner for neuromodulation testing"""
    
    def __init__(self, model_name: str = "gpt2", packs_path: str = "packs/config.json"):
        self.model_name = model_name
        self.packs_path = packs_path
        self.suite = TestSuite(model_name)
        self.neuromod_tool = None
        
        # Initialize neuromodulation system
        self._setup_neuromodulation()
        
        # Register available tests
        self._register_tests()
        
    def _setup_neuromodulation(self):
        """Setup the neuromodulation system"""
        try:
            registry = PackRegistry(self.packs_path)
            print(f"✅ Loaded neuromod registry from {self.packs_path}")
            print(f"Available packs: {registry.list_packs()}")
            
            # Create a placeholder model for the tool (will be replaced when tests run)
            from transformers import AutoTokenizer, AutoModelForCausalLM
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,
                device_map="cpu",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            model = model.cpu()
            model.eval()
            
            self.neuromod_tool = NeuromodTool(registry, model, tokenizer)
            self.suite.set_neuromod_tool(self.neuromod_tool)
            
        except Exception as e:
            print(f"⚠️ Warning: Could not setup neuromodulation system: {e}")
            print("Tests will run without neuromodulation capabilities")
            
    def _register_tests(self):
        """Register all available tests"""
        # Add PDQ test
        pdq_test = PDQTest(self.model_name)
        self.suite.add_test(pdq_test)
        
        # Add SDQ test
        sdq_test = SDQTest(self.model_name)
        self.suite.add_test(sdq_test)
        
        # Add DDQ test
        ddq_test = DDQTest(self.model_name)
        self.suite.add_test(ddq_test)
        
        # Add DiDQ test
        didq_test = DiDQTest(self.model_name)
        self.suite.add_test(didq_test)
        
        # Add EDQ test
        edq_test = EDQTest(self.model_name)
        self.suite.add_test(edq_test)
        
        # Add CDQ test
        cdq_test = CDQTest(self.model_name)
        self.suite.add_test(cdq_test)
        
        # Add PCQ-POP test
        pcq_pop_test = PCQPopTest(self.model_name)
        self.suite.add_test(pcq_pop_test)
        
        # Add ADQ test
        adq_test = ADQTest(self.model_name)
        self.suite.add_test(adq_test)
        
        # Add more tests here as they are implemented
        # self.suite.add_test(OtherTest(self.model_name))
        
    def run_single_test(self, test_name: str = "pdq", packs: List[str] = None) -> Dict[str, Any]:
        """Run a single test by name"""
        test_map = {
            'pdq': 0,
            'sdq': 1,
            'ddq': 2,
            'didq': 3,
            'edq': 4,
            'cdq': 5,
            'pcq_pop': 6,
            'adq': 7,
            # Add more test mappings here
        }
        
        if test_name not in test_map:
            raise ValueError(f"Unknown test: {test_name}. Available: {list(test_map.keys())}")
            
        test_index = test_map[test_name]
        return self.suite.run_single_test(test_index, packs)
        
    def run_test_sequence(self, packs: List[str] = None) -> List[Dict[str, Any]]:
        """Run all tests in sequence"""
        return self.suite.run_test_sequence(packs)
        
    def run_comparison(self, test_name: str = "pdq", pack_combinations: List[List[str]] = None) -> Dict[str, Any]:
        """Run a test with multiple pack combinations"""
        test_map = {
            'pdq': 0,
            'sdq': 1,
            'ddq': 2,
            'didq': 3,
            'edq': 4,
            'cdq': 5,
            'pcq_pop': 6,
            'adq': 7,
            # Add more test mappings here
        }
        
        if test_name not in test_map:
            raise ValueError(f"Unknown test: {test_name}. Available: {list(test_map.keys())}")
            
        test_index = test_map[test_name]
        return self.suite.run_comparison_test(test_index, pack_combinations)
        
    def list_available_tests(self):
        """List all available tests"""
        self.suite.list_tests()
        
    def list_available_packs(self):
        """List all available packs"""
        try:
            registry = PackRegistry(self.packs_path)
            packs = registry.list_packs()
            print(f"\n📦 Available packs:")
            print("=" * 30)
            for pack in packs:
                print(f"  - {pack}")
        except Exception as e:
            print(f"❌ Could not load packs: {e}")
    
    def verify_blinding(self):
        """Verify that all tests maintain blinding"""
        try:
            from .blinding_verification import verify_test_blinding
            return verify_test_blinding()
        except ImportError as e:
            print(f"❌ Could not import blinding verification: {e}")
            return False
    
    def run_statistical_analysis(self, test_name: str = "pdq", 
                                baseline_packs: List[str] = None,
                                treatment_packs: List[str] = None,
                                output_path: Optional[str] = None) -> Dict[str, Any]:
        """Run statistical analysis comparing baseline vs treatment conditions"""
        try:
            from .statistical_analysis import analyze_neuromodulation_results, generate_analysis_report
            
            print(f"📊 Running statistical analysis for {test_name}")
            print(f"Baseline packs: {baseline_packs or 'none'}")
            print(f"Treatment packs: {treatment_packs or 'none'}")
            print("=" * 70)
            
            # Run baseline condition
            baseline_results = []
            if baseline_packs:
                for pack in baseline_packs:
                    result = self.run_single_test(test_name, [pack])
                    baseline_results.append(result)
            else:
                # Run without packs as baseline
                result = self.run_single_test(test_name, [])
                baseline_results.append(result)
            
            # Run treatment condition
            treatment_results = []
            if treatment_packs:
                for pack in treatment_packs:
                    result = self.run_single_test(test_name, [pack])
                    treatment_results.append(result)
            
            # Perform statistical analysis
            analysis_results = analyze_neuromodulation_results(
                baseline_results, treatment_results, test_name
            )
            
            # Generate report
            report = generate_analysis_report(analysis_results, output_path)
            print("\n📋 ANALYSIS REPORT")
            print("=" * 70)
            print(report)
            
            return analysis_results
            
        except ImportError as e:
            print(f"❌ Could not import statistical analysis: {e}")
            return {}
        except Exception as e:
            print(f"❌ Error during statistical analysis: {e}")
            return {}
            
    def cleanup(self):
        """Clean up resources"""
        self.suite.cleanup()

def main():
    """Command-line interface for the test runner"""
    parser = argparse.ArgumentParser(description="Neuromodulation Test Runner")
    parser.add_argument("--model", default="gpt2", help="Model to test")
    parser.add_argument("--packs", default="packs/config.json", help="Path to packs file")
    parser.add_argument("--test", default="pdq", help="Test to run (pdq, sdq, ddq, didq, edq, cdq, pcq_pop, adq, etc.)")
    parser.add_argument("--mode", choices=["single", "sequence", "comparison"], 
                       default="single", help="Test mode")
    parser.add_argument("--packs-to-apply", nargs="*", help="Packs to apply")
    parser.add_argument("--list-tests", action="store_true", help="List available tests")
    parser.add_argument("--list-packs", action="store_true", help="List available packs")
    parser.add_argument("--verify-blinding", action="store_true", help="Verify that all tests maintain blinding")
    parser.add_argument("--statistical-analysis", action="store_true", help="Run statistical analysis comparing baseline vs treatment")
    parser.add_argument("--baseline-packs", nargs="*", help="Packs to use as baseline condition")
    parser.add_argument("--treatment-packs", nargs="*", help="Packs to use as treatment condition")
    parser.add_argument("--output-path", help="Path prefix for output files")
    
    args = parser.parse_args()
    
    # Create test runner
    runner = TestRunner(args.model, args.packs)
    
    try:
        if args.list_tests:
            runner.list_available_tests()
            return
            
        if args.list_packs:
            runner.list_available_packs()
            return
            
        if args.verify_blinding:
            runner.verify_blinding()
            return
            
        if args.statistical_analysis:
            runner.run_statistical_analysis(
                test_name=args.test,
                baseline_packs=args.baseline_packs,
                treatment_packs=args.treatment_packs,
                output_path=args.output_path
            )
            return
            
        print(f"🚀 Neuromodulation Test Runner")
        print(f"Model: {args.model}")
        print(f"Mode: {args.mode}")
        print(f"Test: {args.test}")
        if args.packs_to_apply:
            print(f"Packs: {args.packs_to_apply}")
        print("=" * 70)
        
        if args.mode == "single":
            results = runner.run_single_test(args.test, args.packs_to_apply)
            print(f"\n✅ Single test completed: {args.test}")
            
        elif args.mode == "sequence":
            results = runner.run_test_sequence(args.packs_to_apply)
            print(f"\n✅ Test sequence completed: {len(results)} tests")
            
        elif args.mode == "comparison":
            # Define default combinations for comparison
            combinations = [
                [],  # No packs
                ['nicotine_v1'],  # Single pack
            ]
            if args.packs_to_apply:
                combinations = [args.packs_to_apply]
                
            results = runner.run_comparison(args.test, combinations)
            print(f"\n✅ Comparison test completed: {args.test}")
            
        print("\n" + "=" * 70)
        print("✅ Test runner completed successfully!")
        
    except Exception as e:
        print(f"❌ Test runner failed: {e}")
        return 1
        
    finally:
        runner.cleanup()
        
    return 0

if __name__ == "__main__":
    import torch
    exit(main())
