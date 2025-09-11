#!/usr/bin/env python3
"""
Test Runner for Neuromodulation Tests

Provides a simple interface to run ADQ, CDQ, SDQ, DDQ, PDQ, EDQ tests
"""

import sys
import os
import argparse
from typing import Dict, List, Any, Optional

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from neuromod.pack_system import PackRegistry
    from neuromod.neuromod_tool import NeuromodTool
    from neuromod.neuromod_factory import create_neuromod_tool, cleanup_neuromod_tool
    from neuromod.model_support import create_model_support
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the project root with PYTHONPATH=.")
    sys.exit(1)


class TestRunner:
    """Simple test runner for neuromodulation tests"""
    
    def __init__(self, model_name: str = None, packs_file: str = "packs/config.json", test_mode: bool = True):
        self.model_name = model_name
        self.packs_file = packs_file
        self.test_mode = test_mode
        self.pack_registry = None
        self.neuromod_tool = None
        self.model_info = None
        
        # Import test classes dynamically
        self.available_tests = self._import_test_classes()
        
        # Initialize model support
        self.model_manager = create_model_support(test_mode=test_mode)
    
    def initialize_model(self, model_name: str = None):
        """Initialize the model and neuromod tool"""
        if model_name is None:
            model_name = self.model_manager.get_recommended_model()
        
        self.model_name = model_name
        
        try:
            # Create neuromod tool with model
            self.neuromod_tool, self.model_info = create_neuromod_tool(
                model_name=model_name,
                test_mode=self.test_mode
            )
            
            print(f"✅ Model loaded: {model_name}")
            print(f"   Backend: {self.model_info.get('backend', 'unknown')}")
            print(f"   Size: {self.model_info.get('size', 'unknown')}")
            print(f"   Parameters: {self.model_info.get('parameters', 'unknown')}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to load model {model_name}: {e}")
            return False
    
    def cleanup(self):
        """Clean up resources"""
        if self.neuromod_tool:
            cleanup_neuromod_tool(self.neuromod_tool)
        if self.model_manager:
            self.model_manager.cleanup()
    
    def _import_test_classes(self):
        """Import test classes dynamically"""
        test_classes = {}
        
        try:
            # Try relative imports first (when used as module)
            try:
                from .adq_test import ADQTest
                from .cdq_test import CDQTest
                from .sdq_test import SDQTest
                from .ddq_test import DDQTest
                from .pdq_test import PDQTest
                from .edq_test import EDQTest
                from .pcq_pop_test import PCQPopTest
            except ImportError:
                # Fall back to absolute imports (when run standalone)
                from adq_test import ADQTest
                from cdq_test import CDQTest
                from sdq_test import SDQTest
                from ddq_test import DDQTest
                from pdq_test import PDQTest
                from edq_test import EDQTest
                from pcq_pop_test import PCQPopTest
            
            test_classes = {
                'adq': ADQTest,
                'cdq': CDQTest,
                'sdq': SDQTest,
                'ddq': DDQTest,
                'pdq': PDQTest,
                'edq': EDQTest,
                'pcq_pop': PCQPopTest
            }
            
        except ImportError as e:
            print(f"Error importing test classes: {e}")
            print("Available tests will be limited")
        
        return test_classes
    
    def load_pack_registry(self):
        """Load the pack registry"""
        if self.pack_registry is None:
            try:
                self.pack_registry = PackRegistry(self.packs_file)
                print(f"✅ Loaded pack registry from {self.packs_file}")
            except Exception as e:
                print(f"⚠️  Could not load pack registry: {e}")
                print("   Tests will run without neuromodulation")
                self.pack_registry = None
    
    def create_neuromod_tool(self, model, tokenizer):
        """Create neuromod tool with pack registry"""
        if self.pack_registry:
            return NeuromodTool(self.pack_registry, model, tokenizer)
        else:
            # Create a minimal neuromod tool without packs
            from neuromod.neuromod_tool import NeuromodTool
            tool = NeuromodTool()
            tool.model = model
            tool.tokenizer = tokenizer
            return tool
    
    def run_test(self, test_name: str, packs: List[str] = None) -> Dict[str, Any]:
        """Run a single test"""
        if test_name not in self.available_tests:
            raise ValueError(f"Unknown test: {test_name}. Available: {list(self.available_tests.keys())}")
        
        print(f"🚀 Running {test_name.upper()} test")
        
        # Initialize model if not already done
        if self.neuromod_tool is None:
            if not self.initialize_model():
                return {"error": "Failed to initialize model"}
        
        # Load pack registry
        self.load_pack_registry()
        
        # Create test instance
        test_class = self.available_tests[test_name]
        test = test_class(self.model_name)
        
        try:
            # Set up test with our neuromod tool
            test.set_neuromod_tool(self.neuromod_tool)
            
            # Apply packs if specified
            if packs and self.pack_registry:
                print(f"🎯 Applying packs: {packs}")
                for pack_name in packs:
                    pack = self.pack_registry.get_pack(pack_name)
                    if pack:
                        self.neuromod_tool.apply(pack_name, intensity=0.7)
                        print(f"   ✅ Applied {pack_name}")
                    else:
                        print(f"   ❌ Pack not found: {pack_name}")
            
            # Run the test
            results = test.run_test(self.neuromod_tool)
            
            print(f"✅ {test_name.upper()} test completed")
            return results
            
        except Exception as e:
            print(f"❌ Error running {test_name} test: {e}")
            return {
                'test_name': test_name,
                'status': 'error',
                'error': str(e)
            }
        finally:
            # Cleanup
            test.cleanup()
    
    def run_all_tests(self, packs: List[str] = None) -> Dict[str, Any]:
        """Run all available tests"""
        print(f"🧪 Running all tests with model: {self.model_name}")
        
        results = {}
        for test_name in self.available_tests.keys():
            print(f"\n{'='*60}")
            results[test_name] = self.run_test(test_name, packs)
        
        print(f"\n{'='*60}")
        print("🎉 All tests completed!")
        
        # Summary
        successful = sum(1 for r in results.values() if r.get('status') != 'error')
        total = len(results)
        print(f"📊 Summary: {successful}/{total} tests successful")
        
        return results
    
    def list_tests(self):
        """List available tests"""
        print("Available tests:")
        for test_name, test_class in self.available_tests.items():
            test_instance = test_class()
            print(f"  {test_name}: {test_instance.get_test_name()}")
    
    def list_packs(self):
        """List available packs"""
        self.load_pack_registry()
        if self.pack_registry:
            print("Available packs:")
            for pack_name in self.pack_registry.packs.keys():
                print(f"  {pack_name}")
        else:
            print("No pack registry loaded")


def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(description="Run neuromodulation tests")
    parser.add_argument("--model", default="microsoft/DialoGPT-small", help="Model to use")
    parser.add_argument("--packs", default="packs/config.json", help="Packs configuration file")
    parser.add_argument("--test", help="Specific test to run (e.g., adq, cdq, sdq)")
    parser.add_argument("--packs-to-apply", nargs="*", help="Packs to apply during testing")
    parser.add_argument("--list-tests", action="store_true", help="List available tests")
    parser.add_argument("--list-packs", action="store_true", help="List available packs")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    
    args = parser.parse_args()
    
    runner = TestRunner(args.model, args.packs)
    
    if args.list_tests:
        runner.list_tests()
    elif args.list_packs:
        runner.list_packs()
    elif args.all:
        runner.run_all_tests(args.packs_to_apply)
    elif args.test:
        runner.run_test(args.test, args.packs_to_apply)
    else:
        print("Please specify --test <name>, --all, --list-tests, or --list-packs")
        parser.print_help()


if __name__ == "__main__":
    main()
