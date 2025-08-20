"""
Test Suite for running multiple neuromodulation tests
"""

from typing import Dict, List, Any, Optional
from .base_test import BaseTest
from ..neuromod_tool import NeuromodTool

class TestSuite:
    """Test suite for running multiple neuromodulation tests"""
    
    def __init__(self, model_name: str = "gpt2"):
        self.model_name = model_name
        self.tests: List[BaseTest] = []
        self.neuromod_tool: Optional[NeuromodTool] = None
        
    def add_test(self, test: BaseTest):
        """Add a test to the suite"""
        self.tests.append(test)
        
    def set_neuromod_tool(self, neuromod_tool: NeuromodTool):
        """Set the neuromodulation tool for the suite"""
        self.neuromod_tool = neuromod_tool
        
    def run_single_test(self, test_index: int = 0, packs: List[str] = None) -> Dict[str, Any]:
        """Run a single test with specified packs"""
        if test_index >= len(self.tests):
            raise ValueError(f"Test index {test_index} out of range (0-{len(self.tests)-1})")
            
        test = self.tests[test_index]
        print(f"\n🧪 Running single test: {test.get_test_name()}")
        print("=" * 70)
        
        # Apply packs if specified
        if packs and self.neuromod_tool:
            print(f"📦 Applying packs: {packs}")
            for pack in packs:
                result = self.neuromod_tool.apply(pack, intensity=0.8)
                if result.get("ok"):
                    print(f"✅ Applied pack: {pack}")
                else:
                    print(f"❌ Failed to apply pack: {pack}")
        
        # Set neuromod tool for the test
        test.set_neuromod_tool(self.neuromod_tool)
        
        # Run the test
        results = test.run_test(self.neuromod_tool)
        
        # Clear all packs
        if self.neuromod_tool:
            self.neuromod_tool.clear()
            
        return results
        
    def run_test_sequence(self, packs: List[str] = None) -> List[Dict[str, Any]]:
        """Run all tests in sequence with specified packs"""
        print(f"\n🧪 Running test sequence with packs: {packs or 'none'}")
        print("=" * 70)
        
        all_results = []
        
        for i, test in enumerate(self.tests):
            print(f"\n📋 Test {i+1}/{len(self.tests)}: {test.get_test_name()}")
            
            # Apply packs if specified
            if packs and self.neuromod_tool:
                print(f"📦 Applying packs: {packs}")
                for pack in packs:
                    result = self.neuromod_tool.apply(pack, intensity=0.8)
                    if result.get("ok"):
                        print(f"✅ Applied pack: {pack}")
                    else:
                        print(f"❌ Failed to apply pack: {pack}")
            
            # Set neuromod tool for the test
            test.set_neuromod_tool(self.neuromod_tool)
            
            # Run the test
            results = test.run_test(self.neuromod_tool)
            all_results.append(results)
            
            # Clear all packs
            if self.neuromod_tool:
                self.neuromod_tool.clear()
                
        return all_results
        
    def run_comparison_test(self, test_index: int = 0, pack_combinations: List[List[str]] = None) -> Dict[str, Any]:
        """Run a test with multiple pack combinations for comparison"""
        if test_index >= len(self.tests):
            raise ValueError(f"Test index {test_index} out of range (0-{len(self.tests)-1})")
            
        test = self.tests[test_index]
        print(f"\n🧪 Running comparison test: {test.get_test_name()}")
        print("=" * 70)
        
        comparison_results = {
            'test_name': test.get_test_name(),
            'combinations': {}
        }
        
        # Default combinations if none specified
        if pack_combinations is None:
            pack_combinations = [
                [],  # No packs
                ['nicotine_v1'],  # Single pack
                ['nicotine_v1', 'psychedelic_v1']  # Multiple packs
            ]
        
        for combo in pack_combinations:
            combo_name = " + ".join(combo) if combo else "No packs"
            print(f"\n📦 Testing combination: {combo_name}")
            
            # Apply packs
            if combo and self.neuromod_tool:
                for pack in combo:
                    result = self.neuromod_tool.apply(pack, intensity=0.8)
                    if result.get("ok"):
                        print(f"✅ Applied pack: {pack}")
                    else:
                        print(f"❌ Failed to apply pack: {pack}")
            
            # Set neuromod tool for the test
            test.set_neuromod_tool(self.neuromod_tool)
            
            # Run test
            results = test.run_test(self.neuromod_tool)
            comparison_results['combinations'][combo_name] = results
            
            # Clear all packs
            if self.neuromod_tool:
                self.neuromod_tool.clear()
                
        return comparison_results
        
    def list_tests(self):
        """List all tests in the suite"""
        print(f"\n📋 Test Suite ({len(self.tests)} tests):")
        print("=" * 50)
        for i, test in enumerate(self.tests):
            print(f"{i+1}. {test.get_test_name()}")
            
    def cleanup(self):
        """Clean up all tests"""
        for test in self.tests:
            test.cleanup()
