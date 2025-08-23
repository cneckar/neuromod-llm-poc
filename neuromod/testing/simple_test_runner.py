#!/usr/bin/env python3
"""
Simple Test Runner for Core Tests

Provides a clean way to run individual tests (ADQ, CDQ, SDQ, etc.) with emotion tracking.
"""

import sys
import os
import argparse
from typing import Dict, Any, Optional

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adq_test import ADQTest
from cdq_test import CDQTest
from sdq_test import SDQTest
from ddq_test import DDQTest
from pdq_test import PDQTest
from edq_test import EDQTest
from pcq_pop_test import PCQPopTest

try:
    from neuromod import NeuromodTool, PackRegistry
    NEUROMOD_AVAILABLE = True
except ImportError:
    NEUROMOD_AVAILABLE = False
    print("⚠️  Neuromod system not available - running tests without neuromodulation")


class SimpleTestRunner:
    """Simple test runner for core tests"""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-small"):
        self.model_name = model_name
        self.available_tests = {
            'adq': ADQTest,
            'cdq': CDQTest,
            'sdq': SDQTest,
            'ddq': DDQTest,
            'pdq': PDQTest,
            'edq': EDQTest,
            'pcq': PCQPopTest
        }
        
    def list_tests(self):
        """List all available tests"""
        print("📋 Available Tests:")
        print("=" * 40)
        for test_name, test_class in self.available_tests.items():
            test_instance = test_class(self.model_name)
            print(f"  {test_name.upper()}: {test_instance.get_test_name()}")
        print()
        
    def run_test(self, test_name: str, pack_name: Optional[str] = None) -> Dict[str, Any]:
        """Run a specific test"""
        if test_name not in self.available_tests:
            raise ValueError(f"Unknown test: {test_name}. Available: {list(self.available_tests.keys())}")
        
        print(f"🚀 Running {test_name.upper()} Test")
        print("=" * 50)
        
        # Create test instance
        test_class = self.available_tests[test_name]
        test = test_class(self.model_name)
        
        try:
            # Load model
            print(f"📦 Loading model: {self.model_name}")
            test.load_model()
            
            # Setup neuromodulation if available and requested
            neuromod_tool = None
            if NEUROMOD_AVAILABLE and pack_name:
                print(f"🧠 Setting up neuromodulation with pack: {pack_name}")
                try:
                    registry = PackRegistry()
                    neuromod_tool = NeuromodTool()
                    test.set_neuromod_tool(neuromod_tool)
                    
                    # Apply pack
                    pack = registry.get_pack(pack_name)
                    if pack:
                        neuromod_tool.apply_pack(pack, intensity=0.7)
                        print(f"✅ Applied pack: {pack_name}")
                    else:
                        print(f"⚠️  Pack not found: {pack_name}")
                except Exception as e:
                    print(f"⚠️  Failed to setup neuromodulation: {e}")
                    neuromod_tool = None
            
            # Run test
            print(f"\n🧪 Executing {test_name.upper()} test...")
            results = test.run_test(neuromod_tool)
            
            # Show results summary
            print(f"\n📊 Test Results Summary:")
            print(f"   Status: {results.get('status', 'unknown')}")
            if 'emotion_tracking' in results:
                emotion_trend = results['emotion_tracking'].get('emotional_trend', 'unknown')
                print(f"   Emotional Trend: {emotion_trend}")
            
            # Cleanup
            test.cleanup()
            
            return results
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            test.cleanup()
            raise
    
    def run_all_tests(self, pack_name: Optional[str] = None):
        """Run all available tests"""
        print("🎯 Running All Tests")
        print("=" * 50)
        
        results = {}
        for test_name in self.available_tests.keys():
            try:
                print(f"\n{'='*20} {test_name.upper()} {'='*20}")
                result = self.run_test(test_name, pack_name)
                results[test_name] = result
                print(f"✅ {test_name.upper()} completed successfully")
            except Exception as e:
                print(f"❌ {test_name.upper()} failed: {e}")
                results[test_name] = {'status': 'failed', 'error': str(e)}
        
        # Summary
        print(f"\n📋 Overall Summary:")
        print("=" * 30)
        for test_name, result in results.items():
            status = result.get('status', 'unknown')
            if status == 'completed':
                emotion_trend = result.get('emotion_tracking', {}).get('emotional_trend', 'N/A')
                print(f"  {test_name.upper()}: ✅ {status} - {emotion_trend} emotions")
            else:
                print(f"  {test_name.upper()}: ❌ {status}")
        
        return results


def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(description="Simple Test Runner for Core Tests")
    parser.add_argument('--list', action='store_true', help='List available tests')
    parser.add_argument('--test', type=str, help='Run specific test (adq, cdq, sdq, etc.)')
    parser.add_argument('--all', action='store_true', help='Run all tests')
    parser.add_argument('--pack', type=str, help='Apply neuromodulation pack')
    parser.add_argument('--model', type=str, default='microsoft/DialoGPT-small', 
                       help='Model to use (default: microsoft/DialoGPT-small)')
    
    args = parser.parse_args()
    
    # Create runner
    runner = SimpleTestRunner(args.model)
    
    if args.list:
        runner.list_tests()
        return
    
    if args.all:
        runner.run_all_tests(args.pack)
        return
    
    if args.test:
        try:
            runner.run_test(args.test, args.pack)
        except Exception as e:
            print(f"❌ Failed to run test: {e}")
            sys.exit(1)
        return
    
    # No arguments - show help
    parser.print_help()


if __name__ == "__main__":
    main()
