#!/usr/bin/env python3
"""
Simple Test Runner for Neuromodulation Tests

Run this from the neuromod/testing directory to execute tests.
"""

import sys
import os
import argparse

# Add paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(parent_dir)
sys.path.extend([current_dir, parent_dir, root_dir])

try:
    from neuromod.pack_system import PackRegistry
    from neuromod.neuromod_tool import NeuromodTool
    
    # Import test classes directly
    from adq_test import ADQTest
    from cdq_test import CDQTest
    from sdq_test import SDQTest
    from ddq_test import DDQTest
    from pdq_test import PDQTest
    from edq_test import EDQTest
    from pcq_pop_test import PCQPopTest
    from story_emotion_test import StoryEmotionTest
    
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running this from the neuromod/testing directory")
    sys.exit(1)


def run_test(test_name, model_name="microsoft/DialoGPT-small", packs=None):
    """Run a single test"""
    
    # Available tests
    available_tests = {
        'adq': ADQTest,
        'cdq': CDQTest,
        'sdq': SDQTest,
        'ddq': DDQTest,
        'pdq': PDQTest,
        'edq': EDQTest,
        'pcq_pop': PCQPopTest,
        'story': StoryEmotionTest
    }
    
    if test_name not in available_tests:
        print(f"Unknown test: {test_name}")
        print(f"Available tests: {list(available_tests.keys())}")
        return
    
    print(f"🚀 Running {test_name.upper()} test with model: {model_name}")
    
    # Create test instance
    test_class = available_tests[test_name]
    test = test_class(model_name)
    
    try:
        # Load model
        test.load_model()
        
        # Create neuromod tool if packs specified
        neuromod_tool = None
        if packs:
            try:
                pack_registry = PackRegistry("../../packs/config.json")
                neuromod_tool = NeuromodTool(pack_registry, test.model, test.tokenizer)
                test.set_neuromod_tool(neuromod_tool)
                
                print(f"🎯 Applying packs: {packs}")
                for pack_name in packs:
                    pack = pack_registry.get_pack(pack_name)
                    if pack:
                        neuromod_tool.apply_pack(pack, intensity=0.7)
                        print(f"   ✅ Applied {pack_name}")
                    else:
                        print(f"   ❌ Pack not found: {pack_name}")
            except Exception as e:
                print(f"⚠️  Could not load packs: {e}")
        
        # Run the test
        results = test.run_test(neuromod_tool)
        
        print(f"\n✅ {test_name.upper()} test completed!")
        print(f"📊 Status: {results.get('status', 'unknown')}")
        
        # Show emotion summary if available
        if 'emotion_tracking' in results:
            emotion_trend = results['emotion_tracking'].get('emotional_trend', 'unknown')
            print(f"🎭 Emotional trend: {emotion_trend}")
        
        return results
        
    except Exception as e:
        print(f"❌ Error running {test_name} test: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        # Cleanup
        test.cleanup()


def list_tests():
    """List available tests"""
    tests = {
        'adq': 'AI Digital Enhancer Detection Questionnaire',
        'cdq': 'Cognitive Distortion Questionnaire',
        'sdq': 'Social Desirability Questionnaire',
        'ddq': 'Digital Dependency Questionnaire',
        'pdq': 'Problematic Digital Use Questionnaire',
        'edq': 'Emotional Digital Use Questionnaire',
        'pcq_pop': 'Population-level Cognitive Assessment',
        'story': 'Story-based Emotion Test'
    }
    
    print("Available tests:")
    for test_name, description in tests.items():
        print(f"  {test_name}: {description}")


def list_packs():
    """List available packs"""
    try:
        pack_registry = PackRegistry("../../packs/config.json")
        print("Available packs:")
        for pack_name in pack_registry.packs.keys():
            print(f"  {pack_name}")
    except Exception as e:
        print(f"Could not load packs: {e}")


def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(description="Run neuromodulation tests")
    parser.add_argument("--model", default="microsoft/DialoGPT-small", help="Model to use")
    parser.add_argument("--test", help="Test to run (adq, cdq, sdq, ddq, pdq, edq, pcq_pop, story)")
    parser.add_argument("--packs", nargs="*", help="Packs to apply during testing")
    parser.add_argument("--list-tests", action="store_true", help="List available tests")
    parser.add_argument("--list-packs", action="store_true", help="List available packs")
    
    args = parser.parse_args()
    
    if args.list_tests:
        list_tests()
    elif args.list_packs:
        list_packs()
    elif args.test:
        run_test(args.test, args.model, args.packs)
    else:
        print("Please specify --test <name>, --list-tests, or --list-packs")
        list_tests()


if __name__ == "__main__":
    main()
