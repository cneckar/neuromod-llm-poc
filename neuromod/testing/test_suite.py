#!/usr/bin/env python3
"""
Test Suite for Neuromodulation Tests

Manages running multiple tests with different pack combinations
"""

import sys
import os
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .test_runner import TestRunner


class TestSuite:
    """Test suite for managing multiple neuromodulation tests"""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-small", packs_file: str = "packs/config.json"):
        self.model_name = model_name
        self.packs_file = packs_file
        self.runner = TestRunner(model_name, packs_file)
        self.results = []
    
    def run_single_test(self, test_name: str, packs: List[str] = None) -> Dict[str, Any]:
        """Run a single test with specified packs"""
        return self.runner.run_test(test_name, packs)
    
    def run_test_sequence(self, test_names: List[str] = None, packs: List[str] = None) -> Dict[str, Any]:
        """Run a sequence of tests with the same pack configuration"""
        if test_names is None:
            test_names = list(self.runner.available_tests.keys())
        
        print(f"🧪 Running test sequence: {test_names}")
        print(f"📦 With packs: {packs or 'None'}")
        
        sequence_results = {}
        for test_name in test_names:
            print(f"\n{'='*50}")
            result = self.runner.run_test(test_name, packs)
            sequence_results[test_name] = result
            self.results.append({
                'timestamp': datetime.now().isoformat(),
                'test_name': test_name,
                'packs': packs or [],
                'result': result
            })
        
        return sequence_results
    
    def run_comparison(self, test_name: str, pack_combinations: List[List[str]]) -> Dict[str, Any]:
        """Run a test with different pack combinations for comparison"""
        print(f"🔬 Running comparison test: {test_name}")
        print(f"📊 Pack combinations: {pack_combinations}")
        
        comparison_results = {}
        for i, packs in enumerate(pack_combinations):
            pack_label = f"combination_{i}" if packs else "baseline"
            print(f"\n--- Running {pack_label}: {packs or 'No packs'} ---")
            
            result = self.runner.run_test(test_name, packs)
            comparison_results[pack_label] = {
                'packs': packs or [],
                'result': result
            }
            
            self.results.append({
                'timestamp': datetime.now().isoformat(),
                'test_name': test_name,
                'packs': packs or [],
                'pack_combination_label': pack_label,
                'result': result
            })
        
        return comparison_results
    
    def run_comprehensive_study(self, 
                               test_names: List[str] = None,
                               pack_combinations: List[List[str]] = None) -> Dict[str, Any]:
        """Run a comprehensive study with multiple tests and pack combinations"""
        if test_names is None:
            test_names = ['adq', 'cdq', 'sdq']  # Core tests
        
        if pack_combinations is None:
            pack_combinations = [
                [],  # Baseline
                ['joy'],  # Single pack
                ['caffeine'],  # Different pack
                ['joy', 'caffeine']  # Multiple packs
            ]
        
        print(f"🔬 Running comprehensive study")
        print(f"🧪 Tests: {test_names}")
        print(f"📦 Pack combinations: {pack_combinations}")
        
        study_results = {}
        
        for test_name in test_names:
            print(f"\n{'='*60}")
            print(f"🧪 Testing: {test_name.upper()}")
            
            test_results = self.run_comparison(test_name, pack_combinations)
            study_results[test_name] = test_results
        
        print(f"\n{'='*60}")
        print("🎉 Comprehensive study completed!")
        
        return study_results
    
    def export_results(self, filename: str = None):
        """Export all results to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"test_suite_results_{timestamp}.json"
        
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'model_name': self.model_name,
            'packs_file': self.packs_file,
            'total_results': len(self.results),
            'results': self.results
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"💾 Results exported to: {filename}")
        return filename
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all test results"""
        if not self.results:
            return {'message': 'No results available'}
        
        summary = {
            'total_tests': len(self.results),
            'successful_tests': sum(1 for r in self.results if r['result'].get('status') != 'error'),
            'failed_tests': sum(1 for r in self.results if r['result'].get('status') == 'error'),
            'tests_by_name': {},
            'packs_tested': set()
        }
        
        # Group by test name
        for result in self.results:
            test_name = result['test_name']
            if test_name not in summary['tests_by_name']:
                summary['tests_by_name'][test_name] = []
            summary['tests_by_name'][test_name].append(result)
            
            # Track packs
            for pack in result['packs']:
                summary['packs_tested'].add(pack)
        
        summary['packs_tested'] = list(summary['packs_tested'])
        return summary
    
    def print_summary(self):
        """Print a formatted summary of results"""
        summary = self.get_summary()
        
        if 'message' in summary:
            print(summary['message'])
            return
        
        print("\n📊 Test Suite Summary")
        print("="*50)
        print(f"Total tests run: {summary['total_tests']}")
        print(f"Successful: {summary['successful_tests']}")
        print(f"Failed: {summary['failed_tests']}")
        print(f"Success rate: {summary['successful_tests']/summary['total_tests']*100:.1f}%")
        
        print(f"\n🧪 Tests by name:")
        for test_name, results in summary['tests_by_name'].items():
            successful = sum(1 for r in results if r['result'].get('status') != 'error')
            total = len(results)
            print(f"  {test_name}: {successful}/{total} successful")
        
        print(f"\n📦 Packs tested: {', '.join(summary['packs_tested']) if summary['packs_tested'] else 'None'}")


def main():
    """Command line interface for test suite"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run test suite")
    parser.add_argument("--model", default="microsoft/DialoGPT-small", help="Model to use")
    parser.add_argument("--packs", default="packs/config.json", help="Packs configuration file")
    parser.add_argument("--mode", choices=["single", "sequence", "comparison", "comprehensive"], 
                       default="single", help="Test mode")
    parser.add_argument("--test", help="Test name for single/comparison mode")
    parser.add_argument("--tests", nargs="*", help="Test names for sequence mode")
    parser.add_argument("--packs-to-apply", nargs="*", help="Packs to apply")
    parser.add_argument("--export", help="Export results to file")
    
    args = parser.parse_args()
    
    suite = TestSuite(args.model, args.packs)
    
    if args.mode == "single":
        if not args.test:
            print("Error: --test required for single mode")
            return
        result = suite.run_single_test(args.test, args.packs_to_apply)
        print(f"\n🎯 Single test result: {result.get('status', 'unknown')}")
    
    elif args.mode == "sequence":
        result = suite.run_test_sequence(args.tests, args.packs_to_apply)
        suite.print_summary()
    
    elif args.mode == "comparison":
        if not args.test:
            print("Error: --test required for comparison mode")
            return
        pack_combinations = [
            [],  # Baseline
            args.packs_to_apply or []  # With packs
        ]
        result = suite.run_comparison(args.test, pack_combinations)
        suite.print_summary()
    
    elif args.mode == "comprehensive":
        result = suite.run_comprehensive_study(args.tests)
        suite.print_summary()
    
    if args.export:
        suite.export_results(args.export)


if __name__ == "__main__":
    main()
