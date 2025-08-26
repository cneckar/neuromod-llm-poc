#!/usr/bin/env python3
"""
🧪 Neuromodulation Testing Framework - Single Entry Point
Run all tests with a simple command: python tests/test.py
"""

import sys
import os
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    """Single entry point for all testing"""
    parser = argparse.ArgumentParser(
        description="🧪 Neuromodulation Testing Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tests/test.py                    # Run all tests (5-10 minutes)
  python tests/test.py --quick            # Quick tests (30 seconds)
  python tests/test.py --core             # Core functionality only
  python tests/test.py --full-stack       # Full-stack tests (no deployment)
  python tests/test.py --api              # API and web interface tests
  python tests/test.py --coverage         # Show what's tested
  python tests/test.py --verbose          # Detailed output
        """
    )
    
    # Test categories
    parser.add_argument("--quick", "-q", action="store_true", 
                       help="Quick test suite (30 seconds)")
    parser.add_argument("--core", action="store_true", 
                       help="Core functionality tests (1-2 minutes)")
    parser.add_argument("--integration", action="store_true", 
                       help="Integration tests (2-3 minutes)")
    parser.add_argument("--full-stack", "-f", action="store_true", 
                       help="Full-stack tests without deployment (2-3 minutes)")
    parser.add_argument("--api", action="store_true", 
                       help="API and web interface tests (1-2 minutes)")
    parser.add_argument("--container", "-c", action="store_true", 
                       help="Container simulation tests (1-2 minutes)")
    
    # Options
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Verbose output with detailed information")
    parser.add_argument("--coverage", action="store_true", 
                       help="Show comprehensive coverage report")
    parser.add_argument("--categories", action="store_true", 
                       help="Show available test categories")
    
    args = parser.parse_args()
    
    # Import the comprehensive test runner
    from run_comprehensive_tests import main as run_comprehensive_main
    
    # Convert arguments to the format expected by run_comprehensive_tests
    comprehensive_args = []
    
    if args.quick:
        comprehensive_args.extend(["--quick"])
    elif args.core:
        comprehensive_args.extend(["--category", "core"])
    elif args.integration:
        comprehensive_args.extend(["--category", "integration"])
    elif args.full_stack:
        comprehensive_args.extend(["--full-stack"])
    elif args.api:
        comprehensive_args.extend(["--category", "api_servers"])
    elif args.container:
        comprehensive_args.extend(["--container"])
    elif args.coverage:
        comprehensive_args.extend(["--coverage"])
    elif args.categories:
        comprehensive_args.extend(["--categories"])
    else:
        # Default to all tests
        comprehensive_args.extend(["--category", "all"])
    
    if args.verbose:
        comprehensive_args.extend(["--verbose"])
    
    # Override sys.argv to pass arguments to the comprehensive runner
    original_argv = sys.argv
    sys.argv = ["run_comprehensive_tests.py"] + comprehensive_args
    
    try:
        run_comprehensive_main()
    finally:
        sys.argv = original_argv

if __name__ == "__main__":
    main()
