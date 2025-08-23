#!/usr/bin/env python3
"""
🧪 Neuromodulation Testing - Project Root Entry Point
Simple wrapper to run tests from anywhere in the project
"""

import os
import sys
import subprocess

def main():
    # Get the directory where this script is located (project root)
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # Path to the actual test script
    test_script = os.path.join(project_root, "tests", "test.py")
    
    # Check if test script exists
    if not os.path.exists(test_script):
        print("❌ Test script not found at:", test_script)
        sys.exit(1)
    
    # Run the test script with all arguments passed through
    cmd = [sys.executable, test_script] + sys.argv[1:]
    
    try:
        result = subprocess.run(cmd, cwd=project_root)
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        print("\n⚠️ Testing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
