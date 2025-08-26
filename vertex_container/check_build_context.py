#!/usr/bin/env python3
"""
Check Docker Build Context
Verifies that all required files and directories exist for Docker build
"""

import os
import sys
from pathlib import Path

def check_build_context():
    """Check if all required files exist for Docker build"""
    print("🔍 Checking Docker build context...")
    
    # Get project root (parent of vertex_container)
    project_root = Path(__file__).parent.parent
    print(f"Project root: {project_root}")
    
    required_paths = [
        "neuromod/",
        "packs/",
        "vertex_container/Dockerfile",
        "vertex_container/requirements.txt",
        "vertex_container/prediction_server.py"
    ]
    
    all_good = True
    
    for path in required_paths:
        full_path = project_root / path
        if full_path.exists():
            if full_path.is_dir():
                print(f"✅ Directory: {path}")
            else:
                print(f"✅ File: {path}")
        else:
            print(f"❌ Missing: {path}")
            all_good = False
    
    print()
    
    if all_good:
        print("🎉 Build context is ready!")
        print("✅ All required files and directories exist")
        return True
    else:
        print("⚠️  Build context has issues!")
        print("❌ Some required files or directories are missing")
        return False

if __name__ == "__main__":
    success = check_build_context()
    sys.exit(0 if success else 1)
