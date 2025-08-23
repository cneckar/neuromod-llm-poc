#!/usr/bin/env python3
"""
Build Context Preparation for Vertex AI Container
Prepares the build context for the container build
"""

import os
import sys
from pathlib import Path

def prepare_build_context():
    """Prepare the build context for the Vertex AI container"""
    
    # Get the project root directory
    project_root = Path(__file__).parent.parent
    container_dir = Path(__file__).parent
    
    print(f"Project root: {project_root}")
    print(f"Container dir: {container_dir}")
    
    # Verify required files exist in their proper locations
    required_files = [
        project_root / "neuromod",
        project_root / "packs", 
        container_dir / "requirements.txt",
        container_dir / "prediction_server.py",
        container_dir / "test_neuromodulation.py"
    ]
    
    for file_path in required_files:
        if not file_path.exists():
            print(f"Error: Required file/directory not found: {file_path}")
            return False
        else:
            print(f"✓ Found: {file_path}")
    
    print("Build context verification completed successfully")
    print("Docker will use files directly from their proper locations")
    return True

if __name__ == "__main__":
    if prepare_build_context():
        print("Build context preparation completed successfully")
        sys.exit(0)
    else:
        print("Build context preparation failed")
        sys.exit(1)
