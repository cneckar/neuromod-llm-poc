#!/usr/bin/env python3
"""
Build Context Preparation for Vertex AI Container
Copies necessary files to build the container with neuromodulation support
"""

import os
import shutil
import sys
from pathlib import Path

def prepare_build_context():
    """Prepare the build context for the Vertex AI container"""
    
    # Get the project root directory
    project_root = Path(__file__).parent.parent
    container_dir = Path(__file__).parent
    
    print(f"Project root: {project_root}")
    print(f"Container dir: {container_dir}")
    
    # Create build context directory
    build_context = container_dir / "build_context"
    if build_context.exists():
        shutil.rmtree(build_context)
    build_context.mkdir(exist_ok=True)
    
    # Copy neuromodulation system
    neuromod_src = project_root / "neuromod"
    neuromod_dst = build_context / "neuromod"
    if neuromod_src.exists():
        shutil.copytree(neuromod_src, neuromod_dst)
        print(f"Copied neuromodulation system to {neuromod_dst}")
    else:
        print(f"Warning: neuromodulation system not found at {neuromod_src}")
    
    # Copy packs
    packs_src = project_root / "packs"
    packs_dst = build_context / "packs"
    if packs_src.exists():
        shutil.copytree(packs_src, packs_dst)
        print(f"Copied packs to {packs_dst}")
    else:
        print(f"Warning: packs not found at {packs_src}")
    
    # Copy requirements
    requirements_src = container_dir / "requirements.txt"
    requirements_dst = build_context / "requirements.txt"
    if requirements_src.exists():
        shutil.copy2(requirements_src, requirements_dst)
        print(f"Copied requirements to {requirements_dst}")
    else:
        print(f"Warning: requirements.txt not found at {requirements_src}")
    
    # Copy prediction server
    server_src = container_dir / "prediction_server.py"
    server_dst = build_context / "prediction_server.py"
    if server_src.exists():
        shutil.copy2(server_src, server_dst)
        print(f"Copied prediction server to {server_dst}")
    else:
        print(f"Error: prediction_server.py not found at {server_src}")
        return False
    
    # Copy test script
    test_src = container_dir / "test_neuromodulation.py"
    test_dst = build_context / "test_neuromodulation.py"
    if test_src.exists():
        shutil.copy2(test_src, test_dst)
        print(f"Copied test script to {test_dst}")
    else:
        print(f"Warning: test script not found at {test_src}")
    
    print(f"Build context prepared at: {build_context}")
    return True

if __name__ == "__main__":
    if prepare_build_context():
        print("Build context preparation completed successfully")
        sys.exit(0)
    else:
        print("Build context preparation failed")
        sys.exit(1)
