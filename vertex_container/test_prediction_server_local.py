#!/usr/bin/env python3
"""
Local Test Script for Prediction Server
Test the prediction server locally before Docker deployment
"""

import sys
import os
import time
import requests
import json
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test if all required modules can be imported"""
    print("🧪 Testing imports...")
    
    try:
        # Test neuromodulation system
        from neuromod import NeuromodTool
        print("✅ NeuromodTool imported successfully")
        
        from neuromod.testing.simple_emotion_tracker import SimpleEmotionTracker
        print("✅ SimpleEmotionTracker imported successfully")
        
        # Test transformers
        import torch
        print(f"✅ PyTorch imported: {torch.__version__}")
        
        from transformers import AutoTokenizer, AutoModelForCausalLM
        print("✅ Transformers imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_model_loading():
    """Test if we can load a small model locally"""
    print("\n🧪 Testing model loading...")
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        # Use a very small model for testing
        model_name = "microsoft/DialoGPT-small"
        print(f"Loading test model: {model_name}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("✅ Tokenizer loaded successfully")
        
        # Load model with minimal settings
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="cpu",  # Use CPU for testing
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        print("✅ Model loaded successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False

def test_neuromodulation():
    """Test if neuromodulation system works"""
    print("\n🧪 Testing neuromodulation system...")
    
    try:
        from neuromod import NeuromodTool
        
        # Initialize neuromod tool
        neuromod_tool = NeuromodTool()
        print("✅ NeuromodTool initialized")
        
        # Test pack loading
        packs = neuromod_tool.registry.list_packs()
        print(f"✅ Available packs: {len(packs)}")
        
        # Test emotion tracker
        from neuromod.testing.simple_emotion_tracker import SimpleEmotionTracker
        emotion_tracker = SimpleEmotionTracker()
        print("✅ Emotion tracker initialized")
        
        return True
        
    except Exception as e:
        print(f"❌ Neuromodulation test failed: {e}")
        return False

def test_prediction_server_code():
    """Test if the prediction server code can be imported and initialized"""
    print("\n🧪 Testing prediction server code...")
    
    try:
        # Import the prediction server functions
        from prediction_server import (
            load_model, 
            apply_neuromodulation, 
            generate_text_with_probes,
            register_probe_hooks
        )
        print("✅ Prediction server functions imported")
        
        return True
        
    except Exception as e:
        print(f"❌ Prediction server import failed: {e}")
        return False

def test_flask_server():
    """Test if Flask server can start"""
    print("\n🧪 Testing Flask server...")
    
    try:
        from flask import Flask
        app = Flask(__name__)
        
        @app.route('/test')
        def test():
            return {"status": "ok"}
        
        print("✅ Flask app created successfully")
        return True
        
    except Exception as e:
        print(f"❌ Flask test failed: {e}")
        return False

def test_docker_build():
    """Test if Docker build context is correct"""
    print("\n🧪 Testing Docker build context...")
    
    try:
        # Check if required files exist
        required_files = [
            "neuromod/",
            "packs/",
            "vertex_container/requirements.txt",
            "vertex_container/prediction_server.py"
        ]
        
        for file_path in required_files:
            if os.path.exists(file_path):
                print(f"✅ {file_path} exists")
            else:
                print(f"❌ {file_path} missing")
                return False
        
        # Check requirements.txt
        with open("vertex_container/requirements.txt", "r") as f:
            requirements = f.read()
            if "flask" in requirements.lower():
                print("✅ Flask in requirements.txt")
            else:
                print("❌ Flask missing from requirements.txt")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Docker build context test failed: {e}")
        return False

def run_comprehensive_test():
    """Run all tests"""
    print("🚀 Starting comprehensive prediction server test...\n")
    
    tests = [
        ("Import Test", test_imports),
        ("Model Loading Test", test_model_loading),
        ("Neuromodulation Test", test_neuromodulation),
        ("Prediction Server Test", test_prediction_server_code),
        ("Flask Test", test_flask_server),
        ("Docker Build Context Test", test_docker_build)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*50)
    print("📊 TEST RESULTS SUMMARY")
    print("="*50)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if success:
            passed += 1
    
    print(f"\n🎯 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Ready for Docker deployment.")
        return True
    else:
        print("⚠️ Some tests failed. Fix issues before Docker deployment.")
        return False

if __name__ == "__main__":
    # Change to vertex_container directory
    os.chdir(Path(__file__).parent)
    
    success = run_comprehensive_test()
    
    if success:
        print("\n🚀 Next steps:")
        print("1. Build Docker container: docker build -t neuromod-vertex-ai .")
        print("2. Test container locally: docker run -p 8080:8080 neuromod-vertex-ai")
        print("3. Test endpoints: curl http://localhost:8080/health")
        print("4. Deploy to Vertex AI if all local tests pass")
    else:
        print("\n🔧 Fix the failing tests before proceeding with Docker deployment.")
        sys.exit(1)
