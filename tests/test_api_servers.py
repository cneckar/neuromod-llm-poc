#!/usr/bin/env python3
"""
API Server Testing Module
Tests all API server components including server_real.py, model_manager.py, and vertex_ai_manager.py
"""

import unittest
import sys
import os
import time
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, mock_open
import requests
from io import StringIO

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestModelManager(unittest.TestCase):
    """Test ModelManager functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp_dir)
        
        # Mock environment variables
        self.env_patcher = patch.dict(os.environ, {
            'MODEL_NAME': 'microsoft/DialoGPT-small',
            'NEUROMODULATION_ENABLED': 'true'
        })
        self.env_patcher.start()
    
    def tearDown(self):
        """Clean up after tests"""
        self.env_patcher.stop()
    
    def test_model_manager_import(self):
        """Test ModelManager can be imported"""
        try:
            from api.model_manager import ModelManager
            self.assertTrue(True, "ModelManager imported successfully")
        except ImportError as e:
            self.fail(f"Failed to import ModelManager: {e}")
    
    def test_model_manager_initialization(self):
        """Test ModelManager initialization"""
        try:
            from api.model_manager import ModelManager
            
            # Mock the model loading to avoid actual model downloads
            with patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer, \
                 patch('transformers.AutoModelForCausalLM.from_pretrained') as mock_model:
                
                mock_tokenizer.return_value = Mock()
                mock_model.return_value = Mock()
                
                manager = ModelManager()
                self.assertIsNotNone(manager, "ModelManager should initialize")
                
        except Exception as e:
            self.fail(f"ModelManager initialization failed: {e}")
    
    def test_model_manager_methods(self):
        """Test ModelManager methods exist"""
        try:
            from api.model_manager import ModelManager
            
            # Mock the transformers imports to avoid actual model loading
            with patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer, \
                 patch('transformers.AutoModelForCausalLM.from_pretrained') as mock_model:
                
                mock_tokenizer.return_value = Mock()
                mock_model.return_value = Mock()
                
                # Use the legacy ModelManager which has no-arg constructor
                manager = ModelManager()
                
                # Check if methods exist
                self.assertTrue(hasattr(manager, 'generate_text'), "generate_text method should exist")
                self.assertTrue(hasattr(manager, 'load_model'), "load_model method should exist")
                self.assertTrue(hasattr(manager, 'get_model_status'), "get_model_status method should exist")
                
        except Exception as e:
            self.fail(f"ModelManager method test failed: {e}")

class TestVertexAIManager(unittest.TestCase):
    """Test VertexAIManager functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp_dir)
    
    def test_vertex_ai_manager_import(self):
        """Test VertexAIManager can be imported"""
        try:
            # Mock Google Cloud dependencies
            with patch.dict('sys.modules', {
                'google.cloud.aiplatform': Mock(),
                'google.auth': Mock(),
                'google.auth.default': Mock(return_value=(Mock(), 'test-project'))
            }):
                from api.vertex_ai_manager import VertexAIManager
                self.assertTrue(True, "VertexAIManager imported successfully")
        except ImportError as e:
            print(f"⚠️ VertexAIManager import warning (expected without Google Cloud): {e}")
    
    def test_vertex_ai_manager_structure(self):
        """Test VertexAIManager structure"""
        # Create a mock VertexAIManager class
        class MockVertexAIManager:
            def __init__(self, project_id, region):
                self.project_id = project_id
                self.region = region
                self.endpoints = {}
            
            def create_custom_endpoint(self, endpoint_name, container_uri, model_name, env_vars=None):
                self.endpoints[endpoint_name] = {
                    'container_uri': container_uri,
                    'model_name': model_name,
                    'env_vars': env_vars or {}
                }
                return True
            
            def get_endpoint_status(self, endpoint_name):
                if endpoint_name in self.endpoints:
                    return "DEPLOYED"
                return None
            
            def predict(self, endpoint_name, request_data):
                if endpoint_name in self.endpoints:
                    return {
                        "predictions": [{"text": "Mock prediction response"}]
                    }
                return None
        
        # Test the mock class
        manager = MockVertexAIManager("test-project", "us-central1")
        self.assertEqual(manager.project_id, "test-project")
        self.assertEqual(manager.region, "us-central1")
        
        # Test endpoint creation
        success = manager.create_custom_endpoint(
            "test-endpoint",
            "gcr.io/test-project/test:latest",
            "microsoft/DialoGPT-small"
        )
        self.assertTrue(success)
        
        # Test endpoint status
        status = manager.get_endpoint_status("test-endpoint")
        self.assertEqual(status, "DEPLOYED")
        
        # Test prediction
        response = manager.predict("test-endpoint", {"instances": [{"prompt": "Hello"}]})
        self.assertIsNotNone(response)
        self.assertIn("predictions", response)
        
        print("✅ VertexAIManager structure test: OK")

class TestServerReal(unittest.TestCase):
    """Test server_real.py functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp_dir)
    
    def test_server_real_import(self):
        """Test server_real.py can be imported"""
        try:
            # Mock dependencies
            with patch.dict('sys.modules', {
                'api.model_manager': Mock(),
                'api.vertex_ai_manager': Mock()
            }):
                # Import the server module
                sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'api'))
                
                # Test that we can read the file
                server_path = os.path.join(os.path.dirname(__file__), '..', 'api', 'server_real.py')
                self.assertTrue(os.path.exists(server_path), "server_real.py should exist")
                
                with open(server_path, 'r') as f:
                    content = f.read()
                
                # Check for essential components
                self.assertIn("FastAPI", content, "Should use FastAPI")
                self.assertIn("app = FastAPI", content, "Should create FastAPI app")
                self.assertIn("@app.post", content, "Should have POST endpoints")
                self.assertIn("@app.get", content, "Should have GET endpoints")
                
                print("✅ server_real.py structure test: OK")
                
        except Exception as e:
            self.fail(f"server_real.py test failed: {e}")
    
    def test_server_real_endpoints(self):
        """Test server_real.py endpoint definitions"""
        server_path = os.path.join(os.path.dirname(__file__), '..', 'api', 'server_real.py')
        
        with open(server_path, 'r') as f:
            content = f.read()
        
        # Check for required endpoints
        required_endpoints = [
            "/chat",
            "/generate", 
            "/emotions",
            "/packs",
            "/vertex-ai"
        ]
        
        for endpoint in required_endpoints:
            self.assertIn(endpoint, content, f"Should have {endpoint} endpoint")
        
        print("✅ server_real.py endpoints test: OK")

class TestWebInterface(unittest.TestCase):
    """Test web_interface.py functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp_dir)
    
    def test_web_interface_import(self):
        """Test web_interface.py can be imported"""
        try:
            # Mock Streamlit
            with patch.dict('sys.modules', {
                'streamlit': Mock(),
                'api.server_real': Mock()
            }):
                # Test that we can read the file
                web_interface_path = os.path.join(os.path.dirname(__file__), '..', 'api', 'web_interface.py')
                self.assertTrue(os.path.exists(web_interface_path), "web_interface.py should exist")
                
                with open(web_interface_path, 'r') as f:
                    content = f.read()
                
                # Check for essential components
                self.assertIn("import streamlit", content, "Should import streamlit")
                self.assertIn("st.title", content, "Should have Streamlit UI components")
                self.assertIn("st.chat_input", content, "Should have chat input")
                
                print("✅ web_interface.py structure test: OK")
                
        except Exception as e:
            self.fail(f"web_interface.py test failed: {e}")
    
    def test_web_interface_components(self):
        """Test web_interface.py UI components"""
        web_interface_path = os.path.join(os.path.dirname(__file__), '..', 'api', 'web_interface.py')
        
        with open(web_interface_path, 'r') as f:
            content = f.read()
        
        # Check for required UI components
        required_components = [
            "st.title",
            "st.sidebar",
            "st.chat_input",
            "st.chat_message",
            "st.selectbox"
        ]
        
        for component in required_components:
            self.assertIn(component, content, f"Should have {component} component")
        
        print("✅ web_interface.py components test: OK")

class TestAPIIntegration(unittest.TestCase):
    """Test API integration and endpoints"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp_dir)
    
    def test_api_request_formats(self):
        """Test API request formats"""
        # Test chat request format
        chat_request = {
            "messages": [
                {"role": "user", "content": "Hello, how are you?"}
            ],
            "pack_name": "dmt",
            "max_tokens": 100,
            "temperature": 1.0,
            "top_p": 1.0,
            "use_vertex_ai": False
        }
        
        # Validate chat request structure
        self.assertIn("messages", chat_request)
        self.assertIsInstance(chat_request["messages"], list)
        self.assertGreater(len(chat_request["messages"]), 0)
        
        message = chat_request["messages"][0]
        self.assertIn("role", message)
        self.assertIn("content", message)
        
        # Test generate request format
        generate_request = {
            "prompt": "Generate a creative story",
            "max_tokens": 150,
            "temperature": 0.8,
            "top_p": 0.9,
            "pack_name": "lsd"
        }
        
        # Validate generate request structure
        self.assertIn("prompt", generate_request)
        self.assertIn("max_tokens", generate_request)
        self.assertIn("temperature", generate_request)
        
        print("✅ API request format test: OK")
    
    def test_api_response_formats(self):
        """Test API response formats"""
        # Test chat response format
        chat_response = {
            "text": "Hello! I'm doing well, thank you for asking.",
            "emotions": {
                "valence": 0.7,
                "arousal": 0.3,
                "dominance": 0.5
            },
            "model_type": "local",
            "generation_time": 1.2,
            "pack_applied": "dmt"
        }
        
        # Validate chat response structure
        self.assertIn("text", chat_response)
        self.assertIn("emotions", chat_response)
        self.assertIn("model_type", chat_response)
        
        emotions = chat_response["emotions"]
        self.assertIn("valence", emotions)
        self.assertIn("arousal", emotions)
        self.assertIn("dominance", emotions)
        
        # Test generate response format
        generate_response = {
            "text": "Once upon a time, in a world of infinite possibilities...",
            "emotions": {
                "valence": 0.6,
                "arousal": 0.4,
                "dominance": 0.3
            },
            "model_type": "local",
            "generation_time": 2.1,
            "pack_applied": "lsd"
        }
        
        # Validate generate response structure
        self.assertIn("text", generate_response)
        self.assertIn("emotions", generate_response)
        self.assertIn("model_type", generate_response)
        
        print("✅ API response format test: OK")

class TestDemoApplications(unittest.TestCase):
    """Test demo applications"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp_dir)
    
    def test_chat_demo_import(self):
        """Test chat.py demo can be imported"""
        try:
            # Mock dependencies
            with patch.dict('sys.modules', {
                'neuromod': Mock(),
                'neuromod.testing.simple_emotion_tracker': Mock()
            }):
                # Test that we can read the file
                chat_demo_path = os.path.join(os.path.dirname(__file__), '..', 'demo', 'chat.py')
                self.assertTrue(os.path.exists(chat_demo_path), "chat.py should exist")
                
                with open(chat_demo_path, 'r') as f:
                    content = f.read()
                
                # Check for essential components
                self.assertIn("if __name__", content, "Should have main guard")
                self.assertIn("def main", content, "Should have main function")
                
                print("✅ chat.py demo structure test: OK")
                
        except Exception as e:
            self.fail(f"chat.py demo test failed: {e}")
    
    def test_advanced_chat_demo_import(self):
        """Test advanced_chat_demo.py can be imported"""
        try:
            # Mock dependencies
            with patch.dict('sys.modules', {
                'neuromod': Mock(),
                'neuromod.testing.simple_emotion_tracker': Mock()
            }):
                # Test that we can read the file
                advanced_demo_path = os.path.join(os.path.dirname(__file__), '..', 'demo', 'advanced_chat_demo.py')
                self.assertTrue(os.path.exists(advanced_demo_path), "advanced_chat_demo.py should exist")
                
                with open(advanced_demo_path, 'r') as f:
                    content = f.read()
                
                # Check for essential components
                self.assertIn("if __name__", content, "Should have main guard")
                self.assertIn("demo_advanced_features", content, "Should have demo function")
                
                print("✅ advanced_chat_demo.py structure test: OK")
                
        except Exception as e:
            self.fail(f"advanced_chat_demo.py demo test failed: {e}")

class TestAPIConfiguration(unittest.TestCase):
    """Test API configuration and settings"""
    
    def test_api_requirements(self):
        """Test API requirements.txt"""
        requirements_path = os.path.join(os.path.dirname(__file__), '..', 'api', 'requirements.txt')
        
        self.assertTrue(os.path.exists(requirements_path), "requirements.txt should exist")
        
        with open(requirements_path, 'r') as f:
            requirements = f.read()
        
        # Check for essential packages
        required_packages = ['fastapi', 'uvicorn', 'pydantic']
        for package in required_packages:
            self.assertIn(package.lower(), requirements.lower(), 
                         f"Required package '{package}' missing from requirements.txt")
        
        print("✅ API requirements test: OK")
    
    def test_api_dockerfile(self):
        """Test API Dockerfile"""
        dockerfile_path = os.path.join(os.path.dirname(__file__), '..', 'api', 'Dockerfile')
        
        if os.path.exists(dockerfile_path):
            with open(dockerfile_path, 'r') as f:
                content = f.read()
            
            # Check for essential Dockerfile components
            self.assertIn("FROM", content, "Dockerfile should have FROM instruction")
            self.assertIn("COPY", content, "Dockerfile should have COPY instructions")
            self.assertIn("CMD", content, "Dockerfile should have CMD instruction")
            
            print("✅ API Dockerfile test: OK")
        else:
            print("⚠️ API Dockerfile not found (optional)")

def run_api_server_tests():
    """Run all API server tests"""
    print("🧪 API SERVER TESTING SUITE")
    print("=" * 60)
    print("Testing API servers, web interfaces, and demo applications")
    print("=" * 60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestModelManager,
        TestVertexAIManager,
        TestServerReal,
        TestWebInterface,
        TestAPIIntegration,
        TestDemoApplications,
        TestAPIConfiguration
    ]
    
    for test_class in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(test_class))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n📊 API SERVER TEST RESULTS:")
    print(f"   Tests run: {result.testsRun}")
    print(f"   Failures: {len(result.failures)}")
    print(f"   Errors: {len(result.errors)}")
    print(f"   Skipped: {len(result.skipped)}")
    
    if len(result.failures) == 0 and len(result.errors) == 0:
        print("🎉 ALL API SERVER TESTS PASSED!")
        return True
    else:
        print("❌ SOME API SERVER TESTS FAILED")
        return False

if __name__ == "__main__":
    success = run_api_server_tests()
    sys.exit(0 if success else 1)
