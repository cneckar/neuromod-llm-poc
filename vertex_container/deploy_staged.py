#!/usr/bin/env python3
"""
Staged Deployment Script for Vertex AI
Implements the staged deployment strategy to minimize risks
"""

import sys
import os
import time
import subprocess
import json
import requests
from pathlib import Path
from typing import Dict, List, Optional

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class StagedDeployment:
    def __init__(self, project_id: str, region: str = "us-central1"):
        self.project_id = project_id
        self.region = region
        self.gcr_url = f"gcr.io/{project_id}"
        
    def log(self, message: str, level: str = "INFO"):
        """Log message with timestamp"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {level}: {message}")
        
    def run_command(self, command: List[str], timeout: int = 300) -> bool:
        """Run shell command and return success status"""
        try:
            self.log(f"Running: {' '.join(command)}")
            result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
            
            if result.returncode == 0:
                self.log(f"Command succeeded: {' '.join(command)}")
                return True
            else:
                self.log(f"Command failed: {result.stderr}", "ERROR")
                return False
                
        except subprocess.TimeoutExpired:
            self.log(f"Command timed out: {' '.join(command)}", "ERROR")
            return False
        except Exception as e:
            self.log(f"Command error: {str(e)}", "ERROR")
            return False
            
    def build_and_push_container(self, dockerfile: str, tag: str) -> bool:
        """Build and push Docker container"""
        # Build container
        build_cmd = [
            "docker", "build", 
            "-f", dockerfile,
            "-t", f"{self.gcr_url}/{tag}:latest",
            ".."
        ]
        
        if not self.run_command(build_cmd, timeout=600):  # 10 minutes
            return False
            
        # Push container
        push_cmd = ["docker", "push", f"{self.gcr_url}/{tag}:latest"]
        return self.run_command(push_cmd, timeout=300)  # 5 minutes
        
    def deploy_to_vertex_ai(self, endpoint_name: str, container_tag: str, 
                           model_name: str, env_vars: Dict[str, str] = None) -> bool:
        """Deploy container to Vertex AI"""
        try:
            from api.vertex_ai_manager import VertexAIManager
            
            manager = VertexAIManager(self.project_id, self.region)
            
            # Create endpoint
            success = manager.create_custom_endpoint(
                endpoint_name=endpoint_name,
                container_uri=f"{self.gcr_url}/{container_tag}:latest",
                model_name=model_name,
                env_vars=env_vars
            )
            
            if success:
                self.log(f"Successfully deployed {endpoint_name} to Vertex AI")
                return True
            else:
                self.log(f"Failed to deploy {endpoint_name} to Vertex AI", "ERROR")
                return False
                
        except Exception as e:
            self.log(f"Error deploying to Vertex AI: {str(e)}", "ERROR")
            return False
            
    def test_endpoint(self, endpoint_name: str, timeout: int = 300) -> bool:
        """Test Vertex AI endpoint"""
        try:
            from api.vertex_ai_manager import VertexAIManager
            
            manager = VertexAIManager(self.project_id, self.region)
            
            # Wait for endpoint to be ready
            self.log(f"Waiting for {endpoint_name} to be ready...")
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                status = manager.get_endpoint_status(endpoint_name)
                if status and "DEPLOYED" in status:
                    self.log(f"{endpoint_name} is ready!")
                    break
                time.sleep(30)
            else:
                self.log(f"{endpoint_name} did not become ready within {timeout}s", "ERROR")
                return False
                
            # Test prediction
            test_request = {
                "instances": [{
                    "prompt": "Hello, this is a test",
                    "max_tokens": 10
                }]
            }
            
            response = manager.predict(endpoint_name, test_request)
            if response and "predictions" in response:
                self.log(f"Test prediction successful: {response['predictions'][0]}")
                return True
            else:
                self.log("Test prediction failed", "ERROR")
                return False
                
        except Exception as e:
            self.log(f"Error testing endpoint: {str(e)}", "ERROR")
            return False
            
    def deploy_stage_1(self) -> bool:
        """Deploy Stage 1: Minimal Container"""
        self.log("🚀 Starting Stage 1: Minimal Container Deployment")
        
        # Build and push minimal container
        if not self.build_and_push_container("Dockerfile.test", "neuromod-minimal"):
            return False
            
        # Deploy to Vertex AI
        env_vars = {
            "MODEL_NAME": "microsoft/DialoGPT-small",
            "NEUROMODULATION_ENABLED": "false",
            "PROBE_SYSTEM_ENABLED": "false",
            "EMOTION_TRACKING_ENABLED": "false"
        }
        
        if not self.deploy_to_vertex_ai("neuromod-minimal", "neuromod-minimal", 
                                       "microsoft/DialoGPT-small", env_vars):
            return False
            
        # Test endpoint
        if not self.test_endpoint("neuromod-minimal"):
            return False
            
        self.log("✅ Stage 1 deployment successful!")
        return True
        
    def deploy_stage_2(self) -> bool:
        """Deploy Stage 2: Full Container"""
        self.log("🚀 Starting Stage 2: Full Container Deployment")
        
        # Build and push full container
        if not self.build_and_push_container("Dockerfile", "neuromod-full"):
            return False
            
        # Deploy to Vertex AI
        env_vars = {
            "MODEL_NAME": "microsoft/DialoGPT-small",
            "NEUROMODULATION_ENABLED": "true",
            "PROBE_SYSTEM_ENABLED": "true",
            "EMOTION_TRACKING_ENABLED": "true"
        }
        
        if not self.deploy_to_vertex_ai("neuromod-full", "neuromod-full", 
                                       "microsoft/DialoGPT-small", env_vars):
            return False
            
        # Test endpoint
        if not self.test_endpoint("neuromod-full"):
            return False
            
        # Test neuromodulation features
        if not self.test_neuromodulation_features("neuromod-full"):
            return False
            
        self.log("✅ Stage 2 deployment successful!")
        return True
        
    def deploy_stage_3(self) -> bool:
        """Deploy Stage 3: Production Model"""
        self.log("🚀 Starting Stage 3: Production Model Deployment")
        
        # Deploy with production model
        env_vars = {
            "MODEL_NAME": "meta-llama/Meta-Llama-3.1-8B",
            "NEUROMODULATION_ENABLED": "true",
            "PROBE_SYSTEM_ENABLED": "true",
            "EMOTION_TRACKING_ENABLED": "true"
        }
        
        if not self.deploy_to_vertex_ai("neuromod-production", "neuromod-full", 
                                       "meta-llama/Meta-Llama-3.1-8B", env_vars):
            return False
            
        # Test endpoint
        if not self.test_endpoint("neuromod-production", timeout=600):  # 10 minutes for large model
            return False
            
        # Test performance
        if not self.test_performance("neuromod-production"):
            return False
            
        self.log("✅ Stage 3 deployment successful!")
        return True
        
    def test_neuromodulation_features(self, endpoint_name: str) -> bool:
        """Test neuromodulation features"""
        try:
            from api.vertex_ai_manager import VertexAIManager
            
            manager = VertexAIManager(self.project_id, self.region)
            
            # Test with DMT pack
            test_request = {
                "instances": [{
                    "prompt": "Hello, how are you?",
                    "max_tokens": 20,
                    "pack_name": "dmt"
                }]
            }
            
            response = manager.predict(endpoint_name, test_request)
            if response and "predictions" in response:
                self.log("Neuromodulation test successful")
                return True
            else:
                self.log("Neuromodulation test failed", "ERROR")
                return False
                
        except Exception as e:
            self.log(f"Error testing neuromodulation: {str(e)}", "ERROR")
            return False
            
    def test_performance(self, endpoint_name: str) -> bool:
        """Test performance with load testing"""
        try:
            from api.vertex_ai_manager import VertexAIManager
            
            manager = VertexAIManager(self.project_id, self.region)
            
            # Run load test
            self.log("Running performance test...")
            start_time = time.time()
            
            test_request = {
                "instances": [{
                    "prompt": "This is a performance test",
                    "max_tokens": 50
                }]
            }
            
            # Test multiple requests
            for i in range(5):
                response = manager.predict(endpoint_name, test_request)
                if not response:
                    self.log(f"Performance test failed on request {i+1}", "ERROR")
                    return False
                    
            total_time = time.time() - start_time
            avg_time = total_time / 5
            
            if avg_time < 30:  # Less than 30 seconds average
                self.log(f"Performance test passed: {avg_time:.1f}s average")
                return True
            else:
                self.log(f"Performance test failed: {avg_time:.1f}s average (too slow)", "ERROR")
                return False
                
        except Exception as e:
            self.log(f"Error in performance test: {str(e)}", "ERROR")
            return False
            
    def run_staged_deployment(self, stages: List[int] = None) -> bool:
        """Run staged deployment"""
        if stages is None:
            stages = [1, 2, 3]
            
        self.log(f"Starting staged deployment: stages {stages}")
        
        for stage in stages:
            self.log(f"Deploying Stage {stage}...")
            
            if stage == 1:
                if not self.deploy_stage_1():
                    self.log(f"Stage {stage} failed - stopping deployment", "ERROR")
                    return False
            elif stage == 2:
                if not self.deploy_stage_2():
                    self.log(f"Stage {stage} failed - stopping deployment", "ERROR")
                    return False
            elif stage == 3:
                if not self.deploy_stage_3():
                    self.log(f"Stage {stage} failed - stopping deployment", "ERROR")
                    return False
                    
            self.log(f"Stage {stage} completed successfully")
            
            # Wait between stages
            if stage < max(stages):
                self.log("Waiting 2 minutes before next stage...")
                time.sleep(120)
                
        self.log("🎉 All stages completed successfully!")
        return True
        
    def rollback_stage(self, stage: int) -> bool:
        """Rollback to previous stage"""
        self.log(f"Rolling back Stage {stage}...")
        
        if stage == 1:
            # Rollback to minimal container
            return self.deploy_stage_1()
        elif stage == 2:
            # Rollback to Stage 1
            return self.deploy_stage_1()
        elif stage == 3:
            # Rollback to Stage 2
            return self.deploy_stage_2()
        else:
            self.log(f"Invalid stage for rollback: {stage}", "ERROR")
            return False

def main():
    if len(sys.argv) < 2:
        print("Usage: python deploy_staged.py <project_id> [stages]")
        print("Example: python deploy_staged.py my-project-id 1,2,3")
        sys.exit(1)
        
    project_id = sys.argv[1]
    stages = [1, 2, 3]  # Default to all stages
    
    if len(sys.argv) > 2:
        try:
            stages = [int(s) for s in sys.argv[2].split(",")]
        except ValueError:
            print("Invalid stages format. Use comma-separated numbers: 1,2,3")
            sys.exit(1)
            
    deployer = StagedDeployment(project_id)
    
    print("🚀 Starting Staged Vertex AI Deployment")
    print(f"Project ID: {project_id}")
    print(f"Stages: {stages}")
    print("="*50)
    
    success = deployer.run_staged_deployment(stages)
    
    if success:
        print("\n🎉 Deployment completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Deployment failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
