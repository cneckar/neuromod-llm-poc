# Vertex AI Container

This directory contains the custom container for deploying neuromodulated LLMs to Google Cloud Vertex AI.

## 🏗️ **Build and Deploy**

### Quick Start
```bash
# Navigate to vertex_container directory
cd vertex_container

# Full deployment (build, push, deploy)
bash deploy_vertex_ai.sh deploy

# Just build and push container
bash deploy_vertex_ai.sh build

# Just deploy to Vertex AI (if container already built)
bash deploy_vertex_ai.sh deploy-ai
```

### Manual Steps
```bash
# 1. Verify build context (checks that required files exist)
python build_context.py

# 2. Build container (Docker uses files directly from project structure)
docker build -t gcr.io/YOUR_PROJECT_ID/neuromod-vertex-container:llama-3.1-8b -f Dockerfile ..

# 3. Push to Google Container Registry
docker push gcr.io/YOUR_PROJECT_ID/neuromod-vertex-container:llama-3.1-8b

# 4. Deploy to Vertex AI
bash deploy_vertex_ai.sh deploy-ai
```

## 📁 **Files**

- **`Dockerfile`** - Container definition with Rocky Linux 9 base
- **`deploy_vertex_ai.sh`** - Automated deployment script
- **`build_context.py`** - Verifies required files exist in project structure
- **`prediction_server.py`** - Flask server for Vertex AI
- **`requirements.txt`** - Python dependencies
- **`test_neuromodulation.py`** - Tests neuromodulation system during build

## 🏗️ **File Structure**

The container build uses files directly from their proper locations in the project:
- **`../neuromod/`** - Neuromodulation system from project root
- **`../packs/`** - Pack configurations from project root  
- **`requirements.txt`** - Container-specific requirements
- **`prediction_server.py`** - Container-specific prediction server

This ensures **single source of truth** - no duplicate files to maintain!

## ⚙️ **Configuration**

Set environment variables or use command line options:

```bash
# Environment variables
export PROJECT_ID="your-google-cloud-project-id"
export REGION="us-central1"
export MODEL_NAME="meta-llama/Meta-Llama-3.1-8B"

# Or command line options
bash deploy_vertex_ai.sh --project-id your-project --region us-west1 deploy
```

## 🚀 **Deployment Commands**

```bash
# Full deployment
bash deploy_vertex_ai.sh deploy

# Build only
bash deploy_vertex_ai.sh build

# Deploy only (skip building)
bash deploy_vertex_ai.sh deploy-ai

# Test existing deployment
bash deploy_vertex_ai.sh test

# Show help
bash deploy_vertex_ai.sh --help
```

## 🔧 **Prerequisites**

- Google Cloud CLI (`gcloud`) installed and authenticated
- Docker installed and running
- IAM permissions: Storage Admin, AI Platform Developer
- GPU quota available in your region

## 📊 **What Gets Deployed**

- **Container**: Rocky Linux 9 + Python 3.11 + neuromodulation system
- **Model**: Custom container with Llama 3 integration
- **Endpoint**: Vertex AI endpoint with GPU acceleration
- **Scaling**: Single replica for cost control (configurable)

## 🧪 **Testing**

The deployment script automatically tests the endpoint with a sample prediction request using the "caffeine" neuromodulation pack.
