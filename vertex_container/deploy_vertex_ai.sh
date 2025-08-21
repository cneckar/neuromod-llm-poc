#!/bin/bash

# Vertex AI Deployment Script for Pay-Per-Use Neuromodulation
# Deploys custom container with Llama 3 and neuromodulation effects

set -e

# Configuration
PROJECT_ID=${PROJECT_ID:-"neuromod-469620"}
REGION=${REGION:-"us-central1"}
MODEL_NAME=${MODEL_NAME:-"meta-llama/Meta-Llama-3.1-8B"}
ENDPOINT_NAME=${ENDPOINT_NAME:-"neuromod-vertex-container"}
CONTAINER_NAME=${CONTAINER_NAME:-"neuromod-vertex-container"}
# Create a sanitized Docker tag (no forward slashes)
DOCKER_TAG=${DOCKER_TAG:-"llama-3.1-8b"}

# Hugging Face credentials (read from environment)
HUGGINGFACE_TOKEN=${HUGGINGFACE_TOKEN:-""}
HUGGINGFACE_USERNAME=${HUGGINGFACE_USERNAME:-""}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

# Load environment variables from .env file if it exists
if [ -f ".env" ]; then
    log_info "Loading environment variables from .env file"
    export $(grep -v '^#' .env | xargs)
fi

# Check dependencies
check_dependencies() {
    log_info "Checking dependencies..."
    
    if ! command -v gcloud &> /dev/null; then
        log_error "Google Cloud SDK is not installed"
    fi
    
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
    fi
    
    # Check if user is authenticated
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
        log_error "Not authenticated with Google Cloud. Run 'gcloud auth login'"
    fi
    
    # Check Hugging Face credentials for gated models
    if [[ "$MODEL_NAME" == *"llama"* ]] || [[ "$MODEL_NAME" == *"Llama"* ]]; then
        if [ -z "$HUGGINGFACE_TOKEN" ]; then
            log_warn "HUGGINGFACE_TOKEN not set. Llama models require authentication."
            log_warn "Set HUGGINGFACE_TOKEN environment variable or add to .env file"
            log_warn "You can still proceed, but model loading may fail"
        fi
        
        if [ -z "$HUGGINGFACE_USERNAME" ]; then
            log_warn "HUGGINGFACE_USERNAME not set. Some models may require this."
        fi
    fi
    
    log_info "Dependencies check passed"
}

# Configure project
configure_project() {
    log_info "Configuring Google Cloud project..."
    
    # Set project
    gcloud config set project $PROJECT_ID
    
    # Enable required APIs
    gcloud services enable aiplatform.googleapis.com
    gcloud services enable containerregistry.googleapis.com
    gcloud services enable cloudbuild.googleapis.com
    
    log_info "Project configured successfully"
}

# Build and push container
build_container() {
    log_info "Building and pushing custom container (Rocky Linux 9 base)..."
    
    # Verify build context
    log_info "Verifying build context..."
    python build_context.py
    if [ $? -ne 0 ]; then
        log_error "Build context verification failed"
    fi
    
    # Build container (Docker will use files directly from project structure)
    docker build -t gcr.io/$PROJECT_ID/$CONTAINER_NAME:$DOCKER_TAG \
        -f Dockerfile \
        --build-arg MODEL_NAME=$MODEL_NAME \
        --platform linux/amd64 \
        ..
    
    # Push to Container Registry
    docker push gcr.io/$PROJECT_ID/$CONTAINER_NAME:$DOCKER_TAG
    
    log_info "Container built and pushed successfully"
}

# Deploy to Vertex AI
deploy_to_vertex_ai() {
    log_info "Deploying to Vertex AI..."
    
    # Create deployment script
    cat > deploy_vertex.py << EOF
import os
from google.cloud import aiplatform

# Initialize Vertex AI
aiplatform.init(project='$PROJECT_ID', location='$REGION')

# Create endpoint
endpoint = aiplatform.Endpoint.create(
    display_name='$ENDPOINT_NAME',
    project='$PROJECT_ID',
    location='$REGION'
)

# Prepare environment variables
env_vars = {
    "MODEL_NAME": "$MODEL_NAME",
    "PROJECT_ID": "$PROJECT_ID"
EOF

    # Add Hugging Face credentials if available
    if [ -n "$HUGGINGFACE_TOKEN" ]; then
        cat >> deploy_vertex.py << EOF
    ,"HUGGINGFACE_TOKEN": "$HUGGINGFACE_TOKEN"
EOF
    fi
    
    if [ -n "$HUGGINGFACE_USERNAME" ]; then
        cat >> deploy_vertex.py << EOF
    ,"HUGGINGFACE_USERNAME": "$HUGGINGFACE_USERNAME"
EOF
    fi
    
    cat >> deploy_vertex.py << EOF
}

# Upload model
model = aiplatform.Model.upload(
    display_name='neuromod-$MODEL_NAME',
    serving_container_image_uri="gcr.io/$PROJECT_ID/$CONTAINER_NAME:$DOCKER_TAG",
    serving_container_predict_route="/predict",
    serving_container_health_route="/health",
    serving_container_environment_variables=env_vars
)

# Deploy model to endpoint
model.deploy(
    endpoint=endpoint,
    machine_type="n1-standard-4",
    accelerator_type="NVIDIA_TESLA_T4",
    accelerator_count=1,
    min_replica_count=1,  # Must be at least 1
    max_replica_count=1   # Single replica for cost control
)

print(f"Endpoint created: {endpoint.resource_name}")
print(f"Model deployed: {model.resource_name}")
EOF

    # Run deployment
    python deploy_vertex.py
    
    log_info "Deployment completed successfully"
}

# Test deployment
test_deployment() {
    log_info "Testing deployment..."
    
    # Wait for endpoint to be ready
    sleep 30
    
    # Get endpoint details
    ENDPOINT_ID=$(gcloud ai endpoints list --region=$REGION --filter="displayName=$ENDPOINT_NAME" --format="value(name)")
    
    if [ -z "$ENDPOINT_ID" ]; then
        log_error "Endpoint not found"
    fi
    
    log_info "Endpoint ID: $ENDPOINT_ID"
    
    # Test prediction
    cat > test_prediction.py << EOF
import os
from google.cloud import aiplatform

# Initialize Vertex AI
aiplatform.init(project='$PROJECT_ID', location='$REGION')

# Get endpoint
endpoint = aiplatform.Endpoint(endpoint_name='$ENDPOINT_ID')

# Test prediction
prediction_request = {
    "instances": [{
        "prompt": "Tell me about coffee",
        "max_tokens": 50,
        "temperature": 1.0,
        "top_p": 1.0,
        "pack_name": "caffeine"
    }]
}

response = endpoint.predict(prediction_request)
print("Prediction response:")
print(response.predictions)
EOF

    python test_prediction.py
    
    log_info "Test completed successfully"
}

# Show usage
show_usage() {
    echo "Usage: $0 [OPTIONS] COMMAND"
    echo ""
    echo "Commands:"
    echo "  deploy      Full deployment (build, push, deploy)"
    echo "  build       Build and push container"
    echo "  deploy-ai   Deploy to Vertex AI"
    echo "  test        Test deployment"
    echo "  cleanup     Clean up resources"
    echo ""
    echo "Options:"
    echo "  --project-id PROJECT_ID    Google Cloud project ID"
    echo "  --region REGION           Google Cloud region"
    echo "  --model-name NAME         Model name to deploy"
    echo "  --endpoint-name NAME      Endpoint name"
    echo "  --help                    Show this help"
    echo ""
    echo "Environment Variables:"
    echo "  PROJECT_ID                Google Cloud project ID"
    echo "  REGION                    Google Cloud region"
    echo "  MODEL_NAME                Model name to deploy"
    echo "  ENDPOINT_NAME             Endpoint name"
    echo "  HUGGINGFACE_TOKEN         Hugging Face access token (for gated models)"
    echo "  HUGGINGFACE_USERNAME      Hugging Face username (for some models)"
    echo ""
    echo "Example with Hugging Face credentials:"
    echo "  export HUGGINGFACE_TOKEN='hf_...'"
    echo "  export HUGGINGFACE_USERNAME='your_username'"
    echo "  ./deploy_vertex_ai.sh deploy"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --project-id)
            PROJECT_ID="$2"
            shift 2
            ;;
        --region)
            REGION="$2"
            shift 2
            ;;
        --model-name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --endpoint-name)
            ENDPOINT_NAME="$2"
            shift 2
            ;;
        --help)
            show_usage
            exit 0
            ;;
        *)
            COMMAND="$1"
            shift
            ;;
    esac
done

# Execute command
case $COMMAND in
    deploy)
        check_dependencies
        configure_project
        build_container
        deploy_to_vertex_ai
        test_deployment
        ;;
    build)
        check_dependencies
        configure_project
        build_container
        ;;
    deploy-ai)
        check_dependencies
        deploy_to_vertex_ai
        ;;
    test)
        test_deployment
        ;;
    cleanup)
        log_info "Cleaning up resources..."
        # Add cleanup logic here
        ;;
    *)
        log_error "Unknown command: $COMMAND"
        show_usage
        exit 1
        ;;
esac

log_info "Vertex AI deployment script completed successfully"
