#!/bin/bash

# Vertex AI Deployment Script for Pay-Per-Use Neuromodulation
# Deploys custom container with Llama 3 and neuromodulation effects
[INFO] Checking dependencies...
[INFO] Dependencies check passed
[INFO] Deploying to Vertex AI...
Creating Endpoint
Create Endpoint backing LRO: projects/935902481804/locations/us-central1/endpoints/792109067370758144/operations/2602612759602397184
Endpoint created. Resource name: projects/935902481804/locations/us-central1/endpoints/792109067370758144
To use this Endpoint in another session:
endpoint = aiplatform.Endpoint('projects/935902481804/locations/us-central1/endpoints/792109067370758144')
Creating Model
Create Model backing LRO: projects/935902481804/locations/us-central1/models/6905437698270429184/operations/7968651715614343168
Model created. Resource name: projects/935902481804/locations/us-central1/models/6905437698270429184@1
To use this Model in another session:
model = aiplatform.Model('projects/935902481804/locations/us-central1/models/6905437698270429184@1')
Deploying model to Endpoint : projects/935902481804/locations/us-central1/endpoints/792109067370758144
Traceback (most recent call last):
  File "/Users/cris/.pyenv/versions/3.11.5/lib/python3.11/site-packages/google/api_core/grpc_helpers.py", line 76, in error_remapped_callable
    return callable_(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/cris/.pyenv/versions/3.11.5/lib/python3.11/site-packages/grpc/_interceptor.py", line 277, in __call__
    response, ignored_call = self._with_call(
                             ^^^^^^^^^^^^^^^^
  File "/Users/cris/.pyenv/versions/3.11.5/lib/python3.11/site-packages/grpc/_interceptor.py", line 332, in _with_call
    return call.result(), call
           ^^^^^^^^^^^^^
  File "/Users/cris/.pyenv/versions/3.11.5/lib/python3.11/site-packages/grpc/_channel.py", line 440, in result
    raise self
  File "/Users/cris/.pyenv/versions/3.11.5/lib/python3.11/site-packages/grpc/_interceptor.py", line 315, in continuation
    response, call = self._thunk(new_method).with_call(
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/cris/.pyenv/versions/3.11.5/lib/python3.11/site-packages/grpc/_channel.py", line 1198, in with_call
    return _end_unary_response_blocking(state, call, True, None)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/cris/.pyenv/versions/3.11.5/lib/python3.11/site-packages/grpc/_channel.py", line 1006, in _end_unary_response_blocking
    raise _InactiveRpcError(state)  # pytype: disable=not-instantiable
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
grpc._channel._InactiveRpcError: <_InactiveRpcError of RPC that terminated with:
	status = StatusCode.INVALID_ARGUMENT
	details = "`min_replica_count` must be greater than 0."
	debug_error_string = "UNKNOWN:Error received from peer ipv4:142.251.35.170:443 {grpc_message:"`min_replica_count` must be greater than 0.", grpc_status:3, created_time:"2025-08-20T21:10:06.375655-04:00"}"
>

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/cris/src/neuromod-llm-poc/deploy_vertex.py", line 27, in <module>
    model.deploy(
  File "/Users/cris/.pyenv/versions/3.11.5/lib/python3.11/site-packages/google/cloud/aiplatform/models.py", line 5927, in deploy
    return self._deploy(
           ^^^^^^^^^^^^^
  File "/Users/cris/.pyenv/versions/3.11.5/lib/python3.11/site-packages/google/cloud/aiplatform/base.py", line 863, in wrapper
    return method(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/cris/.pyenv/versions/3.11.5/lib/python3.11/site-packages/google/cloud/aiplatform/models.py", line 6189, in _deploy
    endpoint._deploy_call(
  File "/Users/cris/.pyenv/versions/3.11.5/lib/python3.11/site-packages/google/cloud/aiplatform/models.py", line 2183, in _deploy_call
    operation_future = api_client.deploy_model(
                       ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/cris/.pyenv/versions/3.11.5/lib/python3.11/site-packages/google/cloud/aiplatform_v1/services/endpoint_service/client.py", line 1803, in deploy_model
    response = rpc(
               ^^^^
  File "/Users/cris/.pyenv/versions/3.11.5/lib/python3.11/site-packages/google/api_core/gapic_v1/method.py", line 131, in __call__
    return wrapped_func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/cris/.pyenv/versions/3.11.5/lib/python3.11/site-packages/google/api_core/grpc_helpers.py", line 78, in error_remapped_callable
    raise exceptions.from_grpc_error(exc) from exc
google.api_core.exceptions.InvalidArgument: 400 `min_replica_count` must be greater than 0.
set -e

# Configuration
PROJECT_ID=${PROJECT_ID:-"neuromod-469620"}
REGION=${REGION:-"us-central1"}
MODEL_NAME=${MODEL_NAME:-"meta-llama/Meta-Llama-3.1-8B"}
ENDPOINT_NAME=${ENDPOINT_NAME:-"neuromod-llama-endpoint"}
CONTAINER_NAME=${CONTAINER_NAME:-"neuromod-vertex-container"}
# Create a sanitized Docker tag (no forward slashes)
DOCKER_TAG=${DOCKER_TAG:-"llama-3.1-8b"}

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
    
    # Prepare build context
    log_info "Preparing build context..."
    cd vertex_container
    python build_context.py
    if [ $? -ne 0 ]; then
        log_error "Failed to prepare build context"
    fi
    cd ..
    
    # Build container
    docker build -t gcr.io/$PROJECT_ID/$CONTAINER_NAME:$DOCKER_TAG \
        -f vertex_container/Dockerfile \
        --build-arg MODEL_NAME=$MODEL_NAME \
        --platform linux/amd64 \
        vertex_container/
    
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

# Upload model
model = aiplatform.Model.upload(
    display_name='neuromod-$MODEL_NAME',
    serving_container_image_uri="gcr.io/$PROJECT_ID/$CONTAINER_NAME:$DOCKER_TAG",
    serving_container_predict_route="/predict",
    serving_container_health_route="/health",
    serving_container_environment_variables={
        "MODEL_NAME": "$MODEL_NAME",
        "PROJECT_ID": "$PROJECT_ID"
    }
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
