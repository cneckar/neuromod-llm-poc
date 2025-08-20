#!/bin/bash

# Neuromodulation API Deployment Script
# Supports multiple cloud platforms

set -e

# Configuration
PROJECT_ID=${PROJECT_ID:-"your-project-id"}
REGION=${REGION:-"us-central1"}
SERVICE_NAME=${SERVICE_NAME:-"neuromodulation-api"}
IMAGE_NAME=${IMAGE_NAME:-"gcr.io/$PROJECT_ID/$SERVICE_NAME"}

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
}

# Check dependencies
check_dependencies() {
    log_info "Checking dependencies..."
    
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi
    
    if ! command -v gcloud &> /dev/null; then
        log_error "Google Cloud SDK is not installed"
        exit 1
    fi
    
    log_info "Dependencies check passed"
}

# Build Docker image
build_image() {
    log_info "Building Docker image..."
    
    if [ ! -f "Dockerfile" ]; then
        log_error "Dockerfile not found"
        exit 1
    fi
    
    docker build -t $IMAGE_NAME .
    log_info "Docker image built successfully"
}

# Push to Google Container Registry
push_image() {
    log_info "Pushing image to Google Container Registry..."
    
    # Configure Docker to use gcloud as a credential helper
    gcloud auth configure-docker
    
    # Push the image
    docker push $IMAGE_NAME
    log_info "Image pushed successfully"
}

# Deploy to Cloud Run
deploy_cloud_run() {
    log_info "Deploying to Cloud Run..."
    
    gcloud run deploy $SERVICE_NAME \
        --image $IMAGE_NAME \
        --platform managed \
        --region $REGION \
        --memory 4Gi \
        --cpu 2 \
        --timeout 300 \
        --concurrency 10 \
        --allow-unauthenticated \
        --port 8000
    
    log_info "Deployment completed successfully"
    
    # Get the service URL
    SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region=$REGION --format="value(status.url)")
    log_info "Service URL: $SERVICE_URL"
}

# Deploy to local Docker
deploy_local() {
    log_info "Deploying locally with Docker..."
    
    # Stop existing container if running
    docker stop $SERVICE_NAME 2>/dev/null || true
    docker rm $SERVICE_NAME 2>/dev/null || true
    
    # Run the container
    docker run -d \
        --name $SERVICE_NAME \
        -p 8000:8000 \
        --restart unless-stopped \
        $IMAGE_NAME
    
    log_info "Local deployment completed"
    log_info "API available at: http://localhost:8000"
}

# Deploy web interface
deploy_web_interface() {
    log_info "Deploying web interface..."
    
    # Check if Streamlit is available
    if [ ! -f "web_interface.py" ]; then
        log_warn "Web interface not found, skipping"
        return
    fi
    
    # Deploy to Streamlit Cloud (if configured)
    if [ ! -z "$STREAMLIT_CLOUD_TOKEN" ]; then
        log_info "Deploying to Streamlit Cloud..."
        # Add Streamlit Cloud deployment logic here
    else
        log_info "Running web interface locally..."
        streamlit run web_interface.py --server.port 8501 &
        log_info "Web interface available at: http://localhost:8501"
    fi
}

# Test deployment
test_deployment() {
    log_info "Testing deployment..."
    
    # Wait for service to be ready
    sleep 10
    
    # Test health endpoint
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        log_info "Health check passed"
    else
        log_error "Health check failed"
        exit 1
    fi
    
    # Test packs endpoint
    if curl -f http://localhost:8000/packs > /dev/null 2>&1; then
        log_info "Packs endpoint test passed"
    else
        log_error "Packs endpoint test failed"
        exit 1
    fi
    
    log_info "All tests passed"
}

# Cleanup
cleanup() {
    log_info "Cleaning up..."
    
    # Remove local container
    docker stop $SERVICE_NAME 2>/dev/null || true
    docker rm $SERVICE_NAME 2>/dev/null || true
    
    # Remove local image
    docker rmi $IMAGE_NAME 2>/dev/null || true
    
    log_info "Cleanup completed"
}

# Show usage
show_usage() {
    echo "Usage: $0 [OPTIONS] COMMAND"
    echo ""
    echo "Commands:"
    echo "  build       Build Docker image"
    echo "  push        Push image to registry"
    echo "  deploy      Deploy to Cloud Run"
    echo "  local       Deploy locally with Docker"
    echo "  web         Deploy web interface"
    echo "  test        Test deployment"
    echo "  cleanup     Clean up local resources"
    echo "  full        Full deployment (build, push, deploy)"
    echo ""
    echo "Options:"
    echo "  --project-id PROJECT_ID    Google Cloud project ID"
    echo "  --region REGION           Google Cloud region"
    echo "  --service-name NAME       Service name"
    echo "  --help                    Show this help"
    echo ""
    echo "Environment Variables:"
    echo "  PROJECT_ID                Google Cloud project ID"
    echo "  REGION                    Google Cloud region"
    echo "  SERVICE_NAME              Service name"
    echo "  STREAMLIT_CLOUD_TOKEN     Streamlit Cloud token"
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
        --service-name)
            SERVICE_NAME="$2"
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

# Update IMAGE_NAME with current PROJECT_ID
IMAGE_NAME="gcr.io/$PROJECT_ID/$SERVICE_NAME"

# Execute command
case $COMMAND in
    build)
        check_dependencies
        build_image
        ;;
    push)
        check_dependencies
        push_image
        ;;
    deploy)
        check_dependencies
        deploy_cloud_run
        ;;
    local)
        check_dependencies
        build_image
        deploy_local
        ;;
    web)
        deploy_web_interface
        ;;
    test)
        test_deployment
        ;;
    cleanup)
        cleanup
        ;;
    full)
        check_dependencies
        build_image
        push_image
        deploy_cloud_run
        deploy_web_interface
        test_deployment
        ;;
    *)
        log_error "Unknown command: $COMMAND"
        show_usage
        exit 1
        ;;
esac

log_info "Deployment script completed successfully"
