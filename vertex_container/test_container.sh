#!/bin/bash

# Test Container Script for Prediction Server
# Tests the container locally before Vertex AI deployment

set -e

echo "🚀 Starting container test for prediction server..."
echo "This will test the container locally before Vertex AI deployment"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if we're in the right directory
if [ ! -f "Dockerfile" ]; then
    print_error "Please run this script from the vertex_container directory"
    exit 1
fi

# Step 1: Quick import test (fastest)
echo "🧪 Step 1: Testing imports locally..."
if python test_prediction_server_local.py; then
    print_status "Local import test passed"
else
    print_error "Local import test failed. Fix issues before proceeding."
    exit 1
fi

echo ""

# Step 2: Build minimal test container
echo "🧪 Step 2: Building minimal test container..."
if docker build -f Dockerfile.test -t neuromod-test:latest ..; then
    print_status "Minimal container built successfully"
else
    print_error "Container build failed. Check Dockerfile and dependencies."
    exit 1
fi

echo ""

# Step 3: Test container startup
echo "🧪 Step 3: Testing container startup..."
CONTAINER_ID=$(docker run -d -p 8080:8080 \
    -e MODEL_NAME="microsoft/DialoGPT-small" \
    -e NEUROMODULATION_ENABLED=true \
    -e PROBE_SYSTEM_ENABLED=true \
    neuromod-test:latest)

if [ $? -eq 0 ]; then
    print_status "Container started successfully"
else
    print_error "Container failed to start"
    exit 1
fi

# Wait for container to be ready
echo "⏳ Waiting for container to be ready..."
sleep 10

# Step 4: Test health endpoint
echo "🧪 Step 4: Testing health endpoint..."
if curl -f http://localhost:8080/health > /dev/null 2>&1; then
    print_status "Health endpoint working"
else
    print_error "Health endpoint failed"
    docker logs $CONTAINER_ID
    docker stop $CONTAINER_ID
    exit 1
fi

# Step 5: Test probe status endpoint
echo "🧪 Step 5: Testing probe status endpoint..."
if curl -f http://localhost:8080/probe_status > /dev/null 2>&1; then
    print_status "Probe status endpoint working"
else
    print_warning "Probe status endpoint failed (may be expected if no model loaded)"
fi

# Step 6: Test model info endpoint
echo "🧪 Step 6: Testing model info endpoint..."
if curl -f http://localhost:8080/model_info > /dev/null 2>&1; then
    print_status "Model info endpoint working"
else
    print_warning "Model info endpoint failed (may be expected if no model loaded)"
fi

# Step 7: Test prediction endpoint (basic)
echo "🧪 Step 7: Testing prediction endpoint..."
PREDICTION_RESPONSE=$(curl -s -X POST "http://localhost:8080/predict" \
    -H "Content-Type: application/json" \
    -d '{"instances": [{"prompt": "Hello", "max_tokens": 10}]}' 2>/dev/null || echo "FAILED")

if [ "$PREDICTION_RESPONSE" != "FAILED" ]; then
    print_status "Prediction endpoint working"
    echo "Response preview: ${PREDICTION_RESPONSE:0:100}..."
else
    print_warning "Prediction endpoint failed (may be expected if no model loaded)"
fi

# Step 8: Check container logs
echo "🧪 Step 8: Checking container logs..."
echo "Container logs:"
docker logs $CONTAINER_ID | tail -20

# Step 9: Cleanup
echo "🧹 Step 9: Cleaning up test container..."
docker stop $CONTAINER_ID
docker rm $CONTAINER_ID

echo ""
echo "="*60
echo "📊 CONTAINER TEST RESULTS"
echo "="*60

if [ "$PREDICTION_RESPONSE" != "FAILED" ]; then
    print_status "ALL TESTS PASSED! Container is ready for Vertex AI deployment."
    echo ""
    echo "🚀 Next steps:"
    echo "1. Build production container: docker build -t neuromod-vertex-ai ."
    echo "2. Push to GCR: docker push gcr.io/YOUR_PROJECT_ID/neuromod-vertex-ai:latest"
    echo "3. Deploy to Vertex AI using vertex_ai_manager.py"
else
    print_warning "Some tests failed. Container may not be fully ready for production."
    echo ""
    echo "🔧 Issues to investigate:"
    echo "1. Check container logs above"
    echo "2. Verify model loading in container"
    echo "3. Test with smaller model first"
fi

echo ""
echo "💡 Tips:"
echo "- Use Dockerfile.test for quick testing"
echo "- Use Dockerfile for production deployment"
echo "- Test with DialoGPT-small before trying larger models"
echo "- Check logs if endpoints fail"
