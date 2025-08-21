#!/bin/bash

# Setup Hugging Face Credentials for Vertex AI Deployment
# This script helps you set up credentials and deploy securely

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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

log_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Check if .env file exists
check_env_file() {
    if [ -f ".env" ]; then
        log_info "Found .env file"
        source .env
        return 0
    else
        log_warn "No .env file found"
        return 1
    fi
}

# Interactive credential setup
setup_credentials_interactive() {
    log_step "Setting up Hugging Face credentials interactively..."
    
    echo ""
    echo "To use Llama models, you need Hugging Face credentials:"
    echo "1. Go to https://huggingface.co/settings/tokens"
    echo "2. Create a new token with 'read' permissions"
    echo "3. Copy the token (starts with 'hf_')"
    echo ""
    
    read -p "Enter your Hugging Face token: " hf_token
    
    if [ -z "$hf_token" ]; then
        log_error "Token cannot be empty"
        exit 1
    fi
    
    if [[ ! "$hf_token" =~ ^hf_ ]]; then
        log_warn "Token should start with 'hf_'. Please verify your token."
    fi
    
    read -p "Enter your Hugging Face username (optional): " hf_username
    
    # Export credentials
    export HUGGINGFACE_TOKEN="$hf_token"
    if [ -n "$hf_username" ]; then
        export HUGGINGFACE_USERNAME="$hf_username"
    fi
    
    log_info "Credentials set for this session"
    
    # Ask if user wants to save to .env
    read -p "Save credentials to .env file? (y/n): " save_env
    
    if [[ "$save_env" =~ ^[Yy]$ ]]; then
        save_to_env_file "$hf_token" "$hf_username"
    fi
}

# Save credentials to .env file
save_to_env_file() {
    local token="$1"
    local username="$2"
    
    log_step "Saving credentials to .env file..."
    
    # Create or update .env file
    cat > .env << EOF
# Hugging Face Credentials for Vertex AI Deployment
# These credentials are used to access gated models like Llama
HUGGINGFACE_TOKEN=$token
EOF

    if [ -n "$username" ]; then
        echo "HUGGINGFACE_USERNAME=$username" >> .env
    fi
    
    echo "" >> .env
    echo "# Google Cloud Configuration" >> .env
    echo "PROJECT_ID=neuromod-469620" >> .env
    echo "REGION=us-central1" >> .env
    echo "MODEL_NAME=meta-llama/Meta-Llama-3.1-8B" >> .env
    
    log_info "Credentials saved to .env file"
    log_warn "Remember to add .env to .gitignore to keep credentials secure!"
    
    # Check if .gitignore contains .env
    if [ -f ".gitignore" ] && grep -q "^\.env$" .gitignore; then
        log_info ".env is already in .gitignore"
    else
        log_warn "Adding .env to .gitignore for security"
        echo "" >> .gitignore
        echo "# Environment variables with sensitive data" >> .gitignore
        echo ".env" >> .gitignore
    fi
}

# Verify credentials
verify_credentials() {
    log_step "Verifying Hugging Face credentials..."
    
    if [ -z "$HUGGINGFACE_TOKEN" ]; then
        log_error "HUGGINGFACE_TOKEN not set"
        return 1
    fi
    
    # Test token with huggingface_hub library
    log_info "Testing Hugging Face token..."
    
    # Create a temporary Python script to test credentials
    cat > test_credentials.py << EOF
import os
import sys

try:
    from huggingface_hub import whoami
    user = whoami(token="$HUGGINGFACE_TOKEN")
    print(f"SUCCESS:{user}")
except ImportError:
    print("ERROR:huggingface_hub not installed")
    sys.exit(1)
except Exception as e:
    print(f"ERROR:{str(e)}")
    sys.exit(1)
EOF

    # Run the test
    response=$(python3 test_credentials.py 2>/dev/null || echo "ERROR:Python execution failed")
    
    # Clean up
    rm -f test_credentials.py
    
    if [[ "$response" == "ERROR:"* ]]; then
        log_error "Failed to verify token: ${response#ERROR:}"
        return 1
    fi
    
    if [[ "$response" == "SUCCESS:"* ]]; then
        username="${response#SUCCESS:}"
        log_info "✅ Hugging Face token verified successfully"
        log_info "Username: $username"
        return 0
    fi
    
    log_error "Unexpected response format: $response"
    return 1
}

# Deploy with credentials
deploy_with_credentials() {
    log_step "Deploying to Vertex AI with Hugging Face credentials..."
    
    if ! verify_credentials; then
        log_error "Cannot deploy without valid credentials"
        exit 1
    fi
    
    # Run deployment
    log_info "Starting deployment..."
    ./deploy_vertex_ai.sh deploy
    
    log_info "Deployment completed!"
}

# Show current status
show_status() {
    log_step "Current configuration status:"
    
    echo ""
    echo "Environment Variables:"
    if [ -n "$HUGGINGFACE_TOKEN" ]; then
        echo "  HUGGINGFACE_TOKEN: ✅ Set (${HUGGINGFACE_TOKEN:0:10}...)"
    else
        echo "  HUGGINGFACE_TOKEN: ❌ Not set"
    fi
    
    if [ -n "$HUGGINGFACE_USERNAME" ]; then
        echo "  HUGGINGFACE_USERNAME: ✅ Set ($HUGGINGFACE_USERNAME)"
    else
        echo "  HUGGINGFACE_USERNAME: ❌ Not set"
    fi
    
    if [ -n "$PROJECT_ID" ]; then
        echo "  PROJECT_ID: ✅ Set ($PROJECT_ID)"
    else
        echo "  PROJECT_ID: ❌ Not set"
    fi
    
    if [ -n "$REGION" ]; then
        echo "  REGION: ✅ Set ($REGION)"
    else
        echo "  REGION: ❌ Not set"
    fi
    
    echo ""
    echo "Files:"
    if [ -f ".env" ]; then
        echo "  .env: ✅ Exists"
    else
        echo "  .env: ❌ Missing"
    fi
    
    if [ -f ".gitignore" ] && grep -q "^\.env$" .gitignore; then
        echo "  .gitignore: ✅ Contains .env"
    else
        echo "  .gitignore: ❌ Missing .env entry"
    fi
}

# Show usage
show_usage() {
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  setup       Interactive credential setup"
    echo "  verify      Verify current credentials"
    echo "  deploy      Deploy to Vertex AI with credentials"
    echo "  status      Show current configuration status"
    echo "  help        Show this help"
    echo ""
    echo "Examples:"
    echo "  $0 setup      # Set up credentials interactively"
    echo "  $0 verify     # Test current credentials"
    echo "  $0 deploy     # Deploy with current credentials"
    echo "  $0 status     # Check configuration"
}

# Main execution
case "${1:-help}" in
    setup)
        setup_credentials_interactive
        ;;
    verify)
        check_env_file
        verify_credentials
        ;;
    deploy)
        check_env_file
        deploy_with_credentials
        ;;
    status)
        check_env_file
        show_status
        ;;
    help|--help|-h)
        show_usage
        ;;
    *)
        log_error "Unknown command: $1"
        show_usage
        exit 1
        ;;
esac
