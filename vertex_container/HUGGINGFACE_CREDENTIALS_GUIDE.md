# 🔐 Hugging Face Credentials Guide for Vertex AI

This guide explains how to securely handle Hugging Face credentials when deploying gated models (like Llama) to Vertex AI without exposing them in your Docker container.

## 🚨 **Why This Matters**

- **Llama models are gated** - require Hugging Face authentication
- **Credentials must be secure** - never commit to version control
- **Vertex AI needs access** - to download models during container startup
- **Environment variables** - provide secure credential injection

## 🏗️ **How It Works**

```
Your .env file → Environment Variables → Vertex AI Deployment → Container Environment
     (local)           (deployment)           (secure)            (runtime)
```

1. **Local**: Store credentials in `.env` file (never committed)
2. **Deployment**: Script reads `.env` and passes to Vertex AI
3. **Vertex AI**: Securely injects credentials as environment variables
4. **Container**: Model loading uses credentials from environment

## 🚀 **Quick Start**

### **Step 1: Get Hugging Face Token**

1. Go to [Hugging Face Settings](https://huggingface.co/settings/tokens)
2. Click "New token"
3. Give it a name (e.g., "Vertex AI Deployment")
4. Set permissions to "Read"
5. Copy the token (starts with `hf_`)

### **Step 2: Set Up Credentials**

```bash
cd vertex_container

# Interactive setup (recommended)
./setup_hf_credentials.sh setup

# Or manually create .env file
cat > .env << EOF
HUGGINGFACE_TOKEN=hf_your_token_here
HUGGINGFACE_USERNAME=your_username
PROJECT_ID=neuromod-469620
REGION=us-central1
MODEL_NAME=meta-llama/Meta-Llama-3.1-8B
EOF
```

### **Step 3: Deploy**

```bash
# Deploy with credentials
./setup_hf_credentials.sh deploy

# Or use the main deployment script
source .env
./deploy_vertex_ai.sh deploy
```

## 🔧 **Detailed Setup**

### **Option 1: Interactive Setup (Recommended)**

```bash
./setup_hf_credentials.sh setup
```

This will:
- Prompt for your Hugging Face token
- Ask for username (optional)
- Create `.env` file automatically
- Add `.env` to `.gitignore`
- Verify credentials work

### **Option 2: Manual Setup**

```bash
# Create .env file
cat > .env << EOF
# Hugging Face Credentials
HUGGINGFACE_TOKEN=hf_your_actual_token_here
HUGGINGFACE_USERNAME=your_username

# Google Cloud Configuration
PROJECT_ID=neuromod-469620
REGION=us-central1
MODEL_NAME=meta-llama/Meta-Llama-3.1-8B
EOF

# Add to .gitignore
echo ".env" >> .gitignore

# Source the file
source .env
```

### **Option 3: Environment Variables Only**

```bash
export HUGGINGFACE_TOKEN="hf_your_token_here"
export HUGGINGFACE_USERNAME="your_username"
export PROJECT_ID="neuromod-469620"
export REGION="us-central1"
export MODEL_NAME="meta-llama/Meta-Llama-3.1-8B"
```

## 🔍 **Verification**

### **Check Current Status**

```bash
./setup_hf_credentials.sh status
```

### **Test Credentials**

```bash
./setup_hf_credentials.sh verify
```

### **Manual Verification**

```bash
# Test token with huggingface_hub
python3 -c "
from huggingface_hub import whoami
try:
    user = whoami(token='$HUGGINGFACE_TOKEN')
    print(f'✅ Token valid for user: {user}')
except Exception as e:
    print(f'❌ Token invalid: {e}')
"
```

## 🚀 **Deployment**

### **Deploy with Credentials**

```bash
# Using the helper script
./setup_hf_credentials.sh deploy

# Or manually
source .env
./deploy_vertex_ai.sh deploy
```

### **What Happens During Deployment**

1. **Script reads `.env`** file
2. **Credentials extracted** and validated
3. **Python script generated** with environment variables
4. **Vertex AI deployment** includes credentials securely
5. **Container starts** with credentials in environment
6. **Model downloads** using authenticated access

## 🔒 **Security Features**

### **Automatic .gitignore**

The setup script automatically adds `.env` to `.gitignore`:

```bash
# Environment variables with sensitive data
.env
```

### **Credential Validation**

- Checks token format (`hf_` prefix)
- Verifies token works with Hugging Face API
- Warns about missing credentials

### **Secure Injection**

- Credentials never stored in Docker image
- Passed securely through Vertex AI environment variables
- Available only at runtime

## 📋 **Environment Variables Reference**

### **Required**

| Variable | Description | Example |
|----------|-------------|---------|
| `HUGGINGFACE_TOKEN` | Your Hugging Face access token | `hf_abc123...` |

### **Optional**

| Variable | Description | Example |
|----------|-------------|---------|
| `HUGGINGFACE_USERNAME` | Your Hugging Face username | `your_username` |
| `PROJECT_ID` | Google Cloud project ID | `neuromod-469620` |
| `REGION` | Google Cloud region | `us-central1` |
| `MODEL_NAME` | Model to deploy | `meta-llama/Meta-Llama-3.1-8B` |

## 🐛 **Troubleshooting**

### **Common Issues**

#### **1. "No module named 'flask'" Error**

This suggests the container build didn't include dependencies properly.

**Solution**: Rebuild the container
```bash
./deploy_vertex_ai.sh build
```

#### **2. "Model loading failed" Error**

This usually means Hugging Face authentication failed.

**Solution**: Verify credentials
```bash
./setup_hf_credentials.sh verify
```

#### **3. "Invalid token" Error**

Your Hugging Face token is incorrect or expired.

**Solution**: 
1. Check token at [Hugging Face Settings](https://huggingface.co/settings/tokens)
2. Generate new token if needed
3. Update `.env` file

#### **4. "Permission denied" Error**

Your token doesn't have access to the model.

**Solution**:
1. Accept model terms at [Llama Model Page](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B)
2. Ensure token has "Read" permissions
3. Wait a few minutes for permissions to propagate

### **Debug Commands**

```bash
# Check environment variables
./setup_hf_credentials.sh status

# Test Hugging Face API
./setup_hf_credentials.sh verify

# View container logs
gcloud ai endpoints predict $ENDPOINT_ID \
    --region=$REGION \
    --instances='{"prompt": "test"}'
```

## 🔄 **Updating Credentials**

### **Token Expired**

```bash
# Update .env file
sed -i 's/HUGGINGFACE_TOKEN=.*/HUGGINGFACE_TOKEN=hf_new_token/' .env

# Verify new token
./setup_hf_credentials.sh verify

# Redeploy
./setup_hf_credentials.sh deploy
```

### **Username Changed**

```bash
# Update .env file
sed -i 's/HUGGINGFACE_USERNAME=.*/HUGGINGFACE_USERNAME=new_username/' .env

# Redeploy
./setup_hf_credentials.sh deploy
```

## 📚 **Advanced Usage**

### **Multiple Models**

```bash
# Deploy different models
export MODEL_NAME="meta-llama/Meta-Llama-3.1-70B"
./deploy_vertex_ai.sh deploy

export MODEL_NAME="meta-llama/Meta-Llama-3.1-8B-Instruct"
./deploy_vertex_ai.sh deploy
```

### **Custom Endpoints**

```bash
# Deploy to custom endpoint
export ENDPOINT_NAME="my-custom-endpoint"
./deploy_vertex_ai.sh deploy
```

### **Different Regions**

```bash
# Deploy to different region
export REGION="us-west1"
./deploy_vertex_ai.sh deploy
```

## 🎯 **Best Practices**

1. **Never commit `.env`** - Always in `.gitignore`
2. **Use read-only tokens** - No need for write permissions
3. **Rotate tokens regularly** - For security
4. **Verify before deploying** - Use verification commands
5. **Monitor deployments** - Check logs for issues
6. **Test locally first** - Use Docker run for testing

## 🔗 **Related Documentation**

- [Vertex AI Deployment Guide](../README.md)
- [Enhanced Model Manager](../../api/README_ENHANCED_MODEL_MANAGER.md)
- [API Examples](API_EXAMPLES.md)
- [Hugging Face Token Settings](https://huggingface.co/settings/tokens)

## 🆘 **Getting Help**

If you encounter issues:

1. **Check status**: `./setup_hf_credentials.sh status`
2. **Verify credentials**: `./setup_hf_credentials.sh verify`
3. **Check logs**: Look at Vertex AI endpoint logs
4. **Common issues**: Review troubleshooting section above
5. **Hugging Face**: Ensure model access is granted

---

**Remember**: Your Hugging Face token is sensitive. Keep it secure and never share it publicly! 🔐
