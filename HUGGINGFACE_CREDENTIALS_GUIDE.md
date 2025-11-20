# 🔐 Hugging Face Credentials Guide

This guide explains how to set up Hugging Face credentials for using Llama models with the Neuromod LLM project.

## 🚨 **Why You Need This**

- **Llama models are gated** - require Hugging Face authentication
- **Credentials must be secure** - never commit to version control
- **Local development** - needed for chat interface and API server
- **GPU acceleration** - works with your RTX 4070

## 🚀 **Quick Setup (3 Methods)**

### **Method 1: Automated Setup Script (Recommended)**

```bash
python setup_hf_credentials.py
```

This will:
- Guide you through getting a token
- Create a `.env` file automatically
- Test the connection
- Set up everything for you

### **Method 2: Hugging Face CLI Login (Most Reliable)**

```bash
huggingface-cli login
```

When prompted, paste your token. This saves it permanently.

### **Method 3: Manual .env File**

```bash
# Create .env file
echo "HUGGINGFACE_HUB_TOKEN=your_token_here" > .env
```

## 🔑 **Getting Your Hugging Face Token**

1. **Go to**: https://huggingface.co/settings/tokens
2. **Sign in** to your Hugging Face account
3. **Click "New token"**
4. **Name it**: "neuromod-llm" (or any name)
5. **Select "Read"** access (sufficient for downloading models)
6. **Copy the token** (starts with `hf_...`)

## 🧪 **Testing Your Setup**

### **Test Token Loading**
```bash
python -c "from dotenv import load_dotenv; load_dotenv(); import os; print('Token loaded:', 'HUGGINGFACE_HUB_TOKEN' in os.environ)"
```

### **Test Hugging Face Connection**
```bash
python -c "from huggingface_hub import HfApi; api = HfApi(); print('Connected as:', api.whoami()['name'])"
```

### **Test Model Access**
```bash
python -c "from huggingface_hub import HfApi; api = HfApi(); print('Can access Llama:', api.model_info('meta-llama/Llama-3.1-8B-Instruct').id)"
```

## 🎯 **Using with Neuromod LLM**

### **Chat Interface**
```bash
python demo/chat.py
```
Select option 4 (Llama 2 7B) or 5 (Llama 3.1 8B)

### **API Server**
```bash
cd api
python server.py
```

### **Startup Script**
```bash
python start_neuromod.py
```

## 🔧 **Troubleshooting**

### **"Token is required" Error**
- Make sure your `.env` file exists and has the token
- Try `huggingface-cli login` instead
- Check that the token starts with `hf_`

### **"Cannot access gated repo" Error**
- You need to request access to Llama models
- Go to https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
- Click "Request access"
- Wait for approval (usually instant)

### **"Invalid token" Error**
- Check your token at https://huggingface.co/settings/tokens
- Generate a new token if needed
- Make sure it has "Read" permissions

## 🔒 **Security Best Practices**

1. **Never commit `.env`** - It's already in `.gitignore`
2. **Use read-only tokens** - No need for write permissions
3. **Rotate tokens regularly** - For security
4. **Keep tokens private** - Never share publicly

## 📋 **Environment Variables Reference**

| Variable | Description | Example |
|----------|-------------|---------|
| `HUGGINGFACE_HUB_TOKEN` | Your Hugging Face access token | `hf_abc123...` |

## 🎯 **Best Llama Models for Your RTX 4070**

| Model | Size | VRAM Usage | Quality | Speed | Recommendation |
|-------|------|------------|---------|-------|----------------|
| **Llama 3.1 8B** | 8B | 8-10GB | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | **Best overall** |
| **Llama 2 7B** | 7B | 7-9GB | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | **Fastest** |
| **Llama 2 13B** | 13B | 10-12GB | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | **High quality** |

## 🚀 **Quick Commands**

```bash
# Setup credentials
python setup_hf_credentials.py

# Test connection
python -c "from huggingface_hub import HfApi; api = HfApi(); print('Connected as:', api.whoami()['name'])"

# Run chat with Llama
python demo/chat.py

# Start API server
cd api && python server.py
```

## 🔗 **Related Documentation**

- [Main README](README.md)
- [Vertex AI Credentials Guide](vertex_container/HUGGINGFACE_CREDENTIALS_GUIDE.md)
- [Hugging Face Token Settings](https://huggingface.co/settings/tokens)

---

**Remember**: Your Hugging Face token is sensitive. Keep it secure and never share it publicly! 🔐

