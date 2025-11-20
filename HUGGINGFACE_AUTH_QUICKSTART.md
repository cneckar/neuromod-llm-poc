# HuggingFace Authentication Quick Start

## For Meta Llama Models (30B, 70B, etc.)

Meta Llama models require authentication. Follow these steps:

### Step 1: Accept License Agreement

1. Visit the model page on HuggingFace:
   - **30B**: https://huggingface.co/meta-llama/Llama-3.1-30B-Instruct
   - **70B**: https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct

2. Click **"Agree and access repository"**
3. Read and accept the license terms

### Step 2: Get Your Access Token

1. Go to: https://huggingface.co/settings/tokens
2. Click **"New token"**
3. Name it (e.g., "neuromod-llm")
4. Select **"Read"** access (sufficient for downloading models)
5. Click **"Generate token"**
6. **Copy the token immediately** (you won't see it again!)

### Step 3: Set Environment Variable

#### Windows (PowerShell):
```powershell
$env:HUGGINGFACE_HUB_TOKEN="hf_your_token_here"
```

#### Windows (CMD):
```cmd
set HUGGINGFACE_HUB_TOKEN=hf_your_token_here
```

#### Linux/Mac:
```bash
export HUGGINGFACE_HUB_TOKEN="hf_your_token_here"
```

#### Make it Permanent (Windows):
```powershell
[System.Environment]::SetEnvironmentVariable('HUGGINGFACE_HUB_TOKEN', 'hf_your_token_here', 'User')
```

#### Make it Permanent (Linux/Mac):
Add to `~/.bashrc` or `~/.zshrc`:
```bash
export HUGGINGFACE_HUB_TOKEN="hf_your_token_here"
```

### Step 4: Verify Authentication

```bash
# Test with HuggingFace CLI
pip install huggingface-hub
huggingface-cli whoami
# Should show your username

# Or test model access
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-30B-Instruct')"
```

## Alternative: Use HuggingFace CLI Login

Instead of environment variables, you can use the CLI:

```bash
pip install huggingface-hub
huggingface-cli login
# Enter your token when prompted
```

This stores the token in `~/.huggingface/token` (Linux/Mac) or `%USERPROFILE%\.huggingface\token` (Windows).

## Alternative Models (No Authentication Required)

If you want to avoid authentication, use these open-source alternatives:

- **Qwen/Qwen-2.5-30B-Instruct** - No auth required, similar quality
- **Qwen/Qwen-2.5-7B-Instruct** - Smaller, no auth required
- **mistralai/Mistral-7B-Instruct-v0.2** - No auth required

## Troubleshooting

### "401 Unauthorized" Error
- Ensure you've accepted the license on HuggingFace
- Verify your token is correct
- Check that the token has read access

### "Model not found" Error
- Make sure you've accepted the license agreement
- Try accessing the model page directly in your browser
- Verify the model name is correct

### Token Not Working
- Regenerate a new token
- Ensure no extra spaces in the token
- Check that the environment variable is set correctly:
  ```bash
  # Windows PowerShell
  echo $env:HUGGINGFACE_HUB_TOKEN
  
  # Linux/Mac
  echo $HUGGINGFACE_HUB_TOKEN
  ```

## Next Steps

Once authenticated, you can run the chat interface:

```bash
python demo/chat.py
# Select option 4 to see available models
# Choose meta-llama/Llama-3.1-30B-Instruct
```

