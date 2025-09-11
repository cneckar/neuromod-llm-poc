# 🧠 Neuromodulation API

A FastAPI service that provides neuromodulation capabilities to language models, supporting both local models and Vertex AI endpoints with full probe system integration.

## 🚀 Quick Start

### Prerequisites
```bash
# Install dependencies
pip install -r requirements.txt

# For Vertex AI support (optional)
pip install google-cloud-aiplatform
```

### Start the API Server
```bash
cd api
python server.py
```
**Server runs at**: http://localhost:8000

### Start the Web Interface
```bash
cd api
streamlit run web_interface.py --server.port 8501
```
**Web UI runs at**: http://localhost:8501

### Test the System
```bash
# Chat with DMT pack
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello!"}], "pack_name": "dmt"}'

# Check model status
curl http://localhost:8000/model/status

# Get emotion summary
curl http://localhost:8000/emotions/summary
```

## 🏗️ Architecture

### System Components

```
Neuromodulation API
├── server.py               # Main FastAPI server
├── web_interface.py        # Streamlit web UI
├── model_manager.py        # Local model management
├── vertex_ai_manager.py    # Vertex AI integration
└── neuromod/               # Core neuromodulation engine
```

### Model Management

The API supports two model interfaces:

1. **Local Models** (`LocalModelInterface`)
   - Uses Hugging Face transformers
   - Full neuromodulation support (82 packs, 100+ effects)
   - CPU-optimized for local development
   - Fast local inference with real effects

2. **Vertex AI Models** (`VertexAIInterface`)
   - Connects to deployed GPU endpoints
   - Full neuromodulation and probe system
   - GPU acceleration for production
   - Pay-per-use pricing

### Neuromodulation System

- **82 Predefined Packs**: DMT, LSD, caffeine, nicotine, etc.
- **100+ Individual Effects**: Temperature, attention, steering, memory
- **Real-Time Application**: Effects applied during generation
- **Probe Integration**: Behavioral monitoring during inference
- **Emotion Tracking**: 7 latent axes + 12 discrete emotions

## 📡 API Endpoints

### Core Endpoints
| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | API information + model status |
| `/health` | GET | Health check |
| `/model/status` | GET | Model loading status (Local + Vertex AI) |

### Chat & Generation
| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/chat` | POST | Multi-turn chat with neuromodulation |
| `/generate` | POST | Text generation with effects |
| `/emotions/summary` | GET | Emotion tracking summary |

### Pack Management
| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/packs` | GET | List available neuromodulation packs |
| `/packs/{pack_name}/apply` | POST | Apply specific pack |
| `/packs/clear` | POST | Clear current pack |

### Vertex AI Integration
| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/vertex-ai/models` | GET | List available Vertex AI models |
| `/vertex-ai/endpoints` | GET | List active endpoints |
| `/vertex-ai/deploy` | POST | Deploy new model |

## 🔧 Configuration

### Environment Variables
```bash
# API Configuration
PORT=8000
HOST=0.0.0.0
LOG_LEVEL=INFO

# Model Configuration
MODEL_NAME=microsoft/DialoGPT-small
DEVICE=cpu  # or cuda

# Vertex AI Configuration (optional)
GOOGLE_APPLICATION_CREDENTIALS=path/to/service-account.json
VERTEX_AI_PROJECT_ID=your-google-cloud-project-id
VERTEX_AI_LOCATION=us-central1
```

### Model Configuration
```python
# Default model settings
DEFAULT_MODEL_CONFIG = {
    "max_tokens": 100,
    "temperature": 1.0,
    "top_p": 1.0,
    "timeout": 60,
    "max_concurrent_requests": 10
}
```

## 🐳 Deployment

### Local Development
```bash
# Run directly
python server.py

# With uvicorn
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

### Docker Deployment
```bash
# Build image
docker build -t neuromodulation-api .

# Run container
docker run -p 8000:8000 neuromodulation-api
```

### Vertex AI Deployment
```bash
# Deploy to Vertex AI
cd ../vertex_container
bash deploy_vertex_ai.sh deploy

# Connect API to Vertex AI
curl -X POST "http://localhost:8000/vertex-ai/connect" \
  -H "Content-Type: application/json" \
  -d '{"endpoint_url": "https://your-vertex-ai-endpoint.vertex.ai"}'
```

## 🧪 Testing

### Run API Tests
```bash
# From project root
./test --api

# From tests directory
python tests/test_api_servers.py
```

### Test Pack Loading
```bash
# Test local model with pack
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Tell me a story"}], "pack_name": "caffeine"}'

# Test Vertex AI with pack
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Explain quantum physics"}], "pack_name": "lsd", "use_vertex_ai": true}'
```

## 🔍 Troubleshooting

### Common Issues

**Import Errors**
```bash
# Ensure dependencies are installed
pip install -r requirements.txt

# For Vertex AI support
pip install google-cloud-aiplatform
```

**Model Loading Issues**
```bash
# Check model status
curl http://localhost:8000/model/status

# Check logs for specific errors
python server.py --verbose
```

**Vertex AI Connection Issues**
```bash
# Verify credentials
export GOOGLE_APPLICATION_CREDENTIALS=path/to/service-account.json

# Check endpoint status
curl http://localhost:8000/vertex-ai/endpoints
```

### Debug Mode
```bash
# Run with verbose logging
python server.py --verbose

# Check API documentation
curl http://localhost:8000/docs
```

## 📚 Examples

### Python Client
```python
import requests

# Chat with neuromodulation
response = requests.post("http://localhost:8000/chat", json={
    "messages": [{"role": "user", "content": "Write a poem"}],
    "pack_name": "dmt"
})

print(response.json()["response"])
```

### JavaScript Client
```javascript
// Chat with neuromodulation
const response = await fetch("http://localhost:8000/chat", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({
        messages: [{role: "user", content: "Explain consciousness"}],
        pack_name: "lsd"
    })
});

const result = await response.json();
console.log(result.response);
```

## 🎯 Use Cases

### Research & Development
- **Effect Testing**: Test different neuromodulation packs
- **Behavioral Analysis**: Monitor model responses to interventions
- **A/B Testing**: Compare baseline vs. modified models

### Production Applications
- **Content Generation**: Generate creative content with specific effects
- **Conversational AI**: Chat interfaces with personality modification
- **Educational Tools**: Adaptive learning with cognitive enhancement

### Integration
- **Web Applications**: Embed in existing web services
- **Mobile Apps**: Mobile-friendly API endpoints
- **Enterprise Systems**: Scale with Vertex AI deployment

---

**For detailed development information**: See [`neuromod/README.md`](../neuromod/README.md)  
**For testing framework**: See [`tests/README.md`](../tests/README.md)  
**For Vertex AI deployment**: See [`vertex_container/README.md`](../vertex_container/README.md)
