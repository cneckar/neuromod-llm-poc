# Neuromodulation API

Cloud-native FastAPI service for exposing neuromodulation capabilities to language models.

## 🏗️ Architecture

This API provides a RESTful interface for:
- **Chat Interface**: Multi-turn conversations with neuromodulation effects
- **Pack Management**: Apply/clear predefined neuromodulation packs
- **Effect Management**: Apply individual effects with custom parameters
- **Text Generation**: Simple text generation with effects
- **Model Management**: Load and manage language models

## 🚀 Quick Start

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the API server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Run the web interface
streamlit run web_interface.py
```

### Docker Deployment

```bash
# Build the image
docker build -t neuromodulation-api .

# Run the container
docker run -p 8000:8000 neuromodulation-api

# Access the API
curl http://localhost:8000/health
```

## 📡 API Endpoints

### Health & Status
- `GET /health` - Health check
- `GET /status` - System status
- `GET /model/status` - Model loading status

### Pack Management
- `GET /packs` - List all available packs
- `GET /packs/{pack_name}` - Get pack details
- `POST /packs/{pack_name}/apply` - Apply a pack
- `POST /packs/clear` - Clear current pack

### Effect Management
- `GET /effects` - List all available effects
- `POST /effects/apply` - Apply individual effect

### Generation
- `POST /chat` - Multi-turn chat with effects
- `POST /generate` - Simple text generation

### Model Management
- `POST /model/load` - Load a model (placeholder)

## 🔧 Configuration

### Environment Variables
```bash
# API Configuration
PORT=8000
HOST=0.0.0.0

# Model Configuration
MODEL_NAME=microsoft/DialoGPT-medium
DEVICE=cuda  # or cpu

# Logging
LOG_LEVEL=INFO
```

### API Configuration
The API supports configuration through environment variables or a config file:

```python
# Example configuration
API_CONFIG = {
    "max_tokens": 100,
    "temperature": 1.0,
    "top_p": 1.0,
    "timeout": 60,
    "max_concurrent_requests": 10
}
```

## 🐳 Cloud Deployment

### Google Cloud Run

```bash
# Build and push to Google Container Registry
gcloud builds submit --tag gcr.io/PROJECT_ID/neuromodulation-api

# Deploy to Cloud Run
gcloud run deploy neuromodulation-api \
  --image gcr.io/PROJECT_ID/neuromodulation-api \
  --platform managed \
  --region us-central1 \
  --memory 4Gi \
  --cpu 2 \
  --timeout 300 \
  --concurrency 10
```

### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - PORT=8000
      - MODEL_NAME=microsoft/DialoGPT-medium
    volumes:
      - ./models:/app/models
    restart: unless-stopped

  web:
    build: 
      context: .
      dockerfile: Dockerfile.web
    ports:
      - "8501:8501"
    environment:
      - API_BASE_URL=http://api:8000
    depends_on:
      - api
```

## 🔌 Integration Examples

### Python Client

```python
import requests

# Initialize client
API_BASE = "http://localhost:8000"

# Check health
response = requests.get(f"{API_BASE}/health")
print(response.json())

# List packs
packs = requests.get(f"{API_BASE}/packs").json()
print(f"Available packs: {len(packs)}")

# Apply a pack
requests.post(f"{API_BASE}/packs/caffeine/apply")

# Generate text
response = requests.post(
    f"{API_BASE}/generate",
    params={
        "prompt": "Tell me about coffee",
        "pack_name": "caffeine",
        "max_tokens": 100
    }
)
print(response.json()["generated_text"])
```

### JavaScript Client

```javascript
// Initialize client
const API_BASE = "http://localhost:8000";

// Check health
const health = await fetch(`${API_BASE}/health`);
console.log(await health.json());

// Apply pack
await fetch(`${API_BASE}/packs/caffeine/apply`, { method: 'POST' });

// Generate text
const response = await fetch(`${API_BASE}/generate`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    prompt: "Tell me about coffee",
    pack_name: "caffeine",
    max_tokens: 100
  })
});
const result = await response.json();
console.log(result.generated_text);
```

### cURL Examples

```bash
# Health check
curl http://localhost:8000/health

# List packs
curl http://localhost:8000/packs

# Apply caffeine pack
curl -X POST http://localhost:8000/packs/caffeine/apply

# Generate text
curl -X POST "http://localhost:8000/generate?prompt=Tell%20me%20about%20coffee&pack_name=caffeine&max_tokens=100"

# Chat interface
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Hello, how are you?"}
    ],
    "pack_name": "caffeine",
    "max_tokens": 100
  }'
```

## 🧪 Testing

### Run Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio httpx

# Run tests
pytest tests/

# Run with coverage
pytest --cov=main tests/
```

### Test Examples

```python
# test_api.py
import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_list_packs():
    response = client.get("/packs")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_apply_pack():
    response = client.post("/packs/caffeine/apply")
    assert response.status_code == 200
    assert "applied successfully" in response.json()["message"]
```

## 📊 Monitoring

### Health Checks
The API includes built-in health checks:
- `/health` - Basic health status
- Docker health check with curl
- Kubernetes liveness/readiness probes

### Logging
Structured logging with different levels:
```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

### Metrics
Key metrics to monitor:
- Request latency
- Error rates
- Model loading status
- Pack application success rate
- Generation time

## 🔒 Security

### CORS Configuration
```python
# Configure CORS for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

### Authentication
For production, add authentication:
```python
# Example with API keys
from fastapi import Security, HTTPException
from fastapi.security import HTTPBearer

security = HTTPBearer()

@app.post("/chat")
async def chat(
    request: ChatRequest,
    token: str = Security(security)
):
    if not validate_token(token):
        raise HTTPException(status_code=401, detail="Invalid token")
    # ... rest of function
```

## 🚀 Performance Optimization

### Caching
```python
from functools import lru_cache

@lru_cache(maxsize=100)
def get_pack_config(pack_name: str):
    # Cache pack configurations
    pass
```

### Async Processing
```python
import asyncio

async def generate_text_async(prompt: str):
    # Async text generation
    return await asyncio.to_thread(generate_text, prompt)
```

### Resource Management
- Memory limits for model loading
- CPU allocation for generation
- Timeout handling for long requests
- Connection pooling for external APIs

## 🔧 Troubleshooting

### Common Issues

1. **Model not loaded**
   - Check `/model/status` endpoint
   - Verify model path and permissions
   - Check GPU memory availability

2. **Pack application fails**
   - Verify pack exists in `/packs` endpoint
   - Check pack configuration format
   - Review error logs

3. **Generation timeout**
   - Increase timeout settings
   - Check model performance
   - Monitor resource usage

### Debug Mode
```bash
# Run with debug logging
LOG_LEVEL=DEBUG uvicorn main:app --reload

# Enable FastAPI debug mode
uvicorn main:app --reload --log-level debug
```

## 📈 Scaling

### Horizontal Scaling
- Deploy multiple API instances
- Use load balancer
- Implement session management

### Vertical Scaling
- Increase CPU/memory allocation
- Use GPU instances for model inference
- Optimize model loading

### Auto-scaling
- Configure Cloud Run auto-scaling
- Set appropriate concurrency limits
- Monitor resource utilization

## 🔄 Updates

### API Versioning
```python
# Version your API
app = FastAPI(
    title="Neuromodulation API",
    version="1.0.0",
    openapi_url="/api/v1/openapi.json"
)
```

### Migration Guide
When updating the API:
1. Maintain backward compatibility
2. Document breaking changes
3. Provide migration scripts
4. Test thoroughly

## 📚 Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Docker Documentation](https://docs.docker.com/)
- [Google Cloud Run Documentation](https://cloud.google.com/run/docs)
