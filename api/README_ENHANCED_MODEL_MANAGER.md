# Enhanced Model Manager

The Enhanced Model Manager provides a unified interface for working with both local models and Vertex AI endpoints, with full neuromodulation support for remote models.

## 🎯 **Features**

- **🔄 Dual Interface Support**: Local models (transformers) + Vertex AI endpoints
- **🧠 Full Neuromodulation**: All 82 packs and 100+ effects via Vertex AI
- **⚡ Seamless Switching**: Switch between local and remote models
- **🔒 Backward Compatibility**: Existing code continues to work
- **🛡️ Authentication**: Automatic Google Cloud authentication for Vertex AI

## 🏗️ **Architecture**

```
EnhancedModelManager
├── LocalModelInterface     # Local transformers models
└── VertexAIInterface      # Remote Vertex AI endpoints
```

### **Interface Types**

1. **Local Models** (`LocalModelInterface`)
   - Uses Hugging Face transformers
   - CPU-optimized for Cloud Run
   - No neuromodulation support
   - Fast local inference

2. **Vertex AI Models** (`VertexAIInterface`)
   - Connects to deployed endpoints
   - Full neuromodulation support
   - GPU acceleration
   - Pay-per-use pricing

## 🚀 **Quick Start**

### **Basic Usage**

```python
from model_manager import enhanced_model_manager

# Load local model
enhanced_model_manager.load_local_model("gpt2")

# Generate text
text = enhanced_model_manager.generate_text(
    prompt="Hello world",
    max_tokens=50
)

# Switch to Vertex AI
enhanced_model_manager.connect_vertex_ai(
    endpoint_url="https://your-endpoint.vertex.ai",
    project_id="your-project-id"
)

# Generate with neuromodulation
text = enhanced_model_manager.generate_text(
    prompt="Write a story",
    pack_name="caffeine"
)
```

### **Interface Switching**

```python
# Start with local model
enhanced_model_manager.load_local_model("distilgpt2")

# Switch to Vertex AI
enhanced_model_manager.connect_vertex_ai(
    endpoint_url="https://endpoint.vertex.ai",
    project_id="project-id"
)

# Switch back to local
enhanced_model_manager.load_local_model("gpt2")
```

## 📚 **API Reference**

### **EnhancedModelManager**

#### **Core Methods**

```python
# Local model management
load_local_model(model_name: str) -> bool

# Vertex AI management  
connect_vertex_ai(endpoint_url: str, project_id: str, location: str = "us-central1") -> bool

# Text generation (unified interface)
generate_text(
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 1.0,
    top_p: float = 1.0,
    pack_name: str = None,
    custom_pack: Dict = None,
    individual_effects: List[Dict] = None,
    multiple_packs: List[str] = None
) -> str

# Status and information
get_status() -> Dict[str, Any]
is_available() -> bool

# Neuromodulation (Vertex AI only)
get_available_packs() -> List[str]
get_available_effects() -> List[str]

# Resource management
unload_current_interface()
```

#### **Model Compatibility**

```python
# List compatible local models
list_compatible_models() -> List[Dict[str, Any]]

# Check if model is compatible
is_model_compatible(model_name: str) -> bool

# Get model information
get_model_info(model_name: str) -> Optional[Dict[str, Any]]
```

### **LocalModelInterface**

```python
# Create local interface
interface = LocalModelInterface("gpt2", "causal")

# Generate text (no neuromodulation)
text = interface.generate_text(
    prompt="Hello",
    max_tokens=50,
    temperature=0.8
)

# Get status
status = interface.get_status()

# Check availability
available = interface.is_available()

# Unload and free memory
interface.unload()
```

### **VertexAIInterface**

```python
# Create Vertex AI interface
interface = VertexAIInterface(
    endpoint_url="https://endpoint.vertex.ai",
    project_id="project-id",
    location="us-central1"
)

# Generate with neuromodulation
text = interface.generate_text(
    prompt="Write a story",
    pack_name="lsd",
    max_tokens=100
)

# Get available packs
packs = interface.get_available_packs()

# Get available effects
effects = interface.get_available_effects()
```

## 🧪 **Neuromodulation Examples**

### **Predefined Packs**

```python
# Use caffeine pack
text = enhanced_model_manager.generate_text(
    prompt="Write a professional email",
    pack_name="caffeine",
    max_tokens=100
)

# Use LSD pack for creativity
text = enhanced_model_manager.generate_text(
    prompt="Describe a sunset",
    pack_name="lsd",
    max_tokens=150
)
```

### **Custom Packs**

```python
custom_pack = {
    "name": "research_mode",
    "description": "Optimized for analytical research",
    "effects": [
        {
            "effect": "temperature",
            "weight": 0.1,
            "direction": "down"
        },
        {
            "effect": "steering",
            "weight": 0.8,
            "direction": "up",
            "parameters": {
                "steering_type": "analytical"
            }
        }
    ]
}

text = enhanced_model_manager.generate_text(
    prompt="Analyze climate change impact",
    custom_pack=custom_pack,
    max_tokens=200
)
```

### **Individual Effects**

```python
individual_effects = [
    {
        "effect": "temperature",
        "weight": 0.3,
        "direction": "down"
    },
    {
        "effect": "steering",
        "weight": 0.6,
        "direction": "up",
        "parameters": {
            "steering_type": "creative"
        }
    }
]

text = enhanced_model_manager.generate_text(
    prompt="Write a poem",
    individual_effects=individual_effects,
    max_tokens=100
)
```

### **Multiple Packs**

```python
text = enhanced_model_manager.generate_text(
    prompt="Create a business strategy",
    multiple_packs=["caffeine", "mentor"],
    max_tokens=250
)
```

## 🔧 **Configuration**

### **Environment Variables**

```bash
# Google Cloud project
export GOOGLE_CLOUD_PROJECT="your-project-id"

# Vertex AI endpoint (optional)
export VERTEX_AI_ENDPOINT="https://endpoint.vertex.ai"

# Location (optional, defaults to us-central1)
export VERTEX_AI_LOCATION="us-west1"
```

### **Authentication**

The Vertex AI interface automatically handles authentication:

```python
# Uses gcloud auth application-default login
# Or GOOGLE_APPLICATION_CREDENTIALS environment variable
# Or service account key file

interface = VertexAIInterface(
    endpoint_url="https://endpoint.vertex.ai",
    project_id="project-id"
)
```

## 📊 **Status Information**

### **Local Model Status**

```python
status = enhanced_model_manager.get_status()
# Returns:
{
    "interface_type": "local",
    "interface_available": True,
    "type": "local",
    "model_name": "gpt2",
    "model_type": "causal",
    "model_loaded": True,
    "device": "cpu"
}
```

### **Vertex AI Status**

```python
status = enhanced_model_manager.get_status()
# Returns:
{
    "interface_type": "vertex_ai",
    "interface_available": True,
    "type": "vertex_ai",
    "endpoint_url": "https://endpoint.vertex.ai",
    "project_id": "project-id",
    "location": "us-central1",
    "status": "healthy",
    "model_loaded": True,
    "model_name": "meta-llama/Meta-Llama-3.1-8B"
}
```

## 🚨 **Error Handling**

### **Common Errors**

```python
try:
    text = enhanced_model_manager.generate_text(
        prompt="Hello",
        pack_name="caffeine"
    )
except RuntimeError as e:
    if "No model interface loaded" in str(e):
        print("No model loaded - load a local model or connect to Vertex AI")
    elif "Vertex AI error" in str(e):
        print("Vertex AI endpoint error - check endpoint status")
    else:
        print(f"Generation error: {e}")
```

### **Availability Checks**

```python
# Check if interface is available
if enhanced_model_manager.is_available():
    # Safe to generate text
    text = enhanced_model_manager.generate_text("Hello")
else:
    print("No model interface available")

# Check specific interface type
status = enhanced_model_manager.get_status()
if status.get("interface_type") == "vertex_ai":
    # Vertex AI specific operations
    packs = enhanced_model_manager.get_available_packs()
elif status.get("interface_type") == "local":
    # Local model operations
    pass
```

## 🔄 **Migration from Legacy**

### **Old Code (Still Works)**

```python
from model_manager import model_manager

# These still work exactly the same
model_manager.load_model("gpt2")
text = model_manager.generate_text("Hello")
status = model_manager.get_model_status()
model_manager.unload_model()
```

### **New Enhanced Code**

```python
from model_manager import enhanced_model_manager

# Enhanced functionality
enhanced_model_manager.load_local_model("gpt2")
enhanced_model_manager.connect_vertex_ai("https://endpoint.vertex.ai", "project-id")

# Neuromodulation support
text = enhanced_model_manager.generate_text(
    prompt="Hello",
    pack_name="caffeine"
)
```

## 📝 **Best Practices**

1. **Check Interface Type**: Always verify the current interface before using neuromodulation
2. **Error Handling**: Wrap generation calls in try-catch blocks
3. **Resource Management**: Unload interfaces when done to free memory
4. **Authentication**: Ensure proper Google Cloud authentication for Vertex AI
5. **Endpoint Health**: Check endpoint status before making requests

## 🧪 **Testing**

Run the example script to test functionality:

```bash
cd api
python example_usage.py
```

## 🔮 **Future Enhancements**

- **Multiple Vertex AI endpoints** - Load balancing and failover
- **Caching** - Cache responses for repeated prompts
- **Batch processing** - Generate multiple texts simultaneously
- **Metrics** - Track usage and performance
- **Auto-scaling** - Automatic endpoint scaling based on load

## 📚 **Related Documentation**

- [Vertex AI Container](../vertex_container/README.md)
- [API Examples](../vertex_container/API_EXAMPLES.md)
- [Neuromodulation System](../neuromod/README.md)
- [Pack System](../neuromod/pack_system.py)
