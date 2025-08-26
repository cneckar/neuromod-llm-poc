# Neuromodulation API Examples

This document shows how to use the enhanced Vertex AI prediction server with various neuromodulation options.

## 🎯 **API Endpoints**

- **`POST /predict`** - Main prediction endpoint with neuromodulation
- **`GET /available_packs`** - List all available packs
- **`GET /available_effects`** - List all available individual effects
- **`GET /health`** - Health check
- **`GET /model_info`** - Model information

## 🚀 **Basic Usage Examples**

### **1. Single Predefined Pack**
```bash
curl -X POST "https://your-endpoint/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "instances": [{
      "prompt": "Write a creative story about space exploration",
      "pack_name": "lsd",
      "max_tokens": 200,
      "temperature": 1.2
    }]
  }'
```

**Response:**
```json
{
  "predictions": [{
    "generated_text": "In the vast cosmic dance of stars...",
    "pack_applied": "lsd"
  }]
}
```

### **2. Custom Pack Definition**
```bash
curl -X POST "https://your-endpoint/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "instances": [{
      "prompt": "Explain quantum physics in simple terms",
      "custom_pack": {
        "name": "quantum_clarity",
        "description": "Custom pack for clear quantum explanations",
        "effects": [
          {
            "effect": "temperature",
            "weight": 0.3,
            "direction": "down"
          },
          {
            "effect": "qk_score_scaling",
            "weight": 0.7,
            "direction": "up"
          },
          {
            "effect": "steering",
            "weight": 0.5,
            "direction": "up",
            "parameters": {
              "steering_type": "associative"
            }
          }
        ]
      },
      "max_tokens": 150
    }]
  }'
```

**Response:**
```json
{
  "predictions": [{
    "generated_text": "Quantum physics is like...",
    "custom_pack_applied": "quantum_clarity"
  }]
}
```

### **3. Individual Effects**
```bash
curl -X POST "https://your-endpoint/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "instances": [{
      "prompt": "Write a professional email",
      "individual_effects": [
        {
          "effect": "temperature",
          "weight": 0.2,
          "direction": "down"
        },
        {
          "effect": "steering",
          "weight": 0.6,
          "direction": "up",
          "parameters": {
            "steering_type": "professional"
          }
        }
      ],
      "max_tokens": 100
    }]
  }'
```

**Response:**
```json
{
  "predictions": [{
    "generated_text": "Dear Team,\n\nI hope this email finds you well...",
    "individual_effects_applied": ["temperature", "steering"]
  }]
}
```

### **4. Multiple Packs (Combined Effects)**
```bash
curl -X POST "https://your-endpoint/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "instances": [{
      "prompt": "Create an innovative business strategy",
      "multiple_packs": ["caffeine", "mentor"],
      "max_tokens": 250
    }]
  }'
```

**Response:**
```json
{
  "predictions": [{
    "generated_text": "Here's a strategic approach that combines...",
    "multiple_packs_applied": ["caffeine", "mentor"]
  }]
}
```

## 📊 **Available Packs and Effects**

### **Get Available Packs**
```bash
curl "https://your-endpoint/available_packs"
```

**Response:**
```json
{
  "available_packs": [
    "caffeine", "cocaine", "lsd", "alcohol", "ketamine",
    "mdma", "mentor", "archivist", "firekeeper", "timepiece"
  ],
  "total_count": 82
}
```

### **Get Available Effects**
```bash
curl "https://your-endpoint/available_effects"
```

**Response:**
```json
{
  "available_effects": [
    "temperature", "top_p", "frequency_penalty", "presence_penalty",
    "qk_score_scaling", "head_masking_dropout", "steering",
    "activation_additions", "exponential_decay_kv", "truncation_kv"
  ],
  "total_count": 100
}
```

## 🧪 **Advanced Examples**

### **5. Research Mode with Custom Effects**
```bash
curl -X POST "https://your-endpoint/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "instances": [{
      "prompt": "Analyze the impact of climate change on biodiversity",
      "custom_pack": {
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
          },
          {
            "effect": "verifier_guided_decoding",
            "weight": 0.6,
            "direction": "up",
            "parameters": {
              "verifier_type": "coherence"
            }
          }
        ]
      },
      "max_tokens": 300
    }]
  }'
```

### **6. Creative Mode with Multiple Effects**
```bash
curl -X POST "https://your-endpoint/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "instances": [{
      "prompt": "Write a surrealist poem about technology",
      "individual_effects": [
        {
          "effect": "temperature",
          "weight": 0.9,
          "direction": "up"
        },
        {
          "effect": "steering",
          "weight": 0.7,
          "direction": "up",
          "parameters": {
            "steering_type": "visionary"
          }
        },
        {
          "effect": "noise_injection",
          "weight": 0.4,
          "direction": "up",
          "parameters": {
            "noise_level": 0.1
          }
        }
      ],
      "max_tokens": 200
    }]
  }'
```

### **7. Memory-Enhanced Generation**
```bash
curl -X POST "https://your-endpoint/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "instances": [{
      "prompt": "Continue this story with perfect memory of previous events: 'Once upon a time...'",
      "custom_pack": {
        "name": "memory_enhanced",
        "description": "Optimized for long-term memory",
        "effects": [
          {
            "effect": "exponential_decay_kv",
            "weight": 0.2,
            "direction": "down"
          },
          {
            "effect": "attention_anchors",
            "weight": 0.8,
            "direction": "up"
          }
        ]
      },
      "max_tokens": 150
    }]
  }'
```

## 🔧 **Effect Parameters Reference**

### **Common Effect Parameters**
- **`weight`**: 0.0 to 1.0 (effect strength)
- **`direction`**: "up" or "down" (effect direction)
- **`parameters`**: Effect-specific configuration

### **Steering Effect Parameters**
```json
{
  "effect": "steering",
  "weight": 0.6,
  "direction": "up",
  "parameters": {
    "steering_type": "associative|visionary|synesthesia|ego_thin|prosocial|affiliative|goal_focused|playful|creative|abstract"
  }
}
```

### **Temperature Effect Parameters**
```json
{
  "effect": "temperature",
  "weight": 0.5,
  "direction": "up",
  "parameters": {
    "base_temperature": 1.0,
    "max_multiplier": 2.0
  }
}
```

## 📝 **Request Format Summary**

```json
{
  "instances": [{
    "prompt": "Your text prompt here",
    "max_tokens": 100,
    "temperature": 1.0,
    "top_p": 1.0,
    
    // Choose ONE of these neuromodulation options:
    "pack_name": "caffeine",                    // Single predefined pack
    
    "custom_pack": {                            // Custom pack definition
      "name": "custom_name",
      "description": "Description",
      "effects": [...]
    },
    
    "individual_effects": [                     // Individual effects
      {
        "effect": "effect_name",
        "weight": 0.5,
        "direction": "up",
        "parameters": {...}
      }
    ],
    
    "multiple_packs": ["pack1", "pack2"]        // Multiple packs
  }]
}
```

## 🎯 **Best Practices**

1. **Use predefined packs** for common use cases
2. **Create custom packs** for specific research needs
3. **Combine individual effects** for fine-grained control
4. **Start with low weights** and increase gradually
5. **Monitor response quality** and adjust parameters
6. **Use appropriate steering types** for your task
7. **Consider effect interactions** when combining multiple effects

## 🚨 **Important Notes**

- **Only one neuromodulation method per request** (pack_name, custom_pack, individual_effects, or multiple_packs)
- **Effects are applied per-request** - each request starts with a clean state
- **Custom packs must follow the EffectConfig format**
- **Individual effects require valid effect names from the registry**
- **Multiple packs combine effects from different predefined packs**
- **All effects are cleared between requests for consistency**
