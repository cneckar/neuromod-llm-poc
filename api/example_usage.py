#!/usr/bin/env python3
"""
Example usage of the Enhanced Model Manager
Demonstrates switching between local models and Vertex AI endpoints
"""

from model_manager import enhanced_model_manager
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def example_local_model():
    """Example using a local model with neuromodulation support"""
    print("\n=== Local Model Example ===")
    
    # Load a local model
    success = enhanced_model_manager.load_local_model("gpt2")
    if not success:
        print("❌ Failed to load local model")
        return
    
    # Check status
    status = enhanced_model_manager.get_status()
    print(f"✅ Model loaded: {status}")
    
    # Check available packs (now supported for local models!)
    packs = enhanced_model_manager.get_available_packs()
    print(f"Available packs: {packs[:5] if packs else 'None'}...")
    
    # Generate text with neuromodulation (now supported!)
    try:
        # Using a predefined pack
        text = enhanced_model_manager.generate_text(
            prompt="Write a creative story about coffee",
            max_tokens=50,
            temperature=0.8,
            pack_name="caffeine"  # This now works for local models!
        )
        print(f"Generated text with caffeine pack: {text}")
        
        # Using individual effects
        text = enhanced_model_manager.generate_text(
            prompt="Explain quantum physics simply",
            max_tokens=80,
            temperature=0.3,
            individual_effects=[
                {
                    "effect": "temperature",
                    "weight": 0.2,
                    "direction": "down"
                },
                {
                    "effect": "steering",
                    "weight": 0.7,
                    "direction": "up",
                    "parameters": {
                        "steering_type": "analytical"
                    }
                }
            ]
        )
        print(f"Generated text with custom effects: {text}")
        
    except Exception as e:
        print(f"❌ Generation failed: {e}")

def example_vertex_ai():
    """Example using a Vertex AI endpoint"""
    print("\n=== Vertex AI Example ===")
    
    # Connect to Vertex AI endpoint
    endpoint_url = "https://your-endpoint-id.us-central1-1234567890.vertex.ai"
    project_id = "neuromod-469620"
    
    success = enhanced_model_manager.connect_vertex_ai(
        endpoint_url=endpoint_url,
        project_id=project_id,
        location="us-central1"
    )
    
    if not success:
        print("❌ Failed to connect to Vertex AI endpoint")
        return
    
    # Check status
    status = enhanced_model_manager.get_status()
    print(f"✅ Connected to Vertex AI: {status}")
    
    # Get available packs from Vertex AI
    packs = enhanced_model_manager.get_available_packs()
    print(f"Available packs: {packs[:5]}...")  # Show first 5
    
    # Get available effects
    effects = enhanced_model_manager.get_available_effects()
    print(f"Available effects: {effects[:5]}...")  # Show first 5
    
    # Generate text with neuromodulation
    try:
        # Using a predefined pack
        text = enhanced_model_manager.generate_text(
            prompt="Write a creative story about coffee",
            max_tokens=100,
            temperature=1.2,
            pack_name="caffeine"
        )
        print(f"Generated text with caffeine pack: {text}")
        
        # Using individual effects
        text = enhanced_model_manager.generate_text(
            prompt="Explain quantum physics simply",
            max_tokens=80,
            temperature=0.3,
            individual_effects=[
                {
                    "effect": "temperature",
                    "weight": 0.2,
                    "direction": "down"
                },
                {
                    "effect": "steering",
                    "weight": 0.7,
                    "direction": "up",
                    "parameters": {
                        "steering_type": "associative"
                    }
                }
            ]
        )
        print(f"Generated text with custom effects: {text}")
        
    except Exception as e:
        print(f"❌ Generation failed: {e}")

def example_switching():
    """Example of switching between interfaces"""
    print("\n=== Interface Switching Example ===")
    
    # Start with local model
    print("Loading local model...")
    enhanced_model_manager.load_local_model("distilgpt2")
    
    status = enhanced_model_manager.get_status()
    print(f"Current interface: {status['interface_type']}")
    
    # Switch to Vertex AI
    print("Switching to Vertex AI...")
    enhanced_model_manager.connect_vertex_ai(
        endpoint_url="https://your-endpoint-id.us-central1-1234567890.vertex.ai",
        project_id="neuromod-469620"
    )
    
    status = enhanced_model_manager.get_status()
    print(f"Current interface: {status['interface_type']}")
    
    # Switch back to local
    print("Switching back to local...")
    enhanced_model_manager.load_local_model("gpt2")
    
    status = enhanced_model_manager.get_status()
    print(f"Current interface: {status['interface_type']}")

def main():
    """Main example function"""
    print("🧠 Enhanced Model Manager Examples")
    print("=" * 50)
    
    # Example 1: Local model
    example_local_model()
    
    # Example 2: Vertex AI (commented out - requires real endpoint)
    # example_vertex_ai()
    
    # Example 3: Interface switching
    example_switching()
    
    # Clean up
    enhanced_model_manager.unload_current_interface()
    print("\n✅ Examples completed!")

if __name__ == "__main__":
    main()
