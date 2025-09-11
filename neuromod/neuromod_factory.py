"""
Neuromodulation Tool Factory

This module provides factory functions for creating NeuromodTool instances
with proper model loading and configuration for both test and production environments.
"""

import logging
from typing import Optional, Dict, Any, Tuple
from .neuromod_tool import NeuromodTool
from .pack_system import PackRegistry
from .model_support import ModelSupportManager, create_model_support
from .effects import EffectRegistry

logger = logging.getLogger(__name__)

def create_neuromod_tool(
    model_name: Optional[str] = None,
    test_mode: bool = True,
    model_kwargs: Optional[Dict[str, Any]] = None
) -> Tuple[NeuromodTool, Dict[str, Any]]:
    """
    Create a NeuromodTool instance with proper model loading
    
    Args:
        model_name: Name of model to load (if None, uses recommended model)
        test_mode: Whether to run in test mode (smaller models)
        model_kwargs: Additional arguments for model loading
        
    Returns:
        Tuple of (NeuromodTool instance, model_info)
    """
    # Create model support manager
    model_manager = create_model_support(test_mode=test_mode)
    
    # Get model name
    if model_name is None:
        model_name = model_manager.get_recommended_model()
    
    logger.info(f"Creating NeuromodTool with model: {model_name}")
    
    # Load model
    model, tokenizer, model_info = model_manager.load_model(
        model_name, 
        **(model_kwargs or {})
    )
    
    # Create effect registry
    effect_registry = EffectRegistry()
    
    # Create pack registry
    pack_registry = PackRegistry()
    
    # Create neuromod tool
    neuromod_tool = NeuromodTool(
        registry=pack_registry,
        model=model,
        tokenizer=tokenizer,
        vectors=None  # Will be loaded as needed
    )
    
    # Store model manager reference for cleanup
    neuromod_tool.model_manager = model_manager
    neuromod_tool.model_name = model_name
    
    logger.info(f"NeuromodTool created successfully with {model_name}")
    
    return neuromod_tool, model_info

def create_test_neuromod_tool() -> Tuple[NeuromodTool, Dict[str, Any]]:
    """Create a NeuromodTool for testing with mock model"""
    return create_neuromod_tool(
        model_name="mock",
        test_mode=True
    )

def create_production_neuromod_tool(model_name: str) -> Tuple[NeuromodTool, Dict[str, Any]]:
    """Create a NeuromodTool for production with specified model"""
    return create_neuromod_tool(
        model_name=model_name,
        test_mode=False
    )

def cleanup_neuromod_tool(neuromod_tool: NeuromodTool):
    """Clean up a NeuromodTool and its resources"""
    if hasattr(neuromod_tool, 'model_manager'):
        neuromod_tool.model_manager.cleanup()
        logger.info(f"Cleaned up NeuromodTool with model: {getattr(neuromod_tool, 'model_name', 'unknown')}")

def main():
    """Example usage of the neuromod factory"""
    print("🧠 NeuromodTool Factory Demo")
    print("=" * 50)
    
    try:
        # Create test tool
        print("Creating test NeuromodTool...")
        tool, model_info = create_test_neuromod_tool()
        
        print(f"Model info: {model_info}")
        print(f"Available packs: {list(tool.pack_manager.list_packs().keys())}")
        
        # Test basic functionality
        print("\nTesting basic functionality...")
        
        # Test pack application
        try:
            tool.apply("caffeine", intensity=0.5)
            print("✅ Pack application works")
        except Exception as e:
            print(f"⚠️ Pack application failed: {e}")
        
        # Test generation
        try:
            prompt = "Hello, how are you today?"
            inputs = tool.tokenizer(prompt, return_tensors="pt")
            
            with tool.model.generate if hasattr(tool.model, 'generate') else None:
                if hasattr(tool.model, 'generate'):
                    outputs = tool.model.generate(
                        inputs.input_ids,
                        max_new_tokens=20,
                        temperature=0.7,
                        do_sample=True
                    )
                    response = tool.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    print(f"✅ Generation works: {response}")
                else:
                    print("⚠️ Model doesn't support generation (mock model)")
        except Exception as e:
            print(f"⚠️ Generation failed: {e}")
        
        # Cleanup
        cleanup_neuromod_tool(tool)
        print("\n✅ Demo completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
