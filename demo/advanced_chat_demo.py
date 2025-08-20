#!/usr/bin/env python3
"""
Advanced Chat Interface Demonstration
Shows all the new features: loading from config, custom effects, exporting packs, etc.
"""

import torch
import os
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM
from neuromod.pack_system import PackRegistry, Pack, EffectConfig
from neuromod.neuromod_tool import NeuromodTool
from neuromod.effects import EffectRegistry

# Disable MPS completely to avoid bus errors
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

def demo_advanced_features():
    """Demonstrate all advanced features"""
    print("🎯 ADVANCED CHAT INTERFACE DEMONSTRATION")
    print("=" * 50)
    
    # Load model
    print("\n1️⃣ Loading model...")
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained('gpt2', torch_dtype=torch.float32, device_map='cpu')
    model.eval()
    print("✅ Model loaded")
    
    # Setup neuromodulation
    print("\n2️⃣ Setting up neuromodulation system...")
    registry = PackRegistry('packs/config.json')
    neuromod_tool = NeuromodTool(registry, model, tokenizer)
    effect_registry = EffectRegistry()
    
    print(f"✅ Loaded {len(registry.list_packs())} packs from config")
    print(f"✅ Available {len(effect_registry.list_effects())} effects")
    
    # Demo 1: Load packs from config
    print("\n3️⃣ Demo: Loading packs from config")
    packs = registry.list_packs()
    print(f"Available packs: {', '.join(packs[:5])}...")
    
    # Demo 2: Show available effects
    print("\n4️⃣ Demo: Available effects by category")
    effects = effect_registry.list_effects()
    categories = {
        "Sampler": [e for e in effects if any(x in e for x in ["temperature", "top_p", "frequency", "presence", "pulsed", "contrastive", "expert", "token_class"])],
        "Attention": [e for e in effects if any(x in e for x in ["attention", "qk", "head", "positional"])],
        "Memory": [e for e in effects if any(x in e for x in ["kv", "memory", "segment", "stride", "truncation"])],
        "Activation": [e for e in effects if any(x in e for x in ["activation", "soft_projection", "layer_wise", "noise"])],
    }
    
    for category, effect_list in categories.items():
        if effect_list:
            print(f"   📂 {category}: {len(effect_list)} effects")
    
    # Demo 3: Create custom combination
    print("\n5️⃣ Demo: Creating custom effect combination")
    custom_effects = [
        EffectConfig("temperature", weight=0.4, direction="up"),
        EffectConfig("attention_oscillation", weight=0.6, direction="up"),
        EffectConfig("noise_injection", weight=0.3, direction="up"),
        EffectConfig("kv_decay", weight=0.5, direction="up")
    ]
    
    print("Creating custom pack with effects:")
    for i, effect in enumerate(custom_effects, 1):
        print(f"   {i}. {effect.effect} (weight={effect.weight}, direction={effect.direction})")
    
    # Demo 4: Apply custom combination
    print("\n6️⃣ Demo: Applying custom combination")
    custom_pack = Pack(
        name="demo_custom_pack",
        description="Demo custom combination for testing",
        effects=custom_effects
    )
    
    try:
        # Add the custom pack to the registry first
        registry.add_pack(custom_pack)
        
        # Apply using the pack name
        result = neuromod_tool.apply('demo_custom_pack', intensity=0.7)
        if result.get("ok"):
            print("✅ Custom combination applied successfully")
        else:
            print(f"❌ Failed to apply custom combination: {result.get('error')}")
            return
        
        # Test generation with custom effects
        print("\n7️⃣ Demo: Testing generation with custom effects")
        prompt = "The world around me feels"
        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.cpu() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=30,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=False,
                repetition_penalty=1.1,
                no_repeat_ngram_size=2,
                early_stopping=False
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(prompt):].strip()
        print(f"Prompt: {prompt}")
        print(f"Response: {response}")
        
    except Exception as e:
        print(f"❌ Error applying custom combination: {e}")
    
    # Demo 5: Export custom pack
    print("\n8️⃣ Demo: Exporting custom pack to registry")
    try:
        registry.add_pack(custom_pack)
        print("✅ Added custom pack to registry")
        
        # Save to config
        registry.save_packs_to_json('packs/demo_export.json')
        print("✅ Saved custom pack to config file")
        
    except Exception as e:
        print(f"❌ Error exporting pack: {e}")
    
    # Demo 6: Show pack information
    print("\n9️⃣ Demo: Detailed pack information")
    try:
        pack_info = registry.get_pack_info('demo_custom_pack')
        print(f"Pack: {pack_info['name']}")
        print(f"Description: {pack_info['description']}")
        print(f"Effects: {len(pack_info['effects'])}")
        for effect in pack_info['effects']:
            print(f"   • {effect['effect']} (weight={effect['weight']}, direction={effect['direction']})")
    except Exception as e:
        print(f"❌ Error getting pack info: {e}")
    
    # Demo 7: Reload from config
    print("\n🔟 Demo: Reloading from config")
    try:
        registry.reload_packs()
        print(f"✅ Reloaded {len(registry.list_packs())} packs from config")
    except Exception as e:
        print(f"❌ Error reloading: {e}")
    
    print("\n🎉 Advanced features demonstration completed!")
    print("\n📋 Summary of features demonstrated:")
    print("   ✅ Loading packs from JSON config")
    print("   ✅ Viewing available effects by category")
    print("   ✅ Creating custom effect combinations")
    print("   ✅ Applying custom combinations to model")
    print("   ✅ Testing generation with custom effects")
    print("   ✅ Exporting custom packs to registry")
    print("   ✅ Saving configurations to JSON files")
    print("   ✅ Getting detailed pack information")
    print("   ✅ Hot-reloading from config files")

if __name__ == "__main__":
    demo_advanced_features()
