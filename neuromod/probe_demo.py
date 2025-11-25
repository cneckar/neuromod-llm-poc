#!/usr/bin/env python3
"""
Demo script for the neuro-probe bus system
Shows how to integrate probes with neuromodulation effects
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

# Import neuromodulation components
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neuromod import NeuromodTool
from neuromod.pack_system import PackRegistry, EffectConfig, Pack
from neuromod.probes import ProbeEvent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def demo_probes_with_neuromodulation():
    """Demo probes working with neuromodulation effects"""
    
    print("🧠 Neuro-Probe Bus Demo with Neuromodulation")
    print("=" * 60)
    
    # Load model
    model_name = "distilgpt2"
    print(f"Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    model = model.cpu()
    model.eval()
    
    print("✅ Model loaded successfully")
    
    # Setup neuromodulation with probes
    print("\n🔧 Setting up neuromodulation with probes...")
    registry = PackRegistry("packs/config.json")
    neuromod_tool = NeuromodTool(registry, model, tokenizer)
    
    # Create a custom neuromodulation pack
    print("\n🎛️ Creating custom neuromodulation pack...")
    custom_effects = [
        EffectConfig("temperature", weight=0.7, direction="up"),
        EffectConfig("attention_focus", weight=0.5, direction="up"),
        EffectConfig("noise_injection", weight=0.3, direction="up")
    ]
    
    custom_pack = Pack(
        name="probe_test_pack",
        description="Custom pack for testing probes with neuromodulation",
        effects=custom_effects
    )
    
    # Apply the custom pack
    result = neuromod_tool.apply('probe_test_pack', intensity=0.8)
    if result.get("ok"):
        print("✅ Custom neuromodulation pack applied")
    else:
        print(f"❌ Failed to apply pack: {result.get('error')}")
    
    # Setup probe listeners
    print("\n🔍 Setting up probe listeners...")
    
    def on_novel_link(event: ProbeEvent):
        print(f"🎯 NOVEL_LINK fired at position {event.timestamp}")
        print(f"   Intensity: {event.intensity:.3f}")
        print(f"   Surprisal: {event.raw_signals.get('surprisal', 0):.3f}")
        print(f"   Bridge score: {event.raw_signals.get('bridge_score', 0):.3f}")
        print(f"   Decision: {event.metadata.get('decision_rule', 'unknown')}")
        print()
    
    def on_avoid_guard(event: ProbeEvent):
        print(f"🛡️ AVOID_GUARD fired at position {event.timestamp}")
        print(f"   Intensity: {event.intensity:.3f}")
        print(f"   KL divergence: {event.raw_signals.get('kl_divergence', 0):.3f}")
        print(f"   Suppressed mass: {event.raw_signals.get('suppressed_mass', 0):.3f}")
        print(f"   Mask hits: {event.raw_signals.get('mask_hits', 0)}")
        print()
    
    # Register listeners
    neuromod_tool.add_probe_listener("NOVEL_LINK", on_novel_link)
    neuromod_tool.add_probe_listener("AVOID_GUARD", on_avoid_guard)
    
    print("✅ Probe listeners registered")
    
    # Test generation with both neuromodulation and probes
    print("\n🚀 Testing generation with neuromodulation + probes...")
    
    prompts = [
        "The quantum mechanics of consciousness reveals",
        "In the depths of the human mind, we find",
        "The intersection of neuroscience and AI shows"
    ]
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n--- Test {i}: '{prompt}' ---")
        
        # Reset probes for new generation
        neuromod_tool.reset_probes()
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.cpu() for k, v in inputs.items()}
        
        # Generate with monitoring
        max_tokens = 15
        
        with torch.no_grad():
            for j in range(max_tokens):
                # Forward pass with full outputs
                outputs = model(**inputs, output_attentions=True, output_hidden_states=True)
                raw_logits = outputs.logits
                
                # Get neuromodulation effects
                logits_processors = neuromod_tool.get_logits_processors()
                gen_kwargs = neuromod_tool.get_generation_kwargs()
                
                # Apply logits processors
                guarded_logits = raw_logits.clone()
                for processor in logits_processors:
                    guarded_logits = processor(inputs["input_ids"], guarded_logits)
                
                # Sample next token
                probs = torch.softmax(guarded_logits[:, -1, :], dim=-1)
                next_token = torch.multinomial(probs, 1)
                
                # Process probe signals
                neuromod_tool.process_probe_signals(
                    raw_logits=raw_logits[:, -1, :],
                    guarded_logits=guarded_logits[:, -1, :],
                    sampled_token_id=next_token[0, 0].item(),
                    attention_weights=outputs.attentions,
                    hidden_states=outputs.hidden_states,
                    temperature=0.7
                )
                
                # Add token to sequence
                inputs["input_ids"] = torch.cat([inputs["input_ids"], next_token], dim=1)
                inputs["attention_mask"] = torch.cat([inputs["attention_mask"], torch.ones(1, 1)], dim=1)
                
                # Update token position
                neuromod_tool.update_token_position(j + 1)
        
        # Show final result
        final_text = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
        print(f"Generated: '{final_text}'")
    
    # Show final statistics
    print("\n📊 Final Probe Statistics:")
    print("=" * 40)
    
    stats = neuromod_tool.get_probe_stats()
    for probe_name, probe_stats in stats.items():
        print(f"\n🔍 {probe_name}:")
        for listener_stats in probe_stats.get("listener_stats", []):
            print(f"   Total firings: {listener_stats['total_firings']}")
            print(f"   Average intensity: {listener_stats['average_intensity']:.3f}")
            print(f"   Firing rate: {listener_stats['firing_rate']:.3f}")
    
    print("\n✅ Demo completed successfully!")
    return True

if __name__ == "__main__":
    try:
        success = demo_probes_with_neuromodulation()
        if success:
            print("\n🎉 Demo completed successfully!")
        else:
            print("\n❌ Demo failed!")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
