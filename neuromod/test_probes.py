#!/usr/bin/env python3
"""
Test script for the neuro-probe bus system
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
from neuromod.pack_system import PackRegistry
from neuromod.probes import ProbeEvent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_probe_system():
    """Test the probe system with a simple model"""
    
    print("🧪 Testing Neuro-Probe Bus System")
    print("=" * 50)
    
    # Load a simple model
    model_name = "distilgpt2"
    print(f"Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
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
    
    # Add probe listeners
    def on_novel_link(event: ProbeEvent):
        print(f"🎯 NOVEL_LINK fired at position {event.timestamp}: intensity={event.intensity:.3f}")
        print(f"   Raw signals: {event.raw_signals}")
        print(f"   Metadata: {event.metadata}")
    
    def on_avoid_guard(event: ProbeEvent):
        print(f"🛡️ AVOID_GUARD fired at position {event.timestamp}: intensity={event.intensity:.3f}")
        print(f"   Raw signals: {event.raw_signals}")
        print(f"   Metadata: {event.metadata}")
    
    # Register listeners
    novel_listener = neuromod_tool.add_probe_listener("NOVEL_LINK", on_novel_link)
    avoid_listener = neuromod_tool.add_probe_listener("AVOID_GUARD", on_avoid_guard)
    
    print("✅ Probe listeners registered")
    
    # Test generation with probe monitoring
    print("\n🚀 Testing generation with probe monitoring...")
    prompt = "The quantum mechanics of consciousness reveals"
    
    print(f"Prompt: '{prompt}'")
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    inputs = {k: v.cpu() for k, v in inputs.items()}
    
    # Generate with probe monitoring
    generated_tokens = []
    max_tokens = 20
    
    with torch.no_grad():
        for i in range(max_tokens):
            # Forward pass to get raw logits
            outputs = model(**inputs, output_attentions=True, output_hidden_states=True)
            raw_logits = outputs.logits
            
            # Get neuromodulation effects
            logits_processors = neuromod_tool.get_logits_processors()
            gen_kwargs = neuromod_tool.get_generation_kwargs()
            
            # Apply logits processors to get guarded logits
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
            neuromod_tool.update_token_position(i + 1)
            
            # Decode and store
            token_text = tokenizer.decode(next_token[0], skip_special_tokens=True)
            generated_tokens.append(token_text)
            
            # Print progress
            if i % 5 == 0:
                current_text = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
                print(f"   Position {i+1}: '{current_text}'")
    
    # Final result
    final_text = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
    print(f"\n📝 Final generated text: '{final_text}'")
    
    # Show probe statistics
    print("\n📊 Probe Statistics:")
    print("=" * 30)
    
    stats = neuromod_tool.get_probe_stats()
    for probe_name, probe_stats in stats.items():
        print(f"\n🔍 {probe_name}:")
        for listener_stats in probe_stats.get("listener_stats", []):
            print(f"   Total firings: {listener_stats['total_firings']}")
            print(f"   Average intensity: {listener_stats['average_intensity']:.3f}")
            print(f"   Recent events: {listener_stats['recent_events'][-5:]}")  # Last 5 events
    
    print("\n✅ Probe system test completed!")
    return True

if __name__ == "__main__":
    try:
        success = test_probe_system()
        if success:
            print("\n🎉 All tests passed!")
        else:
            print("\n❌ Tests failed!")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
