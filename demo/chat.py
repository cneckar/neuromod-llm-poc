#!/usr/bin/env python3
"""
Advanced Interactive Chat Interface with Neuromodulation Packs
Chat with a language model under the influence of 0, 1, or more neuromodulation packs
Supports loading from config, exporting to config, and custom effect combinations
"""

import torch
import os
import gc
import json
from typing import Dict, List, Any
from transformers import AutoTokenizer, AutoModelForCausalLM
from neuromod.pack_system import PackRegistry, Pack, EffectConfig
from neuromod.neuromod_tool import NeuromodTool
from neuromod.effects import EffectRegistry

# Disable MPS completely to avoid bus errors
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

class NeuromodChat:
    """Interactive chat interface with neuromodulation packs"""
    
    def __init__(self, model_name: str = "gpt2"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.neuromod_tool = None
        self.active_packs = []
        self.custom_effects = []  # Store custom effect combinations
        
        # Load model and setup
        self._load_model()
        self._setup_neuromodulation()
    
    def _load_model(self):
        """Load the language model with conservative settings"""
        print(f"Loading {self.model_name} model...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        self.model = self.model.cpu()
        self.model.eval()
        print(f"✅ {self.model_name} model loaded successfully")
    
    def _setup_neuromodulation(self):
        """Setup the neuromodulation system"""
        try:
            # Try to load from the new JSON config first
            config_paths = ["packs/config.json"]
            registry = None
            
            for config_path in config_paths:
                if os.path.exists(config_path):
                    registry = PackRegistry(config_path)
                    print(f"✅ Loaded neuromodulation system from {config_path}")
                    break
            
            if registry is None:
                print("⚠️ Warning: No config file found, creating empty registry")
                registry = PackRegistry()
            
            self.registry = registry
            self.neuromod_tool = NeuromodTool(registry, self.model, self.tokenizer)
            self.available_packs = registry.list_packs()
            self.effect_registry = EffectRegistry()
            
            print(f"✅ Neuromodulation system loaded")
            print(f"Available packs: {len(self.available_packs)}")
            print(f"Available effects: {len(self.effect_registry.list_effects())}")
            
        except Exception as e:
            print(f"⚠️ Warning: Could not setup neuromodulation system: {e}")
            print("Chat will run without neuromodulation capabilities")
            self.neuromod_tool = None
            self.available_packs = []
            self.registry = None
            self.effect_registry = None
    
    def list_packs(self):
        """List available packs"""
        if not self.available_packs:
            print("No neuromodulation packs available")
            return
        
        print("\n📦 Available Neuromodulation Packs:")
        print("=" * 40)
        for i, pack in enumerate(self.available_packs, 1):
            print(f"{i}. {pack}")
        print("0. No packs (baseline)")
    
    def select_packs(self):
        """Interactive pack selection"""
        if not self.available_packs:
            print("No neuromodulation packs available")
            return
        
        self.list_packs()
        
        while True:
            try:
                choice = input("\n🎯 Select packs (comma-separated numbers, or '0' for baseline): ").strip()
                
                if choice == "0":
                    self.clear_packs()
                    print("✅ Using baseline (no packs)")
                    return
                
                # Parse comma-separated numbers
                selected_indices = [int(x.strip()) for x in choice.split(",")]
                
                # Validate indices
                if any(idx < 1 or idx > len(self.available_packs) for idx in selected_indices):
                    print(f"❌ Invalid selection. Please choose numbers between 1 and {len(self.available_packs)}")
                    continue
                
                # Get selected packs
                selected_packs = [self.available_packs[idx-1] for idx in selected_indices]
                
                # Apply packs
                self.apply_packs(selected_packs)
                return
                
            except ValueError:
                print("❌ Invalid input. Please enter numbers separated by commas.")
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def apply_packs(self, pack_names: list):
        """Apply specified packs"""
        if not self.neuromod_tool:
            print("❌ Neuromodulation system not available")
            return
        
        # Clear existing packs
        self.neuromod_tool.clear()
        self.active_packs = []
        
        if not pack_names:
            print("✅ Using baseline (no packs)")
            return
        
        print(f"\n📦 Applying packs: {', '.join(pack_names)}")
        
        for pack_name in pack_names:
            try:
                result = self.neuromod_tool.apply(pack_name, intensity=0.8)
                if result.get("ok"):
                    self.active_packs.append(pack_name)
                    print(f"✅ Applied: {pack_name}")
                else:
                    print(f"❌ Failed to apply: {pack_name}")
            except Exception as e:
                print(f"❌ Error applying {pack_name}: {e}")
        
        if self.active_packs:
            print(f"✅ Active packs: {', '.join(self.active_packs)}")
        else:
            print("✅ Using baseline (no packs)")
    
    def clear_packs(self):
        """Clear all active packs"""
        if self.neuromod_tool:
            self.neuromod_tool.clear()
        self.active_packs = []
    
    def generate_response(self, prompt: str, max_tokens: int = 100) -> str:
        """Generate response with current neuromodulation state"""
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
            inputs = {k: v.cpu() for k, v in inputs.items()}
            
            # Get neuromodulation effects if available
            logits_processors = []
            gen_kwargs = {}
            
            if self.neuromod_tool:
                # Update token position for phase-based effects
                self.neuromod_tool.update_token_position(0)
                logits_processors = self.neuromod_tool.get_logits_processors()
                gen_kwargs = self.neuromod_tool.get_generation_kwargs()
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=False,
                    repetition_penalty=1.1,
                    no_repeat_ngram_size=2,
                    early_stopping=False,
                    logits_processor=logits_processors,
                    **gen_kwargs
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response[len(prompt):].strip()
            
            return response if response else "I don't have a response for that."
            
        except Exception as e:
            print(f"❌ Generation error: {e}")
            return "Sorry, I encountered an error while generating a response."
    
    def chat(self):
        """Main chat loop"""
        print("\n" + "=" * 60)
        print("🧠 Neuromodulation Chat Interface")
        print("=" * 60)
        print(f"Model: {self.model_name}")
        
        # Initial pack selection
        self.select_packs()
        
        print("\n💬 Chat Commands:")
        print("  /packs - Show available packs")
        print("  /select - Change packs")
        print("  /clear - Clear all packs (baseline)")
        print("  /status - Show current status")
        print("  /effects - Show available effects")
        print("  /add_effect - Add individual effect to current combination")
        print("  /remove_effect - Remove effect from current combination")
        print("  /show_combination - Show current custom combination")
        print("  /export_pack - Export current combination as a pack")
        print("  /load_pack - Load a pack from config file")
        print("  /save_config - Save current config to file")
        print("  /quit - Exit chat")
        print("  /help - Show this help")
        print("-" * 60)
        
        while True:
            try:
                # Get user input
                user_input = input("\n🤖 You: ").strip()
                
                # Handle commands
                if user_input.startswith("/"):
                    self._handle_command(user_input)
                    continue
                
                if not user_input:
                    continue
                
                # Generate response
                print("🤖 Patient: ", end="", flush=True)
                response = self.generate_response(user_input)
                print(response)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except EOFError:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
    
    def _handle_command(self, command: str):
        """Handle chat commands"""
        cmd = command.lower().strip()
        
        if cmd == "/packs":
            self.list_packs()
        
        elif cmd == "/select":
            self.select_packs()
        
        elif cmd == "/clear":
            self.clear_packs()
            print("✅ Cleared all packs (using baseline)")
        
        elif cmd == "/status":
            print(f"\n📊 Current Status:")
            print(f"  Model: {self.model_name}")
            print(f"  Active packs: {', '.join(self.active_packs) if self.active_packs else 'None (baseline)'}")
            print(f"  Custom effects: {len(self.custom_effects)}")
            if self.custom_effects:
                for i, effect in enumerate(self.custom_effects, 1):
                    print(f"    {i}. {effect.effect} (weight={effect.weight}, direction={effect.direction})")
            print(f"  Neuromodulation: {'Available' if self.neuromod_tool else 'Not available'}")
        
        elif cmd == "/quit":
            print("\n👋 Goodbye!")
            exit(0)
        
        elif cmd == "/effects":
            self.show_effects()
        
        elif cmd == "/add_effect":
            self.add_effect()
        
        elif cmd == "/remove_effect":
            self.remove_effect()
        
        elif cmd == "/show_combination":
            self.show_combination()
        
        elif cmd == "/export_pack":
            self.export_pack()
        
        elif cmd == "/load_pack":
            self.load_pack()
        
        elif cmd == "/save_config":
            self.save_config()
        
        elif cmd == "/help":
            print("\n💬 Chat Commands:")
            print("  /packs - Show available packs")
            print("  /select - Change packs")
            print("  /clear - Clear all packs (baseline)")
            print("  /status - Show current status")
            print("  /effects - Show available effects")
            print("  /add_effect - Add individual effect to current combination")
            print("  /remove_effect - Remove effect from current combination")
            print("  /show_combination - Show current custom combination")
            print("  /export_pack - Export current combination as a pack")
            print("  /load_pack - Load a pack from config file")
            print("  /save_config - Save current config to file")
            print("  /quit - Exit chat")
            print("  /help - Show this help")
        
        else:
            print(f"❌ Unknown command: {command}")
            print("Type /help for available commands")
    
    def show_effects(self):
        """Show available effects"""
        if not self.effect_registry:
            print("❌ Effect registry not available")
            return
        
        effects = self.effect_registry.list_effects()
        print(f"\n🔧 Available Effects ({len(effects)}):")
        print("=" * 50)
        
        # Group effects by category
        categories = {
            "Sampler": [e for e in effects if any(x in e for x in ["temperature", "top_p", "frequency", "presence", "pulsed", "contrastive", "expert", "token_class"])],
            "Attention": [e for e in effects if any(x in e for x in ["attention", "qk", "head", "positional"])],
            "Steering": [e for e in effects if "steering" in e],
            "Memory": [e for e in effects if any(x in e for x in ["kv", "memory", "segment", "stride", "truncation"])],
            "Activation": [e for e in effects if any(x in e for x in ["activation", "soft_projection", "layer_wise", "noise"])],
            "MoE": [e for e in effects if any(x in e for x in ["router", "expert_persistence", "expert_masking"])],
            "Objective": [e for e in effects if any(x in e for x in ["verifier", "style_affect", "risk_preference", "compute_at_test", "retrieval_rate", "persona_voice"])],
            "Input": [e for e in effects if any(x in e for x in ["lexical_jitter", "structured_prefaces"])]
        }
        
        for category, effect_list in categories.items():
            if effect_list:
                print(f"\n📂 {category}:")
                for effect in sorted(effect_list):
                    print(f"   • {effect}")
    
    def add_effect(self):
        """Add individual effect to current combination"""
        if not self.effect_registry:
            print("❌ Effect registry not available")
            return
        
        effects = self.effect_registry.list_effects()
        print(f"\n🔧 Available Effects:")
        for i, effect in enumerate(sorted(effects), 1):
            print(f"{i:2d}. {effect}")
        
        try:
            effect_idx = int(input("\nSelect effect number: ")) - 1
            if effect_idx < 0 or effect_idx >= len(effects):
                print("❌ Invalid effect number")
                return
            
            effect_name = sorted(effects)[effect_idx]
            
            weight = float(input(f"Enter weight (0.0-1.0) for {effect_name}: "))
            if weight < 0.0 or weight > 1.0:
                print("❌ Weight must be between 0.0 and 1.0")
                return
            
            direction = input("Enter direction (up/down): ").lower()
            if direction not in ["up", "down"]:
                print("❌ Direction must be 'up' or 'down'")
                return
            
            # Create effect config
            effect_config = EffectConfig(
                effect=effect_name,
                weight=weight,
                direction=direction
            )
            
            self.custom_effects.append(effect_config)
            print(f"✅ Added {effect_name} (weight={weight}, direction={direction})")
            
            # Apply the new combination
            self._apply_custom_combination()
            
        except ValueError:
            print("❌ Invalid input")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def remove_effect(self):
        """Remove effect from current combination"""
        if not self.custom_effects:
            print("❌ No custom effects to remove")
            return
        
        print(f"\n🔧 Current Custom Effects:")
        for i, effect in enumerate(self.custom_effects, 1):
            print(f"{i}. {effect.effect} (weight={effect.weight}, direction={effect.direction})")
        
        try:
            effect_idx = int(input("\nSelect effect number to remove: ")) - 1
            if effect_idx < 0 or effect_idx >= len(self.custom_effects):
                print("❌ Invalid effect number")
                return
            
            removed_effect = self.custom_effects.pop(effect_idx)
            print(f"✅ Removed {removed_effect.effect}")
            
            # Apply the updated combination
            self._apply_custom_combination()
            
        except ValueError:
            print("❌ Invalid input")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def show_combination(self):
        """Show current custom combination"""
        if not self.custom_effects:
            print("❌ No custom effects defined")
            return
        
        print(f"\n🔧 Current Custom Combination ({len(self.custom_effects)} effects):")
        print("=" * 50)
        for i, effect in enumerate(self.custom_effects, 1):
            print(f"{i}. {effect.effect} (weight={effect.weight}, direction={effect.direction})")
    
    def export_pack(self):
        """Export current combination as a pack"""
        if not self.custom_effects:
            print("❌ No custom effects to export")
            return
        
        pack_name = input("Enter pack name: ").strip()
        if not pack_name:
            print("❌ Pack name cannot be empty")
            return
        
        description = input("Enter pack description: ").strip()
        if not description:
            description = f"Custom pack with {len(self.custom_effects)} effects"
        
        # Create pack
        pack = Pack(
            name=pack_name,
            description=description,
            effects=self.custom_effects.copy()
        )
        
        # Add to registry
        if self.registry:
            self.registry.add_pack(pack)
            print(f"✅ Added pack '{pack_name}' to registry")
            
            # Save to config file
            try:
                self.registry.save_packs_to_json()
                print(f"✅ Saved pack to config file")
            except Exception as e:
                print(f"⚠️ Warning: Could not save to config file: {e}")
        else:
            print("❌ Registry not available")
    
    def load_pack(self):
        """Load a pack from config file"""
        if not self.registry:
            print("❌ Registry not available")
            return
        
        # Reload packs from config
        self.registry.reload_packs()
        self.available_packs = self.registry.list_packs()
        
        print(f"✅ Reloaded {len(self.available_packs)} packs from config")
        self.list_packs()
    
    def save_config(self):
        """Save current config to file"""
        if not self.registry:
            print("❌ Registry not available")
            return
        
        filename = input("Enter filename to save config (default: packs/config.json): ").strip()
        if not filename:
            filename = "packs/config.json"
        
        try:
            self.registry.save_packs_to_json(filename)
            print(f"✅ Saved config to {filename}")
        except Exception as e:
            print(f"❌ Error saving config: {e}")
    
    def _apply_custom_combination(self):
        """Apply current custom effect combination"""
        if not self.neuromod_tool or not self.custom_effects:
            return
        
        # Clear existing effects
        self.neuromod_tool.clear()
        self.active_packs = []
        
        # Create a temporary pack for the custom combination
        temp_pack = Pack(
            name="custom_combination",
            description="Temporary custom combination",
            effects=self.custom_effects
        )
        
        # Add the pack to registry and apply
        try:
            # Add to registry temporarily
            self.registry.add_pack(temp_pack)
            
            # Apply using the pack name
            result = self.neuromod_tool.apply("custom_combination", intensity=1.0)
            if result.get("ok"):
                print(f"✅ Applied custom combination with {len(self.custom_effects)} effects")
            else:
                print(f"❌ Error applying custom combination: {result.get('error')}")
            
            # Remove from registry to avoid clutter
            self.registry.remove_pack("custom_combination")
            
        except Exception as e:
            print(f"❌ Error applying custom combination: {e}")

def main():
    """Main function"""
    print("🚀 Neuromodulation Chat Interface")
    print("=" * 50)
    
    # Model selection
    models = [
        "gpt2",
        "microsoft/DialoGPT-small",
        "microsoft/DialoGPT-medium"
    ]
    
    print("\n🤖 Available Models:")
    for i, model in enumerate(models, 1):
        print(f"{i}. {model}")
    
    while True:
        try:
            choice = input(f"\nSelect model (1-{len(models)}, or enter custom model name): ").strip()
            
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(models):
                    model_name = models[idx]
                    break
                else:
                    print(f"❌ Please choose a number between 1 and {len(models)}")
            else:
                # Custom model name
                model_name = choice
                break
                
        except ValueError:
            print("❌ Invalid input")
    
    # Create and start chat
    try:
        chat = NeuromodChat(model_name)
        chat.chat()
    except Exception as e:
        print(f"❌ Failed to start chat: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
