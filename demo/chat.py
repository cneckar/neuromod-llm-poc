#!/usr/bin/env python3
"""
Advanced Interactive Chat Interface with Neuromodulation Packs
Chat with a language model under the influence of 0, 1, or more neuromodulation packs
Supports loading from config, exporting to config, and custom effect combinations
Now with real-time emotion tracking!
"""

import os
import gc
import json
import sys
import time
import warnings
from typing import Dict, List, Any, Optional

# Suppress warnings from optional dependencies
warnings.filterwarnings('ignore', category=UserWarning, module='neuromod.testing.advanced_statistics')

# Load environment variables from a .env if python-dotenv is installed (optional — remote mode
# only needs `requests`, so don't hard-require dotenv on a lightweight client).
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Add the parent directory to the path to import neuromod modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# NOTE: torch and the heavy neuromod/model stack are imported lazily (inside the LOCAL-mode
# code paths only). This keeps --remote mode fully torch-free so the CLI can drive a deployed
# RunPod endpoint from a laptop with nothing but `requests` installed.

# Disable MPS completely to avoid bus errors (local mode)
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'


def _list_packs_from_config() -> List[str]:
    """Read pack names straight from packs/config.json (torch-free).

    Used in --remote mode, where we don't load the neuromod registry locally — the server
    applies packs; the client only needs their names to offer a menu.
    """
    cfg = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "packs", "config.json")
    try:
        with open(cfg, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        packs = data.get("packs", data)
        return sorted(packs.keys()) if isinstance(packs, dict) else []
    except Exception:
        return []

class NeuromodChat:
    """Interactive chat interface with neuromodulation packs"""
    
    def __init__(self, model_name: str = None, test_mode: bool = True, remote=None):
        self.model_name = model_name
        self.test_mode = test_mode
        self.remote = remote  # RunPodModelInterface or None; set => talk to the endpoint over HTTP
        self.model = None
        self.tokenizer = None
        self.neuromod_tool = None
        self.registry = None
        self.effect_registry = None
        self.active_packs = []
        self.custom_effects = []  # Store custom effect combinations
        self.intensity = 0.8  # dose applied to the active pack (server-side in remote mode)

        # Token configuration presets
        self.token_presets = {
            "short": {"max_tokens": 50, "min_tokens": 5},
            "medium": {"max_tokens": 150, "min_tokens": 10},
            "long": {"max_tokens": 300, "min_tokens": 20},
            "very_long": {"max_tokens": 500, "min_tokens": 30}
        }
        self.current_token_preset = "medium"  # Default to medium

        self.emotion_tracker = None  # local-only; created in _load_model()
        self.chat_session_id = f"chat_{int(os.getpid())}_{int(time.time())}"

        if self.remote is not None:
            self._load_remote()
        else:
            # Load model and setup using centralized system
            self._load_model()
            self._setup_neuromodulation()

    def _load_remote(self):
        """Set up remote (RunPod HTTP) mode: no local model, packs listed from config."""
        self.model_name = getattr(self.remote, "model", None) or "remote"
        self.available_packs = _list_packs_from_config()
        self.neuromod_tool = None
        print("✅ Connected to RunPod endpoint (HTTP)")
        print(f"   Endpoint: {getattr(self.remote, 'endpoint_id', '?')}")
        print(f"   Model:    {self.model_name}")
        print(f"   Packs:    {len(self.available_packs)} available (applied server-side per request)")
        # Probe health (cheap GET); a scale-to-zero endpoint is 'healthy' even with 0 warm workers.
        try:
            if hasattr(self.remote, "is_available") and self.remote.is_available():
                print("   Health:   reachable")
        except Exception:
            pass
    
    def _load_model(self):
        """Load the language model using centralized model support (LOCAL mode)"""
        try:
            # Heavy imports are deferred to here so --remote mode never needs torch.
            from neuromod.neuromod_factory import create_neuromod_tool
            from neuromod.testing.simple_emotion_tracker import SimpleEmotionTracker

            # Set Hugging Face token if available (gated local models)
            if os.getenv('HUGGINGFACE_HUB_TOKEN'):
                from huggingface_hub import login
                login(os.getenv('HUGGINGFACE_HUB_TOKEN'))

            self.emotion_tracker = SimpleEmotionTracker()

            print(f"Loading model using centralized system...")

            # Create neuromod tool with centralized model loading
            self.neuromod_tool, model_info = create_neuromod_tool(
                model_name=self.model_name,
                test_mode=self.test_mode
            )
            
            # Extract model and tokenizer
            self.model = self.neuromod_tool.model
            self.tokenizer = self.neuromod_tool.tokenizer
            self.model_name = model_info.get('name', self.model_name)
            
            print(f"✅ Model loaded successfully")
            print(f"   Model: {self.model_name}")
            print(f"   Backend: {model_info.get('backend', 'unknown')}")
            print(f"   Size: {model_info.get('size', 'unknown')}")
            print(f"   Parameters: {model_info.get('parameters', 'unknown')}")
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise
    
    def _setup_neuromodulation(self):
        """Setup the neuromodulation system (already done in _load_model)"""
        if self.neuromod_tool:
            # Get available packs from the neuromod tool
            # Get packs from the registry (not pack_manager)
            if hasattr(self.neuromod_tool, 'registry') and self.neuromod_tool.registry:
                self.available_packs = list(self.neuromod_tool.registry.list_packs())
            else:
                self.available_packs = []
            print("✅ Neuromodulation system already initialized")
            print(f"Available packs: {len(self.available_packs)}")
        else:
            print("⚠️  Neuromodulation system not available")
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
        # Remote mode: no local model. The endpoint applies the pack per request, and the
        # handler accepts a single pack_name, so we track one active pack + the intensity and
        # send them with each generation.
        if self.remote is not None:
            if not pack_names:
                self.active_packs = []
                print("✅ Using baseline (no packs)")
                return
            if len(pack_names) > 1:
                print(f"⚠️  Remote endpoint applies one pack per request; using '{pack_names[0]}'.")
            self.active_packs = [pack_names[0]]
            print(f"✅ Active pack: {self.active_packs[0]} (intensity {self.intensity:.2f}, applied server-side)")
            return

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
                result = self.neuromod_tool.apply(pack_name, intensity=self.intensity)
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
    
    def generate_response(self, prompt: str, max_tokens: int = None, min_tokens: int = None) -> str:
        """Generate response with current neuromodulation state and emotion tracking"""
        # Use current preset if no tokens specified
        if max_tokens is None or min_tokens is None:
            preset = self.token_presets[self.current_token_preset]
            max_tokens = max_tokens or preset["max_tokens"]
            min_tokens = min_tokens or preset["min_tokens"]

        # Remote mode: one HTTP call to the endpoint; the worker applies the pack + generates.
        if self.remote is not None:
            return self._generate_remote(prompt, max_tokens)

        try:
            import torch
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
            
            # Move inputs to the same device as the model
            if torch.cuda.is_available() and next(self.model.parameters()).is_cuda:
                inputs = {k: v.cuda() for k, v in inputs.items()}
            else:
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
                    min_new_tokens=min_tokens,
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
            
            # Track emotions for this response
            self._track_emotions(response, prompt)
            
            return response if response else "I don't have a response for that."
            
        except Exception as e:
            print(f"❌ Generation error: {e}")
            return "Sorry, I encountered an error while generating a response."

    def _generate_remote(self, prompt: str, max_tokens: int) -> str:
        """Generate via the RunPod HTTP endpoint (no local model/torch)."""
        pack = self.active_packs[0] if self.active_packs else None
        try:
            t0 = time.time()
            res = self.remote.generate_text(
                prompt,
                max_tokens=max_tokens,
                temperature=0.7,
                pack_name=pack,
                intensity=self.intensity,
            )
            text = (res.get("text") or "").strip()
            # Surface server-provided emotions (if the handler computed any) + cost/latency.
            emotions = res.get("emotions") or {}
            if emotions:
                changed = [f"{k}: {v}" for k, v in emotions.items() if v in ("up", "down")]
                print(f"🎭 Emotions: {' | '.join(changed) if changed else 'stable'}")
            gpu = res.get("gpu_seconds")
            meta = f"⏱️  {time.time() - t0:.1f}s wall"
            if gpu is not None:
                meta += f" | {gpu:.2f} GPU-s"
            if pack:
                meta += f" | pack={pack}@{self.intensity:.2f}"
            print(meta)
            return text if text else "I don't have a response for that."
        except Exception as e:
            print(f"❌ Remote generation error: {e}")
            print("   (A scale-to-zero endpoint's first request pays a cold start — if this was a "
                  "timeout, try again; the worker may now be warm.)")
            return "Sorry, I encountered an error talking to the endpoint."

    def chat(self):
        """Main chat loop"""
        print("\n" + "=" * 60)
        print("🧠 Neuromodulation Chat Interface")
        print("=" * 60)
        print(f"Model: {self.model_name}")
        if self.remote is not None:
            print(f"🌐 Backend: RunPod endpoint (HTTP) — packs applied server-side")
            print(f"💊 Intensity: {self.intensity:.2f}  (change with /intensity)")
        else:
            print(f"🎭 Emotion tracking: Active")

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
        print("  /emotions - Show emotion tracking summary")
        print("  /export_emotions - Export emotion results to file")
        print("  /tokens - Show token configuration options")
        print("  /set_tokens - Change response length (short/medium/long/very_long)")
        print("  /intensity - Set pack dose/intensity (0.0-1.0)")
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
            
            # Show token configuration
            current_preset = self.token_presets[self.current_token_preset]
            print(f"  Response length: {self.current_token_preset} (max={current_preset['max_tokens']}, min={current_preset['min_tokens']})")
            
            # Show emotion tracking status
            summary = self.get_emotion_summary()
            if summary:
                print(f"  Emotion tracking: Active ({summary.get('total_states', 0)} assessments)")
                print(f"  Current valence: {summary.get('valence_trend', 'neutral')}")
            else:
                print(f"  Emotion tracking: Inactive")
        
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
        
        elif cmd == "/emotions":
            self.show_emotion_summary()
        
        elif cmd == "/export_emotions":
            self.export_emotion_results()
        
        elif cmd == "/tokens":
            self.show_token_options()
        
        elif cmd.startswith("/set_tokens"):
            self.set_token_preset(command)

        elif cmd.startswith("/intensity"):
            self.set_intensity(command)

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
            print("  /emotions - Show emotion tracking summary")
            print("  /export_emotions - Export emotion results to file")
            print("  /tokens - Show token configuration options")
            print("  /set_tokens - Change response length (short/medium/long/very_long)")
            print("  /intensity - Set pack dose/intensity (0.0-1.0)")
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
    
    def _track_emotions(self, response: str, context: str = ""):
        """Track emotions for the generated response"""
        try:
            # Track emotion changes
            latest_state = self.emotion_tracker.assess_emotion_change(response, self.chat_session_id, context)
            
            if latest_state:
                # Check for any emotion changes
                emotion_changes = []
                for emotion in ['joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust', 'trust', 'anticipation']:
                    emotion_value = getattr(latest_state, emotion)
                    if emotion_value in ['up', 'down']:
                        emotion_changes.append(f"{emotion}: {emotion_value}")
                
                if emotion_changes:
                    print(f"🎭 Emotions: {' | '.join(emotion_changes)} | Valence: {latest_state.valence}")
                else:
                    print(f"🎭 Emotions: stable | Valence: {latest_state.valence}")
                    
        except Exception as e:
            print(f"⚠️ Emotion tracking error: {e}")
    
    def get_emotion_summary(self):
        """Get a summary of emotional changes during the chat session"""
        if self.emotion_tracker is None:
            return None  # remote mode (or tracker unavailable): no local emotion tracking
        try:
            return self.emotion_tracker.get_emotion_summary()
        except Exception as e:
            print(f"⚠️ Error getting emotion summary: {e}")
            return None
    
    def export_emotion_results(self, filename: str = None):
        """Export emotion tracking results to file"""
        try:
            if not filename:
                filename = f"outputs/reports/emotion/emotion_results_{self.chat_session_id}.json"
            self.emotion_tracker.export_results(filename)
            print(f"💾 Emotion results exported to: {filename}")
        except Exception as e:
            print(f"⚠️ Error exporting emotion results: {e}")
    
    def show_emotion_summary(self):
        """Show a summary of emotional changes during the chat session"""
        try:
            summary = self.get_emotion_summary()
            if not summary:
                print("❌ No emotion data available")
                return
            
            print(f"\n🎭 Emotion Tracking Summary:")
            print("=" * 40)
            print(f"  Total assessments: {summary.get('total_states', 0)}")
            print(f"  Overall valence: {summary.get('valence_trend', 'neutral')}")
            
            # Show emotion changes
            emotion_changes = summary.get('emotion_changes', {})
            if emotion_changes:
                print(f"\n  📊 Emotion Changes:")
                for emotion, counts in emotion_changes.items():
                    up_count = counts.get('up', 0)
                    down_count = counts.get('down', 0)
                    stable_count = counts.get('stable', 0)
                    
                    if up_count > 0 or down_count > 0:
                        print(f"    {emotion.capitalize()}: {up_count} up, {down_count} down, {stable_count} stable")
                    elif stable_count > 0:
                        print(f"    {emotion.capitalize()}: {stable_count} stable")
            else:
                print(f"  📊 No emotion changes detected")
                
        except Exception as e:
            print(f"⚠️ Error showing emotion summary: {e}")

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
    
    def cleanup(self):
        """Clean up resources using centralized system"""
        if self.remote is not None:
            print("🧹 Closed remote chat session")
            return

        if self.neuromod_tool:
            from neuromod.neuromod_factory import cleanup_neuromod_tool
            cleanup_neuromod_tool(self.neuromod_tool)

        # Clear references
        self.model = None
        self.tokenizer = None
        self.neuromod_tool = None

        # Force garbage collection
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        print(f"🧹 Cleaned up chat session")
    
    def show_token_options(self):
        """Show available token configuration options"""
        print(f"\n🎯 Token Configuration Options:")
        print(f"Current preset: {self.current_token_preset}")
        print()
        
        for preset_name, config in self.token_presets.items():
            marker = "👉 " if preset_name == self.current_token_preset else "   "
            print(f"{marker}{preset_name}: max={config['max_tokens']}, min={config['min_tokens']}")
        
        print(f"\nUsage: /set_tokens <preset_name>")
        print(f"Example: /set_tokens short")
    
    def set_token_preset(self, command: str):
        """Set the token generation preset"""
        parts = command.split()
        if len(parts) < 2:
            print("❌ Usage: /set_tokens <preset_name>")
            print("Available presets:", ", ".join(self.token_presets.keys()))
            return
        
        preset_name = parts[1].lower()
        if preset_name not in self.token_presets:
            print(f"❌ Unknown preset: {preset_name}")
            print("Available presets:", ", ".join(self.token_presets.keys()))
            return
        
        old_preset = self.current_token_preset
        self.current_token_preset = preset_name
        new_config = self.token_presets[preset_name]
        
        print(f"✅ Changed response length from '{old_preset}' to '{preset_name}'")
        print(f"   New settings: max={new_config['max_tokens']}, min={new_config['min_tokens']}")

    def set_intensity(self, command: str):
        """Set the dose/intensity applied to the active pack (0.0-1.0)."""
        parts = command.split()
        if len(parts) < 2:
            print(f"❌ Usage: /intensity <0.0-1.0>   (current: {self.intensity:.2f})")
            return
        try:
            val = float(parts[1])
        except ValueError:
            print("❌ Intensity must be a number between 0.0 and 1.0")
            return
        if not 0.0 <= val <= 1.0:
            print("❌ Intensity must be between 0.0 and 1.0")
            return
        self.intensity = val
        print(f"✅ Intensity set to {self.intensity:.2f}")
        # Local mode holds pack state on the model, so re-apply at the new dose. Remote mode
        # sends intensity per request, so nothing to re-apply.
        if self.remote is None and self.active_packs:
            self.apply_packs(list(self.active_packs))

def _build_remote_interface(endpoint_id=None, api_key=None, model=None):
    """Construct a RunPodModelInterface from args/env (torch-free)."""
    from api.runpod_client import RunPodModelInterface, interface_from_env
    endpoint_id = endpoint_id or os.environ.get("RUNPOD_ENDPOINT_ID")
    api_key = api_key or os.environ.get("RUNPOD_API_KEY")
    if endpoint_id and api_key:
        return RunPodModelInterface(endpoint_id, api_key, model=model)
    # Fall back to env-only helper (raises a clear error if unset)
    return interface_from_env(model=model)


def main():
    """Main function"""
    import argparse
    parser = argparse.ArgumentParser(
        description="Neuromodulation chat — local (in-process model) or --remote (RunPod HTTP endpoint).")
    parser.add_argument("--remote", action="store_true",
                        help="Talk to a deployed RunPod Serverless endpoint over HTTP (no local model / no torch). "
                             "Uses RUNPOD_ENDPOINT_ID + RUNPOD_API_KEY unless --endpoint-id/--api-key given.")
    parser.add_argument("--endpoint-id", default=None, help="RunPod endpoint id (else $RUNPOD_ENDPOINT_ID)")
    parser.add_argument("--api-key", default=None, help="RunPod API key (else $RUNPOD_API_KEY; prefer the env var)")
    parser.add_argument("--model", default=None,
                        help="Model id/alias to request from the endpoint or load locally (e.g. gpt-oss-120b)")
    args, _ = parser.parse_known_args()

    print("🚀 Neuromodulation Chat Interface")
    print("=" * 50)

    # ---- Remote mode: skip the local model menu entirely (no torch import) ----
    if args.remote or (args.endpoint_id and args.api_key):
        try:
            iface = _build_remote_interface(args.endpoint_id, args.api_key, args.model)
        except Exception as e:
            print(f"❌ Could not set up remote endpoint: {e}")
            return 1
        try:
            chat = NeuromodChat(model_name=args.model, remote=iface)
            chat.chat()
        except Exception as e:
            print(f"❌ Failed to start remote chat: {e}")
            return 1
        return 0

    # Show available models in production mode
    from neuromod.model_support import create_model_support

    print("\n🤖 Model Selection:")
    print("1. Use recommended model (test mode)")
    print("2. Use recommended model (production mode)")
    print("3. Enter custom model name")
    print("4. Show available models and select")
    print("5. Connect to a RunPod endpoint (remote HTTP — no local model)")

    remote_iface = None
    while True:
        try:
            choice = input(f"\nSelect option (1-5): ").strip()

            if choice == "1":
                model_name = None
                test_mode = True
                break
            elif choice == "2":
                model_name = None
                test_mode = False
                break
            elif choice == "3":
                model_name = input("Enter model name: ").strip()
                test_mode = False  # Use production mode for custom models
                break
            elif choice == "4":
                # Show available models
                manager = create_model_support(test_mode=False)
                available_models = manager.get_available_models()
                
                print("\n📋 Available Models (Production Mode):")
                print("=" * 60)
                for i, model in enumerate(available_models, 1):
                    # Get model info if possible
                    try:
                        config = manager.model_configs.get(model)
                        if config:
                            size_info = f"Size: {config.size.value}"
                            quant_info = f", Quantization: {config.quantization}" if config.quantization else ""
                            print(f"{i}. {model}")
                            print(f"   {size_info}{quant_info}, Max length: {config.max_length}")
                        else:
                            print(f"{i}. {model}")
                    except:
                        print(f"{i}. {model}")
                
                print("\n💡 Popular ~30B Models:")
                print("   • Qwen/Qwen2.5-32B-Instruct (32B, 4-bit quantized, ~20-24GB VRAM, no auth required)")
                print("   • meta-llama/Llama-4-Scout-17B-16E-Instruct (17B, 4-bit quantized, ~16-18GB VRAM, requires auth)")
                print("   • meta-llama/Llama-3.1-70B-Instruct (70B, 4-bit quantized, ~40GB VRAM, requires auth)")
                print("   • openai/gpt-oss-20b (20B, 4-bit quantized, ~50GB VRAM, open-source weights)")
                print("   • openai/gpt-oss-120b (120B, multi-GPU / >80GB VRAM recommended)")
                
                try:
                    model_idx = int(input("\nSelect model number: ").strip()) - 1
                    if 0 <= model_idx < len(available_models):
                        model_name = available_models[model_idx]
                        test_mode = False
                        break
                    else:
                        print(f"❌ Invalid selection. Please choose 1-{len(available_models)}")
                        continue
                except ValueError:
                    print("❌ Invalid input. Please enter a number.")
                    continue
            elif choice == "5":
                # Remote HTTP endpoint (no local model).
                ep = os.environ.get("RUNPOD_ENDPOINT_ID") or input("RunPod endpoint id: ").strip()
                key = os.environ.get("RUNPOD_API_KEY") or input("RunPod API key: ").strip()
                mdl = input("Model id/alias (blank = endpoint default): ").strip() or None
                try:
                    remote_iface = _build_remote_interface(ep, key, mdl)
                    model_name = mdl
                    test_mode = False
                    break
                except Exception as e:
                    print(f"❌ Could not set up remote endpoint: {e}")
                    continue
            else:
                print("Please enter 1, 2, 3, 4, or 5")
        except (ValueError, KeyboardInterrupt):
            print("Invalid input")

    # Create and start chat
    try:
        if remote_iface is not None:
            chat = NeuromodChat(model_name=model_name, remote=remote_iface)
        else:
            print(f"\n✅ Selected: {model_name or 'recommended'} ({'test' if test_mode else 'production'} mode)")
            chat = NeuromodChat(model_name, test_mode)
        chat.chat()
    except Exception as e:
        print(f"❌ Failed to start chat: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
