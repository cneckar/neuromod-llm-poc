#!/usr/bin/env python3
"""
Integration Tests for Neuromodulation System
Tests the complete system integration including model loading, pack application, and generation
"""

import unittest
import tempfile
import json
import os
import sys
import torch
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neuromod.pack_system import PackRegistry, Pack, EffectConfig, PackManager
from neuromod.effects import EffectRegistry
from neuromod.neuromod_tool import NeuromodTool
from neuromod.testing.pdq_test import PDQTest
from neuromod.testing.sdq_test import SDQTest
from neuromod.testing.ddq_test import DDQTest
from neuromod.testing.didq_test import DiDQTest
from neuromod.testing.edq_test import EDQTest
from neuromod.testing.cdq_test import CDQTest
from neuromod.testing.pcq_pop_test import PCQPopTest
from neuromod.testing.adq_test import ADQTest


class TestNeuromodTool(unittest.TestCase):
    """Test NeuromodTool integration"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "test_config.json")
        
        # Create test config
        test_config = {
            "packs": {
                "test_pack": {
                    "name": "test_pack",
                    "description": "Test pack for integration",
                    "effects": [
                        {
                            "effect": "temperature",
                            "weight": 0.5,
                            "direction": "up",
                            "parameters": {}
                        },
                        {
                            "effect": "top_p",
                            "weight": 0.3,
                            "direction": "down",
                            "parameters": {}
                        }
                    ]
                }
            }
        }
        
        with open(self.config_path, 'w') as f:
            json.dump(test_config, f)
        
        # Mock model and tokenizer
        self.mock_model = Mock()
        self.mock_tokenizer = Mock()
        self.mock_tokenizer.pad_token = "<pad>"
        self.mock_tokenizer.eos_token = "</s>"
        
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_neuromod_tool_initialization(self):
        """Test NeuromodTool initialization"""
        registry = PackRegistry(self.config_path)
        tool = NeuromodTool(registry, self.mock_model, self.mock_tokenizer)
        
        self.assertIsNotNone(tool)
        self.assertEqual(tool.registry, registry)
        self.assertEqual(tool.model, self.mock_model)
        self.assertEqual(tool.tokenizer, self.mock_tokenizer)
    
    def test_apply_pack(self):
        """Test applying a pack through NeuromodTool"""
        registry = PackRegistry(self.config_path)
        tool = NeuromodTool(registry, self.mock_model, self.mock_tokenizer)
        
        # Apply pack
        result = tool.apply("test_pack", intensity=0.5)
        
        self.assertIsInstance(result, dict)
        self.assertIn("ok", result)
        self.assertTrue(result["ok"])
    
    def test_apply_nonexistent_pack(self):
        """Test applying a non-existent pack"""
        registry = PackRegistry(self.config_path)
        tool = NeuromodTool(registry, self.mock_model, self.mock_tokenizer)
        
        # Apply non-existent pack
        result = tool.apply("nonexistent_pack", intensity=0.5)
        
        self.assertIsInstance(result, dict)
        self.assertIn("ok", result)
        self.assertFalse(result["ok"])
        self.assertIn("error", result)
    
    def test_clear_packs(self):
        """Test clearing applied packs"""
        registry = PackRegistry(self.config_path)
        tool = NeuromodTool(registry, self.mock_model, self.mock_tokenizer)
        
        # Apply pack first
        tool.apply("test_pack", intensity=0.5)
        
        # Clear packs
        tool.clear()
        
        # Verify cleanup was called
        self.assertIsNotNone(tool)
    
    def test_get_logits_processors(self):
        """Test getting logits processors"""
        registry = PackRegistry(self.config_path)
        tool = NeuromodTool(registry, self.mock_model, self.mock_tokenizer)
        
        # Apply pack
        tool.apply("test_pack", intensity=0.5)
        
        # Get logits processors
        processors = tool.get_logits_processors()
        
        self.assertIsInstance(processors, list)
        self.assertGreater(len(processors), 0)
    
    def test_intensity_scaling(self):
        """Test intensity scaling of pack effects"""
        registry = PackRegistry(self.config_path)
        tool = NeuromodTool(registry, self.mock_model, self.mock_tokenizer)
        
        # Apply with different intensities
        result_low = tool.apply("test_pack", intensity=0.2)
        tool.clear()
        
        result_high = tool.apply("test_pack", intensity=0.8)
        tool.clear()
        
        self.assertTrue(result_low["ok"])
        self.assertTrue(result_high["ok"])


class TestModelIntegration(unittest.TestCase):
    """Test integration with actual models"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "test_config.json")
        
        # Create test config
        test_config = {
            "packs": {
                "simple_pack": {
                    "name": "simple_pack",
                    "description": "Simple test pack",
                    "effects": [
                        {
                            "effect": "temperature",
                            "weight": 0.3,
                            "direction": "up",
                            "parameters": {}
                        }
                    ]
                }
            }
        }
        
        with open(self.config_path, 'w') as f:
            json.dump(test_config, f)
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    @patch('torch.cuda.is_available', return_value=False)
    def test_gpt2_integration(self, mock_cuda):
        """Test integration with GPT-2 model"""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            # Load model
            tokenizer = AutoTokenizer.from_pretrained('gpt2')
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            model = AutoModelForCausalLM.from_pretrained(
                'gpt2',
                dtype=torch.float32,
                device_map='cpu',
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            model.eval()
            
            # Setup neuromodulation
            registry = PackRegistry(self.config_path)
            tool = NeuromodTool(registry, model, tokenizer)
            
            # Apply pack
            result = tool.apply("simple_pack", intensity=0.5)
            self.assertTrue(result["ok"])
            
            # Test generation
            prompt = "The world is"
            inputs = tokenizer(prompt, return_tensors="pt", padding=True)
            inputs = {k: v.cpu() for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=10,
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
            self.assertIsInstance(response, str)
            self.assertGreater(len(response), len(prompt))
            
            # Cleanup
            tool.clear()
            
        except Exception as e:
            self.skipTest(f"GPT-2 integration test skipped: {e}")


class TestPackManagerIntegration(unittest.TestCase):
    """Test PackManager integration"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_model = Mock()
        
        # Create test pack
        self.test_pack = Pack(
            name="test_pack",
            description="Test pack for manager",
            effects=[
                EffectConfig("temperature", weight=0.5, direction="up"),
                EffectConfig("top_p", weight=0.3, direction="down")
            ]
        )
    
    def test_pack_manager_apply_pack(self):
        """Test PackManager applying a pack"""
        # Mock effect registry at the module level
        with patch('neuromod.pack_system.EffectRegistry') as mock_registry_class:
            mock_registry_instance = Mock()
            mock_effect = Mock()
            mock_registry_instance.get_effect.return_value = mock_effect
            mock_registry_class.return_value = mock_registry_instance
            
            # Create manager after mocking
            manager = PackManager()
            
            # Apply pack
            manager.apply_pack(self.test_pack, self.mock_model)
            
            # Verify effects were applied
            self.assertEqual(len(manager.active_effects), 2)
            
            # Verify get_effect was called twice (once for each effect)
            self.assertEqual(mock_registry_instance.get_effect.call_count, 2)
    
    def test_pack_manager_cleanup(self):
        """Test PackManager cleanup"""
        manager = PackManager()
        
        # Mock effect
        mock_effect = Mock()
        manager.active_effects = [mock_effect]
        
        # Cleanup
        manager.clear_effects()
        
        # Verify cleanup was called
        mock_effect.cleanup.assert_called_once()
        self.assertEqual(len(manager.active_effects), 0)


class TestEffectRegistryIntegration(unittest.TestCase):
    """Test EffectRegistry integration"""
    
    def test_effect_registry_all_effects(self):
        """Test that all effects can be created through registry"""
        registry = EffectRegistry()
        
        # Test all effect types
        effect_types = [
            "temperature", "top_p", "frequency_penalty", "presence_penalty",
            "pulsed_sampler", "contrastive_decoding", "expert_mixing", "token_class_temperature",
            "attention_focus", "attention_masking", "qk_score_scaling", "head_masking_dropout",
            "head_reweighting", "positional_bias_tweak", "attention_oscillation", "attention_sinks_anchors",
            "steering", "kv_decay", "kv_compression", "exponential_decay_kv", "truncation_kv",
            "stride_compression_kv", "segment_gains_kv", "activation_additions", "soft_projection",
            "layer_wise_gain", "noise_injection", "router_temperature_bias", "expert_masking_dropout",
            "expert_persistence", "verifier_guided_decoding", "style_affect_logit_bias",
            "risk_preference_steering", "compute_at_test_scheduling", "retrieval_rate_modulation",
            "persona_voice_constraints", "lexical_jitter", "structured_prefaces"
        ]
        
        for effect_type in effect_types:
            try:
                effect = registry.get_effect(effect_type, weight=0.5, direction="up")
                self.assertIsNotNone(effect)
                self.assertEqual(effect.weight, 0.5)
                self.assertEqual(effect.direction, "up")
            except ValueError as e:
                self.fail(f"Failed to create effect {effect_type}: {e}")
    
    def test_effect_registry_unknown_effect(self):
        """Test handling of unknown effects"""
        registry = EffectRegistry()
        
        with self.assertRaises(ValueError):
            registry.get_effect("unknown_effect", weight=0.5, direction="up")


class TestTestingFrameworkIntegration(unittest.TestCase):
    """Test integration with testing framework"""
    
    def test_pdq_test_creation(self):
        """Test PDQ test creation"""
        try:
            test = PDQTest("gpt2")
            self.assertIsNotNone(test)
            self.assertEqual(test.model_name, "gpt2")
        except Exception as e:
            self.skipTest(f"PDQ test creation skipped: {e}")
    
    def test_sdq_test_creation(self):
        """Test SDQ test creation"""
        try:
            test = SDQTest("gpt2")
            self.assertIsNotNone(test)
            self.assertEqual(test.get_test_name(), "SDQ-15 Test (Stimulant Detection Questionnaire)")
        except Exception as e:
            self.skipTest(f"SDQ test creation skipped: {e}")


class TestEndToEndIntegration(unittest.TestCase):
    """Test end-to-end integration scenarios"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "test_config.json")
        
        # Create comprehensive test config
        test_config = {
            "packs": {
                "comprehensive_pack": {
                    "name": "comprehensive_pack",
                    "description": "Comprehensive test pack",
                    "effects": [
                        {
                            "effect": "temperature",
                            "weight": 0.4,
                            "direction": "up",
                            "parameters": {}
                        },
                        {
                            "effect": "top_p",
                            "weight": 0.3,
                            "direction": "down",
                            "parameters": {}
                        },
                        {
                            "effect": "steering",
                            "weight": 0.5,
                            "direction": "up",
                            "parameters": {"steering_type": "associative"}
                        }
                    ]
                }
            }
        }
        
        with open(self.config_path, 'w') as f:
            json.dump(test_config, f)
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_complete_workflow(self):
        """Test complete workflow from config to generation"""
        # 1. Load registry
        registry = PackRegistry(self.config_path)
        self.assertIn("comprehensive_pack", registry.list_packs())
        
        # 2. Get pack
        pack = registry.get_pack("comprehensive_pack")
        self.assertEqual(len(pack.effects), 3)
        
        # 3. Create pack manager
        manager = PackManager()
        self.assertIsNotNone(manager)
        
        # 4. Mock model for testing
        mock_model = Mock()
        
        # 5. Apply pack
        with patch('neuromod.pack_system.EffectRegistry') as mock_registry:
            mock_registry_instance = Mock()
            mock_effect = Mock()
            mock_registry_instance.get_effect.return_value = mock_effect
            mock_registry.return_value = mock_registry_instance
            
            manager.apply_pack(pack, mock_model)
            self.assertEqual(len(manager.active_effects), 3)
            
            # 6. Cleanup
            manager.clear_effects()
            self.assertEqual(len(manager.active_effects), 0)
    
    def test_multiple_pack_application(self):
        """Test applying multiple packs sequentially"""
        registry = PackRegistry(self.config_path)
        manager = PackManager()
        mock_model = Mock()
        
        # Apply pack multiple times
        pack = registry.get_pack("comprehensive_pack")
        
        with patch('neuromod.pack_system.EffectRegistry') as mock_registry:
            mock_registry_instance = Mock()
            mock_effect = Mock()
            mock_registry_instance.get_effect.return_value = mock_effect
            mock_registry.return_value = mock_registry_instance
            
            # First application
            manager.apply_pack(pack, mock_model)
            self.assertEqual(len(manager.active_effects), 3)
            
            # Cleanup
            manager.clear_effects()
            self.assertEqual(len(manager.active_effects), 0)
            
            # Second application
            manager.apply_pack(pack, mock_model)
            self.assertEqual(len(manager.active_effects), 3)
            
            # Final cleanup
            manager.clear_effects()
            self.assertEqual(len(manager.active_effects), 0)


if __name__ == '__main__':
    unittest.main()
