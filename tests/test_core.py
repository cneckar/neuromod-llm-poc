#!/usr/bin/env python3
"""
Unit Tests for Core Neuromodulation System
Tests pack system, registry, and basic functionality
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


class TestEffectConfig(unittest.TestCase):
    """Test EffectConfig dataclass functionality"""
    
    def test_effect_config_creation(self):
        """Test creating EffectConfig with various parameters"""
        # Basic creation
        config = EffectConfig("temperature", weight=0.5, direction="up")
        self.assertEqual(config.effect, "temperature")
        self.assertEqual(config.weight, 0.5)
        self.assertEqual(config.direction, "up")
        self.assertEqual(config.parameters, {})
        
        # With parameters
        params = {"layers": "mid", "scaling_type": "uniform"}
        config = EffectConfig("attention_focus", weight=0.7, direction="down", parameters=params)
        self.assertEqual(config.parameters, params)
        
        # Weight validation (no clamping in EffectConfig, validation happens in Pack)
        config = EffectConfig("temperature", weight=1.5, direction="up")
        self.assertEqual(config.weight, 1.5)  # No clamping in EffectConfig
        
        config = EffectConfig("temperature", weight=-0.5, direction="up")
        self.assertEqual(config.weight, -0.5)  # No clamping in EffectConfig
    
    def test_effect_config_from_dict(self):
        """Test creating EffectConfig from dictionary"""
        data = {
            "effect": "temperature",
            "weight": 0.6,
            "direction": "down",
            "parameters": {"layers": "all"}
        }
        
        config = EffectConfig.from_dict(data)
        self.assertEqual(config.effect, "temperature")
        self.assertEqual(config.weight, 0.6)
        self.assertEqual(config.direction, "down")
        self.assertEqual(config.parameters, {"layers": "all"})
        
        # Test with missing optional fields
        data_minimal = {"effect": "top_p"}
        config = EffectConfig.from_dict(data_minimal)
        self.assertEqual(config.effect, "top_p")
        self.assertEqual(config.weight, 0.5)  # Default
        self.assertEqual(config.direction, "up")  # Default
        self.assertEqual(config.parameters, {})  # Default


class TestPack(unittest.TestCase):
    """Test Pack dataclass functionality"""
    
    def test_pack_creation(self):
        """Test creating Pack with effects"""
        effects = [
            EffectConfig("temperature", weight=0.5, direction="up"),
            EffectConfig("top_p", weight=0.3, direction="down")
        ]
        
        pack = Pack(
            name="test_pack",
            description="Test pack for unit testing",
            effects=effects
        )
        
        self.assertEqual(pack.name, "test_pack")
        self.assertEqual(pack.description, "Test pack for unit testing")
        self.assertEqual(len(pack.effects), 2)
        self.assertEqual(pack.effects[0].effect, "temperature")
        self.assertEqual(pack.effects[1].effect, "top_p")
    
    def test_pack_validation(self):
        """Test pack validation"""
        # Valid pack
        effects = [EffectConfig("temperature", weight=0.5, direction="up")]
        pack = Pack("valid_pack", "Valid pack", effects)
        self.assertIsNotNone(pack)
        
        # Invalid weight
        with self.assertRaises(ValueError):
            effects = [EffectConfig("temperature", weight=1.5, direction="up")]
            Pack("invalid_pack", "Invalid pack", effects)
    
    def test_pack_from_dict(self):
        """Test creating Pack from dictionary"""
        data = {
            "name": "test_pack",
            "description": "Test pack from dict",
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
                    "parameters": {"layers": "mid"}
                }
            ]
        }
        
        pack = Pack.from_dict(data)
        self.assertEqual(pack.name, "test_pack")
        self.assertEqual(pack.description, "Test pack from dict")
        self.assertEqual(len(pack.effects), 2)


class TestEffectRegistry(unittest.TestCase):
    """Test EffectRegistry functionality"""
    
    def test_effect_registry_initialization(self):
        """Test EffectRegistry initialization"""
        registry = EffectRegistry()
        self.assertIsNotNone(registry)
        self.assertIsInstance(registry.effects, dict)
        self.assertGreater(len(registry.effects), 0)
    
    def test_list_effects(self):
        """Test listing available effects"""
        registry = EffectRegistry()
        effects = registry.list_effects()
        
        self.assertIsInstance(effects, list)
        self.assertGreater(len(effects), 0)
        
        # Check for expected effect categories
        expected_categories = ["temperature", "top_p", "attention", "steering", "kv"]
        found_effects = [effect for effect in effects if any(cat in effect for cat in expected_categories)]
        self.assertGreater(len(found_effects), 0)
    
    def test_get_effect(self):
        """Test getting effect instances"""
        registry = EffectRegistry()
        
        # Test getting a sampler effect
        effect = registry.get_effect("temperature", weight=0.5, direction="up")
        self.assertIsNotNone(effect)
        self.assertEqual(effect.weight, 0.5)
        self.assertEqual(effect.direction, "up")
        
        # Test getting an attention effect
        effect = registry.get_effect("attention_focus", weight=0.7, direction="down")
        self.assertIsNotNone(effect)
        self.assertEqual(effect.weight, 0.7)
        self.assertEqual(effect.direction, "down")
        
        # Test getting non-existent effect
        with self.assertRaises(ValueError):
            registry.get_effect("non_existent_effect")


class TestPackRegistry(unittest.TestCase):
    """Test PackRegistry functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "test_config.json")
        
        # Create test config
        test_config = {
            "packs": {
                "test_pack_1": {
                    "name": "test_pack_1",
                    "description": "First test pack",
                    "effects": [
                        {
                            "effect": "temperature",
                            "weight": 0.5,
                            "direction": "up",
                            "parameters": {}
                        }
                    ]
                },
                "test_pack_2": {
                    "name": "test_pack_2",
                    "description": "Second test pack",
                    "effects": [
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
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_pack_registry_initialization(self):
        """Test PackRegistry initialization"""
        registry = PackRegistry(self.config_path)
        self.assertIsNotNone(registry)
        self.assertEqual(len(registry.packs), 2)
    
    def test_load_packs(self):
        """Test loading packs from config"""
        registry = PackRegistry(self.config_path)
        packs = registry.list_packs()
        
        self.assertIn("test_pack_1", packs)
        self.assertIn("test_pack_2", packs)
        self.assertEqual(len(packs), 2)
    
    def test_get_pack(self):
        """Test getting pack by name"""
        registry = PackRegistry(self.config_path)
        
        pack = registry.get_pack("test_pack_1")
        self.assertIsNotNone(pack)
        self.assertEqual(pack.name, "test_pack_1")
        self.assertEqual(pack.description, "First test pack")
        self.assertEqual(len(pack.effects), 1)
        
        # Test getting non-existent pack
        with self.assertRaises(ValueError):
            registry.get_pack("non_existent_pack")
    
    def test_add_pack(self):
        """Test adding pack to registry"""
        registry = PackRegistry(self.config_path)
        
        new_pack = Pack(
            name="new_pack",
            description="New test pack",
            effects=[EffectConfig("steering", weight=0.4, direction="up")]
        )
        
        registry.add_pack(new_pack)
        self.assertIn("new_pack", registry.list_packs())
        
        # Test adding duplicate pack (overwrites existing)
        registry.add_pack(new_pack)
        self.assertIn("new_pack", registry.list_packs())
    
    def test_remove_pack(self):
        """Test removing pack from registry"""
        registry = PackRegistry(self.config_path)
        
        registry.remove_pack("test_pack_1")
        self.assertNotIn("test_pack_1", registry.list_packs())
        self.assertIn("test_pack_2", registry.list_packs())
        
        # Test removing non-existent pack (silently ignored)
        registry.remove_pack("non_existent_pack")
        # Should not raise an error
    
    def test_save_packs_to_json(self):
        """Test saving packs to JSON"""
        registry = PackRegistry(self.config_path)
        
        output_path = os.path.join(self.temp_dir, "output_config.json")
        registry.save_packs_to_json(output_path)
        
        # Verify file was created
        self.assertTrue(os.path.exists(output_path))
        
        # Verify content
        with open(output_path, 'r') as f:
            saved_config = json.load(f)
        
        self.assertIn("packs", saved_config)
        self.assertIn("test_pack_1", saved_config["packs"])
        self.assertIn("test_pack_2", saved_config["packs"])
    
    def test_reload_packs(self):
        """Test reloading packs from config"""
        registry = PackRegistry(self.config_path)
        initial_count = len(registry.list_packs())
        
        # Modify config file
        with open(self.config_path, 'r') as f:
            config = json.load(f)
        
        config["packs"]["test_pack_3"] = {
            "name": "test_pack_3",
            "description": "Third test pack",
            "effects": []
        }
        
        with open(self.config_path, 'w') as f:
            json.dump(config, f)
        
        # Reload
        registry.reload_packs()
        new_count = len(registry.list_packs())
        
        self.assertEqual(new_count, initial_count + 1)
        self.assertIn("test_pack_3", registry.list_packs())


class TestPackManager(unittest.TestCase):
    """Test PackManager functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_model = Mock()
        self.mock_tokenizer = Mock()
        
        # Create test pack
        self.test_pack = Pack(
            name="test_pack",
            description="Test pack",
            effects=[
                EffectConfig("temperature", weight=0.5, direction="up"),
                EffectConfig("top_p", weight=0.3, direction="down")
            ]
        )
    
    def test_pack_manager_initialization(self):
        """Test PackManager initialization"""
        manager = PackManager()
        self.assertIsNotNone(manager)
        self.assertEqual(len(manager.active_effects), 0)
    
    def test_apply_pack(self):
        """Test applying pack to model"""
        manager = PackManager()
        
        # Mock effect instances
        with patch('neuromod.pack_system.EffectRegistry') as mock_registry:
            mock_registry_instance = Mock()
            mock_registry_instance.get_effect.return_value = Mock()
            mock_registry.return_value = mock_registry_instance
            
            manager.apply_pack(self.test_pack, self.mock_model)
            
            # Verify effects were applied
            self.assertEqual(len(manager.active_effects), 2)
    
    def test_cleanup(self):
        """Test cleaning up applied effects"""
        manager = PackManager()
        
        # Mock effect instances
        mock_effect = Mock()
        manager.active_effects = [mock_effect]
        
        manager.clear_effects()
        
        # Verify cleanup was called
        mock_effect.cleanup.assert_called_once()
        self.assertEqual(len(manager.active_effects), 0)


if __name__ == '__main__':
    unittest.main()
