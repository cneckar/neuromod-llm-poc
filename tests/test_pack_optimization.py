"""
Tests for Pack Optimization System

Tests the drug design laboratory functionality.
"""

import unittest
import tempfile
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

from neuromod.optimization import (
    BehavioralTarget, TargetManager, PackOptimizer, 
    OptimizationMethod, EvaluationFramework, DrugDesignLab
)
from neuromod.pack_system import PackRegistry

class TestBehavioralTargets(unittest.TestCase):
    """Test behavioral target system"""
    
    def setUp(self):
        self.target_manager = TargetManager()
    
    def test_create_target(self):
        """Test creating a behavioral target"""
        target = self.target_manager.create_target(
            name="test_target",
            description="Test target for unit testing"
        )
        
        self.assertEqual(target.name, "test_target")
        self.assertEqual(target.description, "Test target for unit testing")
        self.assertEqual(len(target.targets), 0)
    
    def test_add_emotion_target(self):
        """Test adding emotion targets"""
        target = self.target_manager.create_target("test", "test")
        target.add_emotion_target("joy", 0.8, weight=2.0)
        
        self.assertEqual(len(target.targets), 1)
        self.assertEqual(target.targets[0].name, "emotion_joy")
        self.assertEqual(target.targets[0].target_value, 0.8)
        self.assertEqual(target.targets[0].weight, 2.0)
    
    def test_add_behavior_target(self):
        """Test adding behavior targets"""
        target = self.target_manager.create_target("test", "test")
        target.add_behavior_target("creativity", 0.7, weight=1.5)
        
        self.assertEqual(len(target.targets), 1)
        self.assertEqual(target.targets[0].name, "behavior_creativity")
        self.assertEqual(target.targets[0].target_value, 0.7)
        self.assertEqual(target.targets[0].weight, 1.5)
    
    def test_compute_loss(self):
        """Test loss computation"""
        target = self.target_manager.create_target("test", "test")
        target.add_emotion_target("joy", 0.8, weight=1.0)
        target.add_behavior_target("creativity", 0.6, weight=1.0)
        
        # Test with matching values
        actual_values = {"emotion_joy": 0.8, "behavior_creativity": 0.6}
        loss = target.compute_loss(actual_values)
        self.assertAlmostEqual(loss, 0.0, places=5)
        
        # Test with different values
        actual_values = {"emotion_joy": 0.5, "behavior_creativity": 0.3}
        loss = target.compute_loss(actual_values)
        self.assertGreater(loss, 0.0)
    
    def test_preset_targets(self):
        """Test preset targets are loaded"""
        targets = self.target_manager.list_targets()
        self.assertIn("joyful_social", targets)
        self.assertIn("creative_focused", targets)
        self.assertIn("calm_reflective", targets)
        
        # Test joyful_social target
        target = self.target_manager.get_target("joyful_social")
        self.assertEqual(target.name, "joyful_social")
        self.assertGreater(len(target.targets), 0)
        self.assertGreater(len(target.test_prompts), 0)

class TestEvaluationFramework(unittest.TestCase):
    """Test evaluation framework"""
    
    def setUp(self):
        self.evaluator = EvaluationFramework()
    
    def test_evaluate_single_text(self):
        """Test evaluating a single text"""
        text = "I am so happy and excited about this wonderful opportunity!"
        metrics = self.evaluator.evaluate_text(text)
        
        self.assertIsInstance(metrics.emotions, dict)
        self.assertIsInstance(metrics.behaviors, dict)
        self.assertIsInstance(metrics.metrics, dict)
        
        # Should detect positive sentiment
        self.assertGreater(metrics.emotions.get("joy", 0), 0)
    
    def test_evaluate_texts(self):
        """Test evaluating multiple texts"""
        texts = [
            "I am happy and joyful!",
            "This is a sad and depressing situation.",
            "I feel calm and peaceful."
        ]
        
        metrics = self.evaluator.evaluate_texts(texts)
        
        self.assertIsInstance(metrics.emotions, dict)
        self.assertIsInstance(metrics.behaviors, dict)
        self.assertIsInstance(metrics.metrics, dict)
    
    def test_compute_target_loss(self):
        """Test computing loss between target and actual metrics"""
        from neuromod.optimization.evaluation import BehavioralMetrics
        
        target_metrics = BehavioralMetrics()
        target_metrics.emotions = {"joy": 0.8}
        target_metrics.behaviors = {"creativity": 0.6}
        
        actual_metrics = BehavioralMetrics()
        actual_metrics.emotions = {"joy": 0.5}
        actual_metrics.behaviors = {"creativity": 0.3}
        
        loss = self.evaluator.compute_target_loss(target_metrics, actual_metrics)
        self.assertGreater(loss, 0.0)

class TestDrugDesignLab(unittest.TestCase):
    """Test drug design laboratory"""
    
    def setUp(self):
        self.lab = DrugDesignLab()
    
    def test_create_session(self):
        """Test creating a laboratory session"""
        session = self.lab.create_session("joyful_social", "none")
        
        self.assertEqual(session.target.name, "joyful_social")
        self.assertIsNotNone(session.base_pack)
        self.assertIsNone(session.optimized_pack)
        self.assertEqual(len(session.test_results), 0)
    
    def test_list_sessions(self):
        """Test listing sessions"""
        # Create a session
        session = self.lab.create_session("joyful_social", "none")
        
        # List sessions
        sessions = self.lab.list_sessions()
        self.assertEqual(len(sessions), 1)
        self.assertEqual(sessions[0]["session_id"], session.session_id)
    
    def test_save_load_session(self):
        """Test saving and loading sessions"""
        # Create session
        session = self.lab.create_session("joyful_social", "none")
        
        # Save session
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            self.lab.save_session(session.session_id, temp_path)
            
            # Load session
            loaded_session_id = self.lab.load_session(temp_path)
            
            self.assertEqual(loaded_session_id, session.session_id)
            self.assertIn(loaded_session_id, self.lab.sessions)
            
        finally:
            os.unlink(temp_path)

class TestPackOptimization(unittest.TestCase):
    """Test pack optimization (simplified tests)"""
    
    def setUp(self):
        self.pack_registry = PackRegistry()
        self.target_manager = TargetManager()
    
    def test_pack_registry_available(self):
        """Test that pack registry has packs available"""
        packs = self.pack_registry.list_packs()
        self.assertGreater(len(packs), 0)
        self.assertIn("none", packs)
    
    def test_target_manager_available(self):
        """Test that target manager has targets available"""
        targets = self.target_manager.list_targets()
        self.assertGreater(len(targets), 0)
        self.assertIn("joyful_social", targets)

class TestOptimizationIntegration(unittest.TestCase):
    """Test integration between optimization components"""
    
    def test_target_to_dict_roundtrip(self):
        """Test that targets can be serialized and deserialized"""
        target_manager = TargetManager()
        original = target_manager.get_target("joyful_social")
        
        # Convert to dict and back
        target_dict = original.to_dict()
        reconstructed = BehavioralTarget.from_dict(target_dict)
        
        self.assertEqual(original.name, reconstructed.name)
        self.assertEqual(original.description, reconstructed.description)
        self.assertEqual(len(original.targets), len(reconstructed.targets))
    
    def test_behavioral_metrics_serialization(self):
        """Test that behavioral metrics can be serialized"""
        from neuromod.optimization.evaluation import BehavioralMetrics
        
        metrics = BehavioralMetrics()
        metrics.emotions = {"joy": 0.8, "sadness": 0.2}
        metrics.behaviors = {"creativity": 0.6}
        metrics.metrics = {"coherence": 0.7}
        
        # Convert to dict
        metrics_dict = metrics.to_dict()
        
        self.assertIn("emotions", metrics_dict)
        self.assertIn("behaviors", metrics_dict)
        self.assertIn("metrics", metrics_dict)
        self.assertEqual(metrics_dict["emotions"]["joy"], 0.8)

if __name__ == '__main__':
    unittest.main()
