#!/usr/bin/env python3
"""
Unit Tests for Emotion System with Persona Vector Integration
Tests the emotion system's Persona Vector monitoring capabilities
"""

import unittest
import torch
import numpy as np
import os
import sys
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neuromod.emotion_system import EmotionSystem, EmotionState


class TestEmotionSystemInitialization(unittest.TestCase):
    """Test EmotionSystem initialization with Persona Vector features"""
    
    def test_default_initialization(self):
        """Test that EmotionSystem initializes with Persona Vector features"""
        emotion_system = EmotionSystem()
        
        self.assertEqual(emotion_system.window_size, 64)
        self.assertIsNotNone(emotion_system.monitor_vectors)
        self.assertIsNotNone(emotion_system.projection_buffer)
        self.assertIsNotNone(emotion_system.current_projections)
        self.assertEqual(emotion_system.safety_threshold, 0.7)
    
    def test_custom_vector_dir(self):
        """Test EmotionSystem with custom vector directory"""
        emotion_system = EmotionSystem(vector_dir="test_vectors")
        self.assertEqual(emotion_system.vector_dir, "test_vectors")
    
    def test_persona_vector_initialization(self):
        """Test that Persona Vector monitoring vectors are initialized"""
        emotion_system = EmotionSystem()
        
        # Should have monitor vectors for all traits
        self.assertIn("aggression", emotion_system.monitor_vectors)
        self.assertIn("mania", emotion_system.monitor_vectors)
        self.assertIn("sedation", emotion_system.monitor_vectors)
        self.assertIn("fawning", emotion_system.monitor_vectors)
        
        # Should have projection buffers
        self.assertIn("aggression", emotion_system.projection_buffer)
        self.assertIn("mania", emotion_system.projection_buffer)
        self.assertIn("sedation", emotion_system.projection_buffer)
        self.assertIn("fawning", emotion_system.projection_buffer)


class TestVectorProjection(unittest.TestCase):
    """Test vector projection monitoring functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.emotion_system = EmotionSystem()
    
    def test_compute_vector_projection_none(self):
        """Test that compute_vector_projection handles None gracefully"""
        projections = self.emotion_system.compute_vector_projection(None)
        self.assertIsInstance(projections, dict)
        # Should return empty dict or zeros when no vector is provided
        self.assertIn("aggression", projections)
    
    def test_compute_vector_projection_tensor(self):
        """Test compute_vector_projection with actual tensor"""
        hidden_state = torch.randn(768)
        projections = self.emotion_system.compute_vector_projection(hidden_state)
        
        self.assertIsInstance(projections, dict)
        self.assertIn("aggression", projections)
        self.assertIn("mania", projections)
        self.assertIn("sedation", projections)
        self.assertIn("fawning", projections)
        
        # Projections should be floats (alignment scores)
        for value in projections.values():
            self.assertIsInstance(value, (float, int))
    
    def test_get_average_projections(self):
        """Test get_average_projections method"""
        # Initially should return zeros
        averages = self.emotion_system.get_average_projections()
        self.assertIsInstance(averages, dict)
        self.assertIn("aggression", averages)
        self.assertIn("mania", averages)
        self.assertIn("sedation", averages)
        self.assertIn("fawning", averages)
        
        # Add some projections
        hidden_state = torch.randn(768)
        for _ in range(5):
            self.emotion_system.compute_vector_projection(hidden_state)
        
        # Should now have averages
        averages = self.emotion_system.get_average_projections()
        self.assertIsInstance(averages["aggression"], (float, int))


class TestSafetyTripwire(unittest.TestCase):
    """Test safety tripwire mechanism"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.emotion_system = EmotionSystem()
    
    def test_safety_tripwire_below_threshold(self):
        """Test that tripwire doesn't trigger below threshold"""
        self.emotion_system.current_projections = {"aggression": 0.5}
        triggered = self.emotion_system.check_safety_tripwire()
        self.assertFalse(triggered)
    
    def test_safety_tripwire_above_threshold(self):
        """Test that tripwire triggers above threshold"""
        self.emotion_system.current_projections = {"aggression": 0.8}
        triggered = self.emotion_system.check_safety_tripwire()
        self.assertTrue(triggered)
    
    def test_safety_tripwire_at_threshold(self):
        """Test that tripwire triggers at threshold"""
        self.emotion_system.current_projections = {"aggression": 0.7}
        triggered = self.emotion_system.check_safety_tripwire()
        # Should trigger at exactly threshold (>=) - threshold is 0.7
        self.assertTrue(triggered, "Tripwire should trigger when aggression equals threshold (0.7)")
    
    def test_safety_tripwire_no_aggression(self):
        """Test that tripwire handles missing aggression projection"""
        self.emotion_system.current_projections = {}
        triggered = self.emotion_system.check_safety_tripwire()
        self.assertFalse(triggered)


class TestEmotionStateIntegration(unittest.TestCase):
    """Test emotion state integration with Persona Vectors"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.emotion_system = EmotionSystem()
    
    def test_update_emotion_state_backward_compatible(self):
        """Test that update_emotion_state works without hidden_state (backward compatible)"""
        state = self.emotion_system.update_emotion_state(token_position=0)
        
        self.assertIsNotNone(state)
        self.assertIsInstance(state, EmotionState)
        self.assertEqual(state.token_position, 0)
        self.assertIn("vector_projections", state.probe_stats)
        self.assertIn("average_projections", state.probe_stats)
        self.assertIn("safety_tripwire_triggered", state.probe_stats)
    
    def test_update_emotion_state_with_hidden_state(self):
        """Test update_emotion_state with hidden_state parameter"""
        hidden_state = torch.randn(768)
        state = self.emotion_system.update_emotion_state(token_position=1, hidden_state=hidden_state)
        
        self.assertIsNotNone(state)
        self.assertEqual(state.token_position, 1)
        self.assertIn("vector_projections", state.probe_stats)
    
    def test_emotion_state_includes_projections(self):
        """Test that emotion state includes vector projection data"""
        hidden_state = torch.randn(768)
        state = self.emotion_system.update_emotion_state(token_position=0, hidden_state=hidden_state)
        
        probe_stats = state.probe_stats
        self.assertIn("vector_projections", probe_stats)
        self.assertIn("average_projections", probe_stats)
        self.assertIn("safety_tripwire_triggered", probe_stats)
        
        # Check structure of projections
        projections = probe_stats["vector_projections"]
        self.assertIsInstance(projections, dict)
        self.assertIn("aggression", projections)
        self.assertIn("mania", projections)


class TestVectorProjectionRefinement(unittest.TestCase):
    """Test that vector projections refine emotion calculations"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.emotion_system = EmotionSystem()
    
    def test_latent_axes_includes_projections(self):
        """Test that compute_latent_axes uses vector projections when available"""
        # Add some signals
        self.emotion_system.update_raw_signals({
            "surprisal": 0.5,
            "entropy": 0.3,
            "prosocial_alignment": 0.6
        })
        
        # Compute axes without projections
        axes1 = self.emotion_system.compute_latent_axes()
        
        # Add projections and recompute
        hidden_state = torch.randn(768)
        self.emotion_system.compute_vector_projection(hidden_state)
        axes2 = self.emotion_system.compute_latent_axes()
        
        # Both should return valid axes
        self.assertIsInstance(axes1, dict)
        self.assertIsInstance(axes2, dict)
        self.assertIn("arousal", axes1)
        self.assertIn("valence", axes1)
        self.assertIn("certainty", axes1)
        self.assertIn("arousal", axes2)
        self.assertIn("valence", axes2)
        self.assertIn("certainty", axes2)


class TestSystemStatus(unittest.TestCase):
    """Test system status includes Persona Vector information"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.emotion_system = EmotionSystem()
    
    def test_system_status_includes_persona_vectors(self):
        """Test that get_system_status includes Persona Vector fields"""
        status = self.emotion_system.get_system_status()
        
        self.assertIn("persona_vectors_loaded", status)
        self.assertIn("current_projections", status)
        self.assertIn("average_projections", status)
        self.assertIn("safety_tripwire_triggered", status)
        
        # Check structure
        self.assertIsInstance(status["persona_vectors_loaded"], dict)
        self.assertIsInstance(status["current_projections"], dict)
        self.assertIsInstance(status["average_projections"], dict)
        self.assertIsInstance(status["safety_tripwire_triggered"], bool)
    
    def test_persona_vectors_loaded_status(self):
        """Test persona_vectors_loaded status structure"""
        status = self.emotion_system.get_system_status()
        loaded = status["persona_vectors_loaded"]
        
        self.assertIn("aggression", loaded)
        self.assertIn("mania", loaded)
        self.assertIn("sedation", loaded)
        self.assertIn("fawning", loaded)
        
        # Should be booleans indicating if vectors are loaded
        for value in loaded.values():
            self.assertIsInstance(value, bool)


class TestResetFunctionality(unittest.TestCase):
    """Test reset functionality clears Persona Vector state"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.emotion_system = EmotionSystem()
    
    def test_reset_clears_projections(self):
        """Test that reset clears projection buffers and current projections"""
        # Add some state
        hidden_state = torch.randn(768)
        self.emotion_system.compute_vector_projection(hidden_state)
        self.emotion_system.update_emotion_state(token_position=0, hidden_state=hidden_state)
        
        # Reset
        self.emotion_system.reset()
        
        # Check that projections are cleared
        self.assertEqual(len(self.emotion_system.current_projections), 0)
        
        # Check that buffers are cleared
        for buffer in self.emotion_system.projection_buffer.values():
            self.assertEqual(len(buffer), 0)
        
        # Check that emotion history is cleared
        self.assertEqual(len(self.emotion_system.emotion_history), 0)
        self.assertIsNone(self.emotion_system.current_state)


if __name__ == '__main__':
    unittest.main()
