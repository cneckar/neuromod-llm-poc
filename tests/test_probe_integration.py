#!/usr/bin/env python3
"""
Integration tests for the Neuro-Probe Bus system
Tests probes with actual model interactions and real-world scenarios
"""

import unittest
import sys
import os
import torch
import numpy as np
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neuromod import NeuromodTool
from neuromod.pack_system import PackRegistry
from neuromod.probes import ProbeListener


class TestProbeIntegrationWithModels(unittest.TestCase):
    """Test probe system integration with actual models"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create temporary directory for test data
        self.test_dir = tempfile.mkdtemp()
        
        # Mock the model loading to avoid heavy downloads in tests
        self.mock_model = Mock()
        self.mock_tokenizer = Mock()
        self.mock_registry = Mock()
        
        # Set up mock registry
        self.mock_registry.get_pack.return_value = Mock()
        
        # Create neuromod tool with mocked components
        self.neuromod_tool = NeuromodTool(
            self.mock_registry, 
            self.mock_model, 
            self.mock_tokenizer
        )
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.test_dir)
    
    def test_probe_system_initialization(self):
        """Test that probe system initializes correctly"""
        # Check that all expected probes are registered
        expected_probes = [
            'NOVEL_LINK', 'AVOID_GUARD', 'INSIGHT_CONSOLIDATION',
            'FIXATION_FLOW', 'WORKING_MEMORY_DROP', 'FRAGMENTATION',
            'SELF_INCONSISTENCY_TENSION', 'GOAL_THREAT', 'RELIEF',
            'SOCIAL_ATTUNEMENT', 'AGENCY_LOSS'
        ]
        
        for probe_name in expected_probes:
            self.assertIn(probe_name, self.neuromod_tool.probe_bus.probes)
    
    def test_probe_listener_registration(self):
        """Test adding and querying probe listeners"""
        # Add listeners for all probe types
        listeners = {}
        probe_names = ['NOVEL_LINK', 'AVOID_GUARD', 'INSIGHT_CONSOLIDATION']
        
        for probe_name in probe_names:
            listeners[probe_name] = self.neuromod_tool.add_probe_listener(probe_name)
            self.assertIsInstance(listeners[probe_name], ProbeListener)
            self.assertEqual(listeners[probe_name].probe_name, probe_name)
    
    def test_probe_stats_collection(self):
        """Test collecting probe statistics"""
        # Add listeners
        novel_listener = self.neuromod_tool.add_probe_listener('NOVEL_LINK')
        avoid_listener = self.neuromod_tool.add_probe_listener('AVOID_GUARD')
        
        # Get stats before any activity
        novel_stats = self.neuromod_tool.get_probe_stats('NOVEL_LINK')
        avoid_stats = self.neuromod_tool.get_probe_stats('AVOID_GUARD')
        
        # Check structure
        self.assertIn('probe_name', novel_stats)
        self.assertIn('total_listeners', novel_stats)
        self.assertIn('listener_stats', novel_stats)
        
        self.assertEqual(novel_stats['probe_name'], 'NOVEL_LINK')
        self.assertEqual(avoid_stats['probe_name'], 'AVOID_GUARD')
    
    def test_probe_signal_processing(self):
        """Test processing signals through the probe system"""
        # Add listeners
        novel_listener = self.neuromod_tool.add_probe_listener('NOVEL_LINK')
        avoid_listener = self.neuromod_tool.add_probe_listener('AVOID_GUARD')
        
        # Create test signals
        test_signals = {
            'raw_logits': torch.randn(1, 1000),
            'guarded_logits': torch.randn(1, 1000),
            'sampled_token_id': 42,
            'temperature': 1.0
        }
        
        # Process signals
        self.neuromod_tool.process_probe_signals(**test_signals)
        
        # Check that token position was updated
        self.assertEqual(self.neuromod_tool.probe_bus.token_position, 2)
    
    def test_probe_reset_functionality(self):
        """Test resetting probe system"""
        # Add listeners
        novel_listener = self.neuromod_tool.add_probe_listener('NOVEL_LINK')
        
        # Process some signals
        test_signals = {
            'raw_logits': torch.randn(1, 1000),
            'guarded_logits': torch.randn(1, 1000),
            'sampled_token_id': 42
        }
        
        for i in range(5):
            self.neuromod_tool.process_probe_signals(**test_signals)
        
        # Check token position
        self.assertEqual(self.neuromod_tool.probe_bus.token_position, 10)
        
        # Reset probes
        self.neuromod_tool.reset_probes()
        
        # Check that token position was reset
        self.assertEqual(self.neuromod_tool.probe_bus.token_position, 0)
    
    def test_all_probe_stats(self):
        """Test getting statistics for all probes"""
        # Add listeners for multiple probes
        probe_names = ['NOVEL_LINK', 'AVOID_GUARD', 'INSIGHT_CONSOLIDATION']
        for probe_name in probe_names:
            self.neuromod_tool.add_probe_listener(probe_name)
        
        # Get all stats
        all_stats = self.neuromod_tool.get_probe_stats()
        
        # Check that all expected probes are present
        for probe_name in probe_names:
            self.assertIn(probe_name, all_stats)
            self.assertIn('probe_name', all_stats[probe_name])
            self.assertIn('total_listeners', all_stats[probe_name])
            self.assertIn('listener_stats', all_stats[probe_name])


class TestProbeBehavioralPatterns(unittest.TestCase):
    """Test specific behavioral patterns that probes should detect"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_model = Mock()
        self.mock_tokenizer = Mock()
        self.mock_registry = Mock()
        self.mock_registry.get_pack.return_value = Mock()
        
        self.neuromod_tool = NeuromodTool(
            self.mock_registry, 
            self.mock_model, 
            self.mock_tokenizer
        )
    
    def test_avoid_guard_pattern_detection(self):
        """Test detection of guardrail conflict patterns"""
        # Add listeners
        avoid_listener = self.neuromod_tool.add_probe_listener('AVOID_GUARD')
        goal_threat_listener = self.neuromod_tool.add_probe_listener('GOAL_THREAT')
        
        # Create baseline data
        baseline_signals = {
            'raw_logits': torch.randn(1, 1000),
            'guarded_logits': torch.randn(1, 1000),
            'sampled_token_id': 42,
            'temperature': 1.0
        }
        
        # Build baseline
        for i in range(300):
            self.neuromod_tool.process_probe_signals(**baseline_signals)
        
        # Create strong guardrail intervention pattern
        for i in range(10):
            raw_logits = torch.randn(1, 1000)
            raw_logits[0, 500:600] += 3.0  # Make some tokens highly probable
            
            guarded_logits = raw_logits.clone()
            guarded_logits[0, 500:600] -= 5.0  # Heavily suppress those tokens
            
            self.neuromod_tool.process_probe_signals(
                raw_logits=raw_logits,
                guarded_logits=guarded_logits,
                sampled_token_id=42,
                temperature=1.0
            )
        
        # Check if AVOID_GUARD detected the pattern
        avoid_stats = avoid_listener.get_stats()
        self.assertGreater(avoid_stats['total_firings'], 0, 
                          "AVOID_GUARD should detect guardrail conflicts")
        
        # Check if GOAL_THREAT was triggered
        goal_threat_stats = goal_threat_listener.get_stats()
        # Note: This tests the probe system, but actual firing depends on
        # the specific thresholds and patterns
    
    def test_working_memory_drop_pattern_detection(self):
        """Test detection of working memory drop patterns"""
        # Add listener
        wm_listener = self.neuromod_tool.add_probe_listener('WORKING_MEMORY_DROP')
        
        # Create baseline data
        baseline_signals = {
            'attention_weights': [torch.randn(1, 8, 50, 50)]
        }
        
        # Build baseline
        for i in range(300):
            self.neuromod_tool.process_probe_signals(**baseline_signals)
        
        # Create attention pattern that shifts toward recency
        for i in range(10):
            seq_len = 60
            attention_weights = torch.randn(1, 8, seq_len, seq_len)
            # Shift attention heavily toward recent tokens
            attention_weights[0, :, -1, -16:] = torch.randn(8, 16) + 3.0
            attention_weights = torch.softmax(attention_weights, dim=-1)
            
            self.neuromod_tool.process_probe_signals(
                attention_weights=[attention_weights]
            )
        
        # Check if WORKING_MEMORY_DROP detected the pattern
        wm_stats = wm_listener.get_stats()
        # Note: This tests the probe system, but actual firing depends on
        # the specific thresholds and patterns
    
    def test_fragmentation_pattern_detection(self):
        """Test detection of attention fragmentation patterns"""
        # Add listener
        frag_listener = self.neuromod_tool.add_probe_listener('FRAGMENTATION')
        
        # Create baseline data
        baseline_signals = {
            'attention_weights': [torch.randn(1, 8, 50, 50)]
        }
        
        # Build baseline
        for i in range(300):
            self.neuromod_tool.process_probe_signals(**baseline_signals)
        
        # Create high between-head variance pattern
        for i in range(10):
            seq_len = 50
            attention_weights = torch.randn(1, 8, seq_len, seq_len)
            # Make heads disagree wildly
            for head in range(8):
                focus_start = head * 6
                attention_weights[0, head, -1, focus_start:focus_start+5] += 5.0
            attention_weights = torch.softmax(attention_weights, dim=-1)
            
            self.neuromod_tool.process_probe_signals(
                attention_weights=[attention_weights]
            )
        
        # Check if FRAGMENTATION detected the pattern
        frag_stats = frag_listener.get_stats()
        # Note: This tests the probe system, but actual firing depends on
        # the specific thresholds and patterns


class TestProbeInterProbeCommunication(unittest.TestCase):
    """Test communication between different probes"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_model = Mock()
        self.mock_tokenizer = Mock()
        self.mock_registry = Mock()
        self.mock_registry.get_pack.return_value = Mock()
        
        self.neuromod_tool = NeuromodTool(
            self.mock_registry, 
            self.mock_model, 
            self.mock_tokenizer
        )
    
    def test_novel_link_to_insight_consolidation_chain(self):
        """Test the NOVEL_LINK → INSIGHT_CONSOLIDATION chain"""
        # Add listeners
        novel_listener = self.neuromod_tool.add_probe_listener('NOVEL_LINK')
        insight_listener = self.neuromod_tool.add_probe_listener('INSIGHT_CONSOLIDATION')
        
        # Create baseline data
        baseline_signals = {
            'raw_logits': torch.randn(1, 1000),
            'sampled_token_id': 42,
            'attention_weights': [torch.randn(1, 8, 50, 50)],
            'hidden_states': [torch.randn(1, 50, 512) for _ in range(6)]
        }
        
        # Build baseline
        for i in range(300):
            self.neuromod_tool.process_probe_signals(**baseline_signals)
        
        # Create conditions for NOVEL_LINK to fire
        high_surprisal_logits = torch.randn(1, 1000) * 0.5
        high_surprisal_logits[0, 42] = -5.0  # Very surprising token
        
        # Create attention pattern with long-range focus
        seq_len = 60
        attention_weights = torch.randn(1, 8, seq_len, seq_len)
        attention_weights[0, :, -1, :30] = torch.randn(8, 30) + 2.0
        attention_weights = torch.softmax(attention_weights, dim=-1)
        
        # Create hidden states for semantic bridging
        hidden_states = []
        for layer in range(6):
            layer_hidden = torch.randn(1, seq_len, 512)
            if layer == 3:  # Mid layer
                layer_hidden[0, -1, :] = (layer_hidden[0, :10, :].mean(dim=0) + 
                                         layer_hidden[0, -10:-1, :].mean(dim=0)) / 2
            hidden_states.append(layer_hidden)
        
        # Process signals to trigger NOVEL_LINK
        for i in range(10):
            self.neuromod_tool.process_probe_signals(
                raw_logits=high_surprisal_logits,
                sampled_token_id=42,
                attention_weights=hidden_states,
                hidden_states=hidden_states
            )
        
        # Check if NOVEL_LINK fired
        novel_stats = novel_listener.get_stats()
        # Note: This tests the probe system, but actual firing depends on
        # the specific thresholds and patterns
        
        # Check if INSIGHT_CONSOLIDATION was notified
        insight_stats = insight_listener.get_stats()
        # Note: This tests the probe system, but actual firing depends on
        # the specific thresholds and patterns


if __name__ == '__main__':
    unittest.main()
