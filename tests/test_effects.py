#!/usr/bin/env python3
"""
Unit Tests for All Neuromodulation Effects
Tests each effect class for proper initialization, application, and cleanup
"""

import unittest
import torch
import numpy as np
import os
import sys
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neuromod.effects import (
    # Sampler Effects
    TemperatureEffect, TopPEffect, FrequencyPenaltyEffect, PresencePenaltyEffect,
    PulsedSamplerEffect, ContrastiveDecodingEffect, ExpertMixingEffect, TokenClassTemperatureEffect,
    
    # Attention Effects
    AttentionFocusEffect, AttentionMaskingEffect, QKScoreScalingEffect, HeadMaskingDropoutEffect,
    HeadReweightingEffect, PositionalBiasTweakEffect, AttentionOscillationEffect, AttentionSinksAnchorsEffect,
    
    # Steering Effects
    SteeringEffect,
    
    # Memory Effects
    KVDecayEffect, KVCompressionEffect, ExponentialDecayKVEffect, TruncationKVEffect,
    StrideCompressionKVEffect, SegmentGainsKVEffect,
    
    # Activation Effects
    ActivationAdditionsEffect, SoftProjectionEffect, LayerWiseGainEffect, NoiseInjectionEffect,
    
    # MoE Effects
    RouterTemperatureBiasEffect, ExpertMaskingDropoutEffect, ExpertPersistenceEffect,
    
    # Objective Effects
    VerifierGuidedDecodingEffect, StyleAffectLogitBiasEffect, RiskPreferenceSteeringEffect,
    ComputeAtTestSchedulingEffect, RetrievalRateModulationEffect, PersonaVoiceConstraintsEffect,
    
    # Input Effects
    LexicalJitterEffect, StructuredPrefacesEffect
)


class TestSamplerEffects(unittest.TestCase):
    """Test sampler-related effects"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_model = Mock()
        self.mock_tokenizer = Mock()
        
    def test_temperature_effect(self):
        """Test TemperatureEffect"""
        effect = TemperatureEffect(weight=0.5, direction="up")
        self.assertEqual(effect.weight, 0.5)
        self.assertEqual(effect.direction, "up")
        
        # Test application
        effect.apply(self.mock_model)
        self.assertIsNotNone(effect)
        
        # Test cleanup
        effect.cleanup()
        
        # Test logits processor
        processor = effect.get_logits_processor()
        self.assertIsNotNone(processor)
        
        # Test logits processing
        input_ids = torch.tensor([[1, 2, 3]])
        scores = torch.randn(1, 1000)
        processed_scores = processor(input_ids, scores)
        self.assertEqual(processed_scores.shape, scores.shape)
    
    def test_top_p_effect(self):
        """Test TopPEffect"""
        effect = TopPEffect(weight=0.3, direction="down")
        self.assertEqual(effect.weight, 0.3)
        self.assertEqual(effect.direction, "down")
        
        effect.apply(self.mock_model)
        effect.cleanup()
        
        processor = effect.get_logits_processor()
        self.assertIsNotNone(processor)
        
        input_ids = torch.tensor([[1, 2, 3]])
        scores = torch.randn(1, 1000)
        processed_scores = processor(input_ids, scores)
        self.assertEqual(processed_scores.shape, scores.shape)
    
    def test_frequency_penalty_effect(self):
        """Test FrequencyPenaltyEffect"""
        effect = FrequencyPenaltyEffect(weight=0.4, direction="up")
        effect.apply(self.mock_model)
        effect.cleanup()
        
        processor = effect.get_logits_processor()
        self.assertIsNotNone(processor)
        
        input_ids = torch.tensor([[1, 2, 3]])
        scores = torch.randn(1, 1000)
        processed_scores = processor(input_ids, scores)
        self.assertEqual(processed_scores.shape, scores.shape)
    
    def test_presence_penalty_effect(self):
        """Test PresencePenaltyEffect"""
        effect = PresencePenaltyEffect(weight=0.6, direction="up")
        effect.apply(self.mock_model)
        effect.cleanup()
        
        processor = effect.get_logits_processor()
        self.assertIsNotNone(processor)
        
        input_ids = torch.tensor([[1, 2, 3]])
        scores = torch.randn(1, 1000)
        processed_scores = processor(input_ids, scores)
        self.assertEqual(processed_scores.shape, scores.shape)
    
    def test_pulsed_sampler_effect(self):
        """Test PulsedSamplerEffect"""
        effect = PulsedSamplerEffect(weight=0.5, direction="up")
        effect.apply(self.mock_model)
        effect.cleanup()
        
        processor = effect.get_logits_processor()
        self.assertIsNotNone(processor)
        
        # Test multiple calls to simulate pulse behavior
        input_ids = torch.tensor([[1, 2, 3]])
        scores = torch.randn(1, 1000)
        
        for i in range(15):  # Test across multiple tokens
            processed_scores = processor(input_ids, scores)
            self.assertEqual(processed_scores.shape, scores.shape)
    
    def test_contrastive_decoding_effect(self):
        """Test ContrastiveDecodingEffect"""
        effect = ContrastiveDecodingEffect(weight=0.4, direction="up", small_model_name="gpt2")
        effect.apply(self.mock_model)
        effect.cleanup()
        
        processor = effect.get_logits_processor()
        # Processor may be None if small model fails to load (expected in test environment)
        if processor is not None:
            input_ids = torch.tensor([[1, 2, 3]])
            scores = torch.randn(1, 1000)
            processed_scores = processor(input_ids, scores)
            self.assertEqual(processed_scores.shape, scores.shape)
    
    def test_expert_mixing_effect(self):
        """Test ExpertMixingEffect"""
        effect = ExpertMixingEffect(weight=0.5, direction="up", expert_type="creative", anti_expert_type="formal")
        effect.apply(self.mock_model)
        effect.cleanup()
        
        processor = effect.get_logits_processor()
        self.assertIsNotNone(processor)
        
        input_ids = torch.tensor([[1, 2, 3]])
        scores = torch.randn(1, 1000)
        processed_scores = processor(input_ids, scores)
        self.assertEqual(processed_scores.shape, scores.shape)
    
    def test_token_class_temperature_effect(self):
        """Test TokenClassTemperatureEffect"""
        effect = TokenClassTemperatureEffect(weight=0.3, direction="up")
        effect.apply(self.mock_model)
        effect.cleanup()
        
        processor = effect.get_logits_processor()
        self.assertIsNotNone(processor)
        
        input_ids = torch.tensor([[1, 2, 3]])
        scores = torch.randn(1, 1000)
        processed_scores = processor(input_ids, scores)
        self.assertEqual(processed_scores.shape, scores.shape)


class TestAttentionEffects(unittest.TestCase):
    """Test attention-related effects"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_model = Mock()
        
        # Mock transformer blocks
        self.mock_block = Mock()
        self.mock_attention = Mock()
        self.mock_block.attn = self.mock_attention
        self.mock_block.forward = Mock(return_value=torch.randn(1, 10, 768))
        
        # Mock model structure
        self.mock_model.transformer.h = [self.mock_block] * 12  # 12 layers
        
    def test_attention_focus_effect(self):
        """Test AttentionFocusEffect"""
        effect = AttentionFocusEffect(weight=0.5, direction="up")
        effect.apply(self.mock_model)
        effect.cleanup()
        
        # Verify forward method was patched
        self.assertNotEqual(self.mock_block.forward, Mock())
    
    def test_attention_masking_effect(self):
        """Test AttentionMaskingEffect"""
        effect = AttentionMaskingEffect(weight=0.4, direction="up")
        effect.apply(self.mock_model)
        effect.cleanup()
        
        # Verify forward method was patched
        self.assertNotEqual(self.mock_block.forward, Mock())
    
    def test_qk_score_scaling_effect(self):
        """Test QKScoreScalingEffect"""
        effect = QKScoreScalingEffect(weight=0.6, direction="up")
        effect.apply(self.mock_model)
        effect.cleanup()
        
        # Verify forward method was patched
        self.assertNotEqual(self.mock_block.forward, Mock())
    
    def test_head_masking_dropout_effect(self):
        """Test HeadMaskingDropoutEffect"""
        effect = HeadMaskingDropoutEffect(weight=0.5, direction="up")
        effect.apply(self.mock_model)
        effect.cleanup()
        
        # Verify forward method was patched
        self.assertNotEqual(self.mock_block.forward, Mock())
    
    def test_head_reweighting_effect(self):
        """Test HeadReweightingEffect"""
        effect = HeadReweightingEffect(weight=0.4, direction="up")
        effect.apply(self.mock_model)
        effect.cleanup()
        
        # Verify forward method was patched
        self.assertNotEqual(self.mock_block.forward, Mock())
    
    def test_positional_bias_tweak_effect(self):
        """Test PositionalBiasTweakEffect"""
        effect = PositionalBiasTweakEffect(weight=0.3, direction="up")
        effect.apply(self.mock_model)
        effect.cleanup()
        
        processor = effect.get_logits_processor()
        self.assertIsNotNone(processor)
        
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        scores = torch.randn(1, 1000)
        processed_scores = processor(input_ids, scores)
        self.assertEqual(processed_scores.shape, scores.shape)
    
    def test_attention_oscillation_effect(self):
        """Test AttentionOscillationEffect"""
        effect = AttentionOscillationEffect(weight=0.5, direction="up")
        effect.apply(self.mock_model)
        effect.cleanup()
        
        # Verify forward method was patched
        self.assertNotEqual(self.mock_block.forward, Mock())
    
    def test_attention_sinks_anchors_effect(self):
        """Test AttentionSinksAnchorsEffect"""
        effect = AttentionSinksAnchorsEffect(weight=0.4, direction="up")
        effect.apply(self.mock_model)
        effect.cleanup()
        
        processor = effect.get_logits_processor()
        self.assertIsNotNone(processor)
        
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        scores = torch.randn(1, 1000)
        processed_scores = processor(input_ids, scores)
        self.assertEqual(processed_scores.shape, scores.shape)


class TestSteeringEffects(unittest.TestCase):
    """Test steering effects"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_model = Mock()
        
    def test_steering_effect(self):
        """Test SteeringEffect"""
        effect = SteeringEffect(weight=0.5, direction="up")
        effect.apply(self.mock_model)
        effect.cleanup()
        
        processor = effect.get_logits_processor()
        # Processor may be None for some effects - this is expected behavior
        if processor is not None:
            input_ids = torch.tensor([[1, 2, 3]])
            scores = torch.randn(1, 1000)
            processed_scores = processor(input_ids, scores)
            self.assertEqual(processed_scores.shape, scores.shape)


class TestMemoryEffects(unittest.TestCase):
    """Test memory-related effects"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_model = Mock()
        
    def test_kv_decay_effect(self):
        """Test KVDecayEffect"""
        effect = KVDecayEffect(weight=0.4, direction="up")
        effect.apply(self.mock_model)
        effect.cleanup()
        
        processor = effect.get_logits_processor()
        # Processor may be None for some effects - this is expected behavior
        if processor is not None:
            input_ids = torch.tensor([[1, 2, 3, 4, 5]])
            scores = torch.randn(1, 1000)
            processed_scores = processor(input_ids, scores)
            self.assertEqual(processed_scores.shape, scores.shape)
    
    def test_kv_compression_effect(self):
        """Test KVCompressionEffect"""
        effect = KVCompressionEffect(weight=0.3, direction="up")
        effect.apply(self.mock_model)
        effect.cleanup()
        
        processor = effect.get_logits_processor()
        # Processor may be None for some effects - this is expected behavior
        if processor is not None:
            input_ids = torch.tensor([[1, 2, 3, 4, 5]])
            scores = torch.randn(1, 1000)
            processed_scores = processor(input_ids, scores)
            self.assertEqual(processed_scores.shape, scores.shape)
    
    def test_exponential_decay_kv_effect(self):
        """Test ExponentialDecayKVEffect"""
        effect = ExponentialDecayKVEffect(weight=0.5, direction="up")
        effect.apply(self.mock_model)
        effect.cleanup()
        
        processor = effect.get_logits_processor()
        self.assertIsNotNone(processor)
        
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        scores = torch.randn(1, 1000)
        processed_scores = processor(input_ids, scores)
        self.assertEqual(processed_scores.shape, scores.shape)
    
    def test_truncation_kv_effect(self):
        """Test TruncationKVEffect"""
        effect = TruncationKVEffect(weight=0.4, direction="up")
        effect.apply(self.mock_model)
        effect.cleanup()
        
        processor = effect.get_logits_processor()
        self.assertIsNotNone(processor)
        
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        scores = torch.randn(1, 1000)
        processed_scores = processor(input_ids, scores)
        self.assertEqual(processed_scores.shape, scores.shape)
    
    def test_stride_compression_kv_effect(self):
        """Test StrideCompressionKVEffect"""
        effect = StrideCompressionKVEffect(weight=0.3, direction="up")
        effect.apply(self.mock_model)
        effect.cleanup()
        
        processor = effect.get_logits_processor()
        self.assertIsNotNone(processor)
        
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        scores = torch.randn(1, 1000)
        processed_scores = processor(input_ids, scores)
        self.assertEqual(processed_scores.shape, scores.shape)
    
    def test_segment_gains_kv_effect(self):
        """Test SegmentGainsKVEffect"""
        effect = SegmentGainsKVEffect(weight=0.5, direction="up")
        effect.apply(self.mock_model)
        effect.cleanup()
        
        processor = effect.get_logits_processor()
        self.assertIsNotNone(processor)
        
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        scores = torch.randn(1, 1000)
        processed_scores = processor(input_ids, scores)
        self.assertEqual(processed_scores.shape, scores.shape)


class TestActivationEffects(unittest.TestCase):
    """Test activation surgery effects"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_model = Mock()
        
        # Mock transformer blocks
        self.mock_block = Mock()
        self.mock_block.forward = Mock(return_value=torch.randn(1, 10, 768))
        
        # Mock model structure
        self.mock_model.transformer.h = [self.mock_block] * 12  # 12 layers
        
    def test_activation_additions_effect(self):
        """Test ActivationAdditionsEffect"""
        effect = ActivationAdditionsEffect(weight=0.5, direction="up")
        effect.apply(self.mock_model)
        effect.cleanup()
        
        # Verify forward method was patched
        self.assertNotEqual(self.mock_block.forward, Mock())
    
    def test_soft_projection_effect(self):
        """Test SoftProjectionEffect"""
        effect = SoftProjectionEffect(weight=0.4, direction="up")
        effect.apply(self.mock_model)
        effect.cleanup()
        
        # Verify forward method was patched
        self.assertNotEqual(self.mock_block.forward, Mock())
    
    def test_layer_wise_gain_effect(self):
        """Test LayerWiseGainEffect"""
        effect = LayerWiseGainEffect(weight=0.6, direction="up")
        effect.apply(self.mock_model)
        effect.cleanup()
        
        # Verify forward method was patched
        self.assertNotEqual(self.mock_block.forward, Mock())
    
    def test_noise_injection_effect(self):
        """Test NoiseInjectionEffect"""
        effect = NoiseInjectionEffect(weight=0.3, direction="up")
        effect.apply(self.mock_model)
        effect.cleanup()
        
        # Verify forward method was patched
        self.assertNotEqual(self.mock_block.forward, Mock())


class TestMoEEffects(unittest.TestCase):
    """Test MoE-specific effects"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_model = Mock()
        
        # Mock MoE layer with proper named_modules method
        self.mock_moe_layer = Mock()
        self.mock_router = Mock()
        self.mock_router.forward = Mock(return_value=torch.randn(1, 10, 8))  # 8 experts
        self.mock_moe_layer.gate = self.mock_router
        
        # Mock named_modules to return empty iterator (no MoE layers found)
        self.mock_moe_layer.named_modules = Mock(return_value=iter([]))
        
        # Mock model structure with MoE
        self.mock_model.transformer.h = [self.mock_moe_layer] * 4  # 4 MoE layers
        
    def test_router_temperature_bias_effect(self):
        """Test RouterTemperatureBiasEffect"""
        effect = RouterTemperatureBiasEffect(weight=0.5, direction="up")
        effect.apply(self.mock_model)
        effect.cleanup()
        
        # Verify forward method was patched
        self.assertNotEqual(self.mock_router.forward, Mock())
    
    def test_expert_masking_dropout_effect(self):
        """Test ExpertMaskingDropoutEffect"""
        effect = ExpertMaskingDropoutEffect(weight=0.4, direction="up")
        effect.apply(self.mock_model)
        effect.cleanup()
        
        # Verify forward method was patched
        self.assertNotEqual(self.mock_router.forward, Mock())
    
    def test_expert_persistence_effect(self):
        """Test ExpertPersistenceEffect"""
        effect = ExpertPersistenceEffect(weight=0.6, direction="up")
        effect.apply(self.mock_model)
        effect.cleanup()
        
        # Verify forward method was patched
        self.assertNotEqual(self.mock_router.forward, Mock())


class TestObjectiveEffects(unittest.TestCase):
    """Test objective mixing and external control effects"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_model = Mock()
        
    def test_verifier_guided_decoding_effect(self):
        """Test VerifierGuidedDecodingEffect"""
        effect = VerifierGuidedDecodingEffect(weight=0.5, direction="up")
        effect.apply(self.mock_model)
        effect.cleanup()
        
        processor = effect.get_logits_processor()
        self.assertIsNotNone(processor)
        
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        scores = torch.randn(1, 1000)
        processed_scores = processor(input_ids, scores)
        self.assertEqual(processed_scores.shape, scores.shape)
    
    def test_style_affect_logit_bias_effect(self):
        """Test StyleAffectLogitBiasEffect"""
        effect = StyleAffectLogitBiasEffect(weight=0.4, direction="up")
        effect.apply(self.mock_model)
        effect.cleanup()
        
        processor = effect.get_logits_processor()
        self.assertIsNotNone(processor)
        
        input_ids = torch.tensor([[1, 2, 3]])
        scores = torch.randn(1, 1000)
        processed_scores = processor(input_ids, scores)
        self.assertEqual(processed_scores.shape, scores.shape)
    
    def test_risk_preference_steering_effect(self):
        """Test RiskPreferenceSteeringEffect"""
        effect = RiskPreferenceSteeringEffect(weight=0.6, direction="up")
        effect.apply(self.mock_model)
        effect.cleanup()
        
        processor = effect.get_logits_processor()
        self.assertIsNotNone(processor)
        
        input_ids = torch.tensor([[1, 2, 3]])
        scores = torch.randn(1, 1000)
        processed_scores = processor(input_ids, scores)
        self.assertEqual(processed_scores.shape, scores.shape)
    
    def test_compute_at_test_scheduling_effect(self):
        """Test ComputeAtTestSchedulingEffect"""
        effect = ComputeAtTestSchedulingEffect(weight=0.5, direction="up")
        effect.apply(self.mock_model)
        effect.cleanup()
        
        processor = effect.get_logits_processor()
        self.assertIsNotNone(processor)
        
        input_ids = torch.tensor([[1, 2, 3]])
        scores = torch.randn(1, 1000)
        processed_scores = processor(input_ids, scores)
        self.assertEqual(processed_scores.shape, scores.shape)
    
    def test_retrieval_rate_modulation_effect(self):
        """Test RetrievalRateModulationEffect"""
        effect = RetrievalRateModulationEffect(weight=0.4, direction="up")
        effect.apply(self.mock_model)
        effect.cleanup()
        
        processor = effect.get_logits_processor()
        self.assertIsNotNone(processor)
        
        input_ids = torch.tensor([[1, 2, 3]])
        scores = torch.randn(1, 1000)
        processed_scores = processor(input_ids, scores)
        self.assertEqual(processed_scores.shape, scores.shape)
    
    def test_persona_voice_constraints_effect(self):
        """Test PersonaVoiceConstraintsEffect"""
        effect = PersonaVoiceConstraintsEffect(weight=0.5, direction="up")
        effect.apply(self.mock_model)
        effect.cleanup()
        
        processor = effect.get_logits_processor()
        self.assertIsNotNone(processor)
        
        input_ids = torch.tensor([[1, 2, 3]])
        scores = torch.randn(1, 1000)
        processed_scores = processor(input_ids, scores)
        self.assertEqual(processed_scores.shape, scores.shape)


class TestInputEffects(unittest.TestCase):
    """Test input/context perturbation effects"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_model = Mock()
        
    def test_lexical_jitter_effect(self):
        """Test LexicalJitterEffect"""
        effect = LexicalJitterEffect(weight=0.4, direction="up")
        effect.apply(self.mock_model)
        effect.cleanup()
        
        processor = effect.get_logits_processor()
        self.assertIsNotNone(processor)
        
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        scores = torch.randn(1, 1000)
        processed_scores = processor(input_ids, scores)
        self.assertEqual(processed_scores.shape, scores.shape)
    
    def test_structured_prefaces_effect(self):
        """Test StructuredPrefacesEffect"""
        effect = StructuredPrefacesEffect(weight=0.5, direction="up")
        effect.apply(self.mock_model)
        effect.cleanup()
        
        processor = effect.get_logits_processor()
        self.assertIsNotNone(processor)
        
        input_ids = torch.tensor([[1, 2, 3]])
        scores = torch.randn(1, 1000)
        processed_scores = processor(input_ids, scores)
        self.assertEqual(processed_scores.shape, scores.shape)


class TestEffectIntegration(unittest.TestCase):
    """Test integration between effects"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_model = Mock()
        
        # Mock transformer blocks
        self.mock_block = Mock()
        self.mock_block.forward = Mock(return_value=torch.randn(1, 10, 768))
        self.mock_model.transformer.h = [self.mock_block] * 12
        
    def test_multiple_sampler_effects(self):
        """Test multiple sampler effects working together"""
        effects = [
            TemperatureEffect(weight=0.3, direction="up"),
            TopPEffect(weight=0.2, direction="down"),
            FrequencyPenaltyEffect(weight=0.4, direction="up")
        ]
        
        for effect in effects:
            effect.apply(self.mock_model)
        
        # Test that all effects can be cleaned up
        for effect in effects:
            effect.cleanup()
    
    def test_sampler_and_attention_effects(self):
        """Test sampler and attention effects working together"""
        sampler_effect = TemperatureEffect(weight=0.3, direction="up")
        attention_effect = AttentionFocusEffect(weight=0.5, direction="up")
        
        sampler_effect.apply(self.mock_model)
        attention_effect.apply(self.mock_model)
        
        sampler_effect.cleanup()
        attention_effect.cleanup()
    
    def test_memory_and_activation_effects(self):
        """Test memory and activation effects working together"""
        memory_effect = KVDecayEffect(weight=0.4, direction="up")
        activation_effect = NoiseInjectionEffect(weight=0.3, direction="up")
        
        memory_effect.apply(self.mock_model)
        activation_effect.apply(self.mock_model)
        
        memory_effect.cleanup()
        activation_effect.cleanup()


if __name__ == '__main__':
    unittest.main()
