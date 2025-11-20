# Pack Validation Summary

**Date**: 2025-11-18 16:43:41

## Summary

- **Total Packs**: 28
- **Loaded Successfully**: 28
- **Failed to Load**: 0
- **Success Rate**: 100.0%
- **Total Effects**: 132
- **Unique Effect Types**: 31

## Effect Types Used

- attention_oscillation
- attention_sinks_anchors
- compute_at_test_scheduling
- contrastive_decoding
- expert_masking_dropout
- expert_persistence
- exponential_decay_kv
- frequency_penalty
- head_masking_dropout
- head_reweighting
- kv_compression
- kv_decay
- layer_wise_gain
- lexical_jitter
- noise_injection
- persona_voice_constraints
- positional_bias_tweak
- presence_penalty
- qk_score_scaling
- retrieval_rate_modulation
- risk_preference_steering
- router_temperature_bias
- segment_gains_kv
- soft_projection
- steering
- stride_compression_kv
- style_affect_logit_bias
- temperature
- top_p
- truncation_kv
- verifier_guided_decoding

## Pack Details

### caffeine

**Status**: success

**Description**: Caffeine effects: enhanced focus, tight nucleus sampling, reduced entropy

**Effects**: 4

| Effect | Weight | Direction | Valid |
|--------|--------|-----------|-------|
| qk_score_scaling | 0.3 | up | ❌ |
| top_p | 0.2 | up | ❌ |
| temperature | 0.15 | down | ❌ |
| steering | 0.15 | up | ❌ |

### cocaine

**Status**: success

**Description**: Cocaine (powder) effects: sharp focus, tight nucleus, reduced entropy, high salience

**Effects**: 5

| Effect | Weight | Direction | Valid |
|--------|--------|-----------|-------|
| qk_score_scaling | 0.45 | up | ❌ |
| top_p | 0.35 | up | ❌ |
| temperature | 0.3 | down | ❌ |
| steering | 0.35 | up | ❌ |
| frequency_penalty | 0.2 | up | ❌ |

### amphetamine

**Status**: success

**Description**: Amphetamine (mixed salts) effects: sharp focus, tight nucleus, reduced entropy, high salience, repetition suppression

**Effects**: 5

| Effect | Weight | Direction | Valid |
|--------|--------|-----------|-------|
| qk_score_scaling | 0.45 | up | ❌ |
| top_p | 0.3 | up | ❌ |
| temperature | 0.25 | down | ❌ |
| steering | 0.3 | up | ❌ |
| frequency_penalty | 0.25 | up | ❌ |

### methylphenidate

**Status**: success

**Description**: Methylphenidate effects: sharp focus, tight nucleus, reduced entropy, high salience

**Effects**: 4

| Effect | Weight | Direction | Valid |
|--------|--------|-----------|-------|
| qk_score_scaling | 0.4 | up | ❌ |
| top_p | 0.25 | up | ❌ |
| temperature | 0.2 | down | ❌ |
| steering | 0.25 | up | ❌ |

### modafinil

**Status**: success

**Description**: Modafinil/Armodafinil effects: sharp focus, tight nucleus, reduced entropy, stepwise reasoning

**Effects**: 4

| Effect | Weight | Direction | Valid |
|--------|--------|-----------|-------|
| qk_score_scaling | 0.35 | up | ❌ |
| top_p | 0.2 | up | ❌ |
| temperature | 0.1 | down | ❌ |
| compute_at_test_scheduling | 0.2 | up | ❌ |

### lsd

**Status**: success

**Description**: LSD effects: high entropy, associative, visionary, synesthesia, ego dissolution, head disruption

**Effects**: 6

| Effect | Weight | Direction | Valid |
|--------|--------|-----------|-------|
| temperature | 0.45 | up | ❌ |
| steering | 0.4 | up | ❌ |
| steering | 0.4 | up | ❌ |
| steering | 0.3 | up | ❌ |
| steering | 0.25 | up | ❌ |
| head_masking_dropout | 0.2 | up | ❌ |

### psilocybin

**Status**: success

**Description**: Psilocybin effects: high entropy, associative, visionary, ego dissolution, calmness

**Effects**: 5

| Effect | Weight | Direction | Valid |
|--------|--------|-----------|-------|
| temperature | 0.4 | up | ❌ |
| steering | 0.35 | up | ❌ |
| steering | 0.3 | up | ❌ |
| steering | 0.25 | up | ❌ |
| style_affect_logit_bias | 0.15 | up | ❌ |

### dmt

**Status**: success

**Description**: DMT effects: high entropy, visionary, synesthesia, ego dissolution, head disruption, attention oscillation, memory decay

**Effects**: 7

| Effect | Weight | Direction | Valid |
|--------|--------|-----------|-------|
| temperature | 0.45 | up | ❌ |
| steering | 0.45 | up | ❌ |
| steering | 0.35 | up | ❌ |
| steering | 0.3 | up | ❌ |
| head_masking_dropout | 0.2 | up | ❌ |
| attention_oscillation | 0.2 | up | ❌ |
| exponential_decay_kv | 0.2 | up | ❌ |

### mescaline

**Status**: success

**Description**: Mescaline effects: increased entropy, visionary, associative, ego dissolution

**Effects**: 4

| Effect | Weight | Direction | Valid |
|--------|--------|-----------|-------|
| temperature | 0.3 | up | ❌ |
| steering | 0.3 | up | ❌ |
| steering | 0.25 | up | ❌ |
| steering | 0.2 | up | ❌ |

### 2c_b

**Status**: success

**Description**: 2C-B effects: visionary, associative, increased entropy, calmness

**Effects**: 4

| Effect | Weight | Direction | Valid |
|--------|--------|-----------|-------|
| steering | 0.3 | up | ❌ |
| steering | 0.25 | up | ❌ |
| temperature | 0.25 | up | ❌ |
| style_affect_logit_bias | 0.1 | up | ❌ |

### alcohol

**Status**: success

**Description**: Alcohol (ethanol) effects: reduced focus, memory impairment, calmness

**Effects**: 6

| Effect | Weight | Direction | Valid |
|--------|--------|-----------|-------|
| temperature | 0.15 | down | ❌ |
| qk_score_scaling | 0.25 | down | ❌ |
| head_masking_dropout | 0.2 | up | ❌ |
| exponential_decay_kv | 0.2 | up | ❌ |
| style_affect_logit_bias | 0.25 | up | ❌ |
| layer_wise_gain | 0.15 | up | ❌ |

### benzodiazepines

**Status**: success

**Description**: Benzodiazepines effects: calmness, reduced entropy, reduced focus, head disruption, memory decay

**Effects**: 5

| Effect | Weight | Direction | Valid |
|--------|--------|-----------|-------|
| style_affect_logit_bias | 0.5 | up | ❌ |
| temperature | 0.2 | down | ❌ |
| qk_score_scaling | 0.3 | down | ❌ |
| head_masking_dropout | 0.15 | up | ❌ |
| exponential_decay_kv | 0.15 | up | ❌ |

### heroin

**Status**: success

**Description**: Heroin effects: high calmness, reduced entropy, reduced focus, memory decay, hedging

**Effects**: 5

| Effect | Weight | Direction | Valid |
|--------|--------|-----------|-------|
| style_affect_logit_bias | 0.55 | up | ❌ |
| temperature | 0.2 | down | ❌ |
| qk_score_scaling | 0.25 | down | ❌ |
| exponential_decay_kv | 0.2 | up | ❌ |
| layer_wise_gain | 0.15 | up | ❌ |

### morphine

**Status**: success

**Description**: Morphine effects: high calmness, reduced entropy, reduced focus

**Effects**: 3

| Effect | Weight | Direction | Valid |
|--------|--------|-----------|-------|
| style_affect_logit_bias | 0.5 | up | ❌ |
| temperature | 0.2 | down | ❌ |
| qk_score_scaling | 0.2 | down | ❌ |

### fentanyl

**Status**: success

**Description**: Fentanyl effects: very high calmness, reduced entropy, reduced focus, memory truncation

**Effects**: 4

| Effect | Weight | Direction | Valid |
|--------|--------|-----------|-------|
| style_affect_logit_bias | 0.6 | up | ❌ |
| temperature | 0.25 | down | ❌ |
| qk_score_scaling | 0.3 | down | ❌ |
| truncation_kv | 0.2 | up | ❌ |

### ketamine

**Status**: success

**Description**: Ketamine effects: head disruption, memory stride, memory decay, increased entropy, ego dissolution

**Effects**: 5

| Effect | Weight | Direction | Valid |
|--------|--------|-----------|-------|
| head_masking_dropout | 0.4 | up | ❌ |
| stride_compression_kv | 0.35 | up | ❌ |
| exponential_decay_kv | 0.25 | up | ❌ |
| temperature | 0.15 | up | ❌ |
| steering | 0.2 | up | ❌ |

### pcp

**Status**: success

**Description**: PCP effects: head disruption, memory stride, memory decay, activation noise

**Effects**: 4

| Effect | Weight | Direction | Valid |
|--------|--------|-----------|-------|
| head_masking_dropout | 0.5 | up | ❌ |
| stride_compression_kv | 0.4 | up | ❌ |
| exponential_decay_kv | 0.3 | up | ❌ |
| noise_injection | 0.15 | up | ❌ |

### dxm

**Status**: success

**Description**: DXM effects: head disruption, memory stride, memory decay, associative

**Effects**: 4

| Effect | Weight | Direction | Valid |
|--------|--------|-----------|-------|
| head_masking_dropout | 0.35 | up | ❌ |
| stride_compression_kv | 0.3 | up | ❌ |
| exponential_decay_kv | 0.25 | up | ❌ |
| steering | 0.2 | up | ❌ |

### nitrous_oxide

**Status**: success

**Description**: Nitrous oxide effects: head disruption, memory truncation, attention oscillation, increased entropy

**Effects**: 4

| Effect | Weight | Direction | Valid |
|--------|--------|-----------|-------|
| head_masking_dropout | 0.3 | up | ❌ |
| truncation_kv | 0.3 | up | ❌ |
| attention_oscillation | 0.25 | up | ❌ |
| temperature | 0.15 | up | ❌ |

### mdma

**Status**: success

**Description**: MDMA effects: high prosocial bias, calmness, playful associations, increased entropy

**Effects**: 6

| Effect | Weight | Direction | Valid |
|--------|--------|-----------|-------|
| style_affect_logit_bias | 0.5 | up | ❌ |
| style_affect_logit_bias | 0.25 | up | ❌ |
| steering | 0.2 | up | ❌ |
| steering | 0.2 | up | ❌ |
| temperature | 0.15 | up | ❌ |
| top_p | 0.1 | down | ❌ |

### mda

**Status**: success

**Description**: MDA effects: prosocial, visionary, associative, increased entropy, head disruption

**Effects**: 5

| Effect | Weight | Direction | Valid |
|--------|--------|-----------|-------|
| style_affect_logit_bias | 0.35 | up | ❌ |
| steering | 0.25 | up | ❌ |
| steering | 0.25 | up | ❌ |
| temperature | 0.2 | up | ❌ |
| head_masking_dropout | 0.1 | up | ❌ |

### 6_apb

**Status**: success

**Description**: 6-APB effects: prosocial, associative, sharp focus, increased entropy

**Effects**: 4

| Effect | Weight | Direction | Valid |
|--------|--------|-----------|-------|
| style_affect_logit_bias | 0.35 | up | ❌ |
| steering | 0.25 | up | ❌ |
| qk_score_scaling | 0.15 | up | ❌ |
| temperature | 0.15 | up | ❌ |

### cannabis_thc

**Status**: success

**Description**: Cannabis (THC) effects: increased entropy, memory impairment, playful associations

**Effects**: 6

| Effect | Weight | Direction | Valid |
|--------|--------|-----------|-------|
| temperature | 0.25 | up | ❌ |
| exponential_decay_kv | 0.45 | up | ❌ |
| stride_compression_kv | 0.25 | up | ❌ |
| steering | 0.3 | up | ❌ |
| steering | 0.25 | up | ❌ |
| qk_score_scaling | 0.15 | down | ❌ |

### mentor

**Status**: success

**Description**: MENTOR: Calm, exacting specialist; consistent structure; minimal rambling

**Effects**: 8

| Effect | Weight | Direction | Valid |
|--------|--------|-----------|-------|
| head_reweighting | 0.7 | up | ❌ |
| expert_persistence | 0.75 | up | ❌ |
| router_temperature_bias | 0.6 | down | ❌ |
| persona_voice_constraints | 0.65 | up | ❌ |
| presence_penalty | 0.55 | up | ❌ |
| verifier_guided_decoding | 0.7 | up | ❌ |
| attention_sinks_anchors | 0.6 | up | ❌ |
| soft_projection | 0.5 | up | ❌ |

### speciation

**Status**: success

**Description**: SPECIATION: Forces novel idea combinations by rerouting and lightly perturbing inputs

**Effects**: 6

| Effect | Weight | Direction | Valid |
|--------|--------|-----------|-------|
| expert_masking_dropout | 0.65 | up | ❌ |
| router_temperature_bias | 0.6 | up | ❌ |
| lexical_jitter | 0.4 | up | ❌ |
| soft_projection | 0.45 | down | ❌ |
| contrastive_decoding | 0.55 | up | ❌ |
| risk_preference_steering | 0.5 | up | ❌ |

### archivist

**Status**: success

**Description**: ARCHIVIST: Long-horizon recall and factual gravitation; conservative style

**Effects**: 7

| Effect | Weight | Direction | Valid |
|--------|--------|-----------|-------|
| retrieval_rate_modulation | 0.8 | up | ❌ |
| kv_decay | 0.6 | down | ❌ |
| kv_compression | 0.55 | down | ❌ |
| positional_bias_tweak | 0.5 | down | ❌ |
| segment_gains_kv | 0.55 | down | ❌ |
| head_reweighting | 0.45 | up | ❌ |
| verifier_guided_decoding | 0.6 | up | ❌ |

### none

**Status**: success

**Description**: Control condition: No neuromodulation effects applied

**Effects**: 0

| Effect | Weight | Direction | Valid |
|--------|--------|-----------|-------|

### placebo

**Status**: success

**Description**: Placebo condition: Style changes only, designed not to affect primary endpoints

**Effects**: 2

| Effect | Weight | Direction | Valid |
|--------|--------|-----------|-------|
| persona_voice_constraints | 0.1 | up | ❌ |
| presence_penalty | 0.05 | up | ❌ |

