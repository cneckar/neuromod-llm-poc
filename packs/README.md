# Neuromodulation Effects & Packs Reference

This document provides a comprehensive catalog of all implemented neuromodulation effects and their biological analogues in the Neuromod-LLM system.

## Table of Contents

1. [Effect Categories](#effect-categories)
2. [Sampler Effects](#sampler-effects)
3. [Attention Effects](#attention-effects)
4. [Memory Effects](#memory-effects)
5. [Steering Effects](#steering-effects)
6. [Activation Effects](#activation-effects)
7. [MoE/Router Effects](#moe-router-effects)
8. [Objective Effects](#objective-effects)
9. [Input Effects](#input-effects)
10. [Pack Categories](#pack-categories)
11. [Real-World Substances](#real-world-substances)
12. [Fictional Substances](#fictional-substances)
13. [Specialized Packs](#specialized-packs)

---

## Effect Categories

The neuromodulation system implements 8 main categories of effects:

### 1. **Sampler Effects** - Logits Processing
- Temperature, Top-p, Frequency/Presence Penalties
- Pulsed sampling, Contrastive decoding
- Expert mixing, Token-class temperature

### 2. **Attention Effects** - Attention Mechanism Modification
- QK score scaling, Head masking/dropout
- Head reweighting, Positional bias tweaking
- Attention oscillation, Sinks/anchors

### 3. **Memory Effects** - KV-Cache Manipulation
- Exponential decay, Truncation
- Stride compression, Segment gains

### 4. **Steering Effects** - Activation Additions
- Δh steering vectors for various cognitive states

### 5. **Activation Effects** - Representation Surgery
- Activation additions, Soft projections
- Layer-wise gain, Noise injection

### 6. **MoE/Router Effects** - Expert System Control
- Router temperature/bias, Expert masking
- Expert persistence

### 7. **Objective Effects** - External Control
- Verifier-guided decoding, Style/affect bias
- Risk preference, Compute scheduling
- Retrieval modulation, Persona constraints

### 8. **Input Effects** - Context Perturbation
- Lexical jitter, Structured prefaces

---

## Sampler Effects

### TemperatureEffect
**What it does:** Modifies the temperature parameter for logits processing
- **Up-regulation:** Higher temperature → more random/creative sampling
- **Down-regulation:** Lower temperature → more focused/deterministic sampling
- **Biological analogue:** Cortical excitability modulation (GABA/glutamate balance)

### TopPEffect
**What it does:** Controls nucleus sampling by adjusting cumulative probability threshold
- **Up-regulation:** Higher top-p → broader vocabulary, more diverse outputs
- **Down-regulation:** Lower top-p → more focused, predictable outputs
- **Biological analogue:** Attention breadth and focus mechanisms

### FrequencyPenaltyEffect
**What it does:** Penalizes frequently occurring tokens
- **Up-regulation:** Stronger penalty → more diverse vocabulary, less repetition
- **Down-regulation:** Weaker penalty → allows repetition, more predictable patterns
- **Biological analogue:** Repetition suppression in neural circuits

### PresencePenaltyEffect
**What it does:** Penalizes tokens based on their presence in context
- **Up-regulation:** Stronger penalty → avoids context repetition
- **Down-regulation:** Weaker penalty → allows context repetition
- **Biological analogue:** Context-dependent inhibition

### PulsedSamplerEffect
**What it does:** Applies periodic temperature changes during generation
- **Up-regulation:** Creates microbursts of creativity/randomness
- **Down-regulation:** Creates microbursts of focus/determinism
- **Biological analogue:** Nicotine-like microbursts, phasic dopamine release

### ContrastiveDecodingEffect
**What it does:** Uses small model logits to guide large model generation
- **Up-regulation:** Stronger contrast → more divergence from small model
- **Down-regulation:** Weaker contrast → more alignment with small model
- **Biological analogue:** Top-down cortical control mechanisms

### ExpertMixingEffect
**What it does:** Mixes expert and anti-expert attribute vectors
- **Up-regulation:** Stronger mixing → more specialized outputs
- **Down-regulation:** Weaker mixing → more balanced outputs
- **Biological analogue:** Specialized neural pathway activation

### TokenClassTemperatureEffect
**What it does:** Applies different temperatures to content vs modifier tokens
- **Up-regulation:** Content tokens get higher temperature, modifiers get lower
- **Down-regulation:** Uniform temperature across token classes
- **Biological analogue:** Selective attention to different linguistic features

---

## Attention Effects

### QKScoreScalingEffect
**What it does:** Scales Query-Key attention scores
- **Up-regulation:** Sharper attention focus, more selective processing
- **Down-regulation:** Broader attention, more distributed processing
- **Biological analogue:** Attention sharpening via acetylcholine modulation

### HeadMaskingDropoutEffect
**What it does:** Randomly masks attention heads
- **Up-regulation:** More head dropout → reduced attention capacity
- **Down-regulation:** Less head dropout → full attention capacity
- **Biological analogue:** Attention deficit, head injury effects

### HeadReweightingEffect
**What it does:** Boosts specific attention heads based on function
- **Up-regulation:** Enhances specialized head functions (stylistic, semantic, etc.)
- **Down-regulation:** Balanced head activation
- **Biological analogue:** Specialized attention pathway enhancement

### PositionalBiasTweakEffect
**What it does:** Modifies attention to recent vs historical tokens
- **Up-regulation:** Bias toward recent tokens (recency)
- **Down-regulation:** Bias toward earlier tokens (primacy)
- **Biological analogue:** Temporal attention mechanisms

### AttentionOscillationEffect
**What it does:** Applies periodic attention gain changes
- **Up-regulation:** Oscillating attention strength
- **Down-regulation:** Stable attention strength
- **Biological analogue:** Attention oscillation in certain neurological conditions

### AttentionSinksAnchorsEffect
**What it does:** Maintains stable sink tokens for top-down control
- **Up-regulation:** Stronger sink maintenance → more stable processing
- **Down-regulation:** Weaker sink maintenance → more flexible processing
- **Biological analogue:** Attentional anchors, working memory maintenance

---

## Memory Effects

### ExponentialDecayKVEffect
**What it does:** Applies exponential decay to older KV cache entries
- **Up-regulation:** Faster decay → shorter effective memory
- **Down-regulation:** Slower decay → longer effective memory
- **Biological analogue:** Memory consolidation and forgetting

### TruncationKVEffect
**What it does:** Limits KV cache to last N tokens
- **Up-regulation:** Shorter memory window
- **Down-regulation:** Longer memory window
- **Biological analogue:** Working memory capacity limits

### StrideCompressionKVEffect
**What it does:** Keeps every Nth token from older context
- **Up-regulation:** More compression → less detailed memory
- **Down-regulation:** Less compression → more detailed memory
- **Biological analogue:** Memory compression and summarization

### SegmentGainsKVEffect
**What it does:** Applies different gains to old vs new memory segments
- **Up-regulation:** Emphasizes recent segments
- **Down-regulation:** Emphasizes older segments
- **Biological analogue:** Recency vs primacy effects in memory

---

## Steering Effects

### SteeringEffect
**What it does:** Adds Δh steering vectors to hidden states
- **Types:** Associative, Visionary, Synesthesia, Ego-thin, Prosocial, Affiliative, Goal-focused, Playful, Creative, Abstract
- **Up-regulation:** Stronger steering → more pronounced cognitive state
- **Down-regulation:** Weaker steering → more neutral cognitive state
- **Biological analogue:** Neuromodulator effects (serotonin, dopamine, etc.)

---

## Activation Effects

### ActivationAdditionsEffect
**What it does:** Adds steering vectors to hidden states during generation
- **Up-regulation:** Stronger activation additions
- **Down-regulation:** Weaker activation additions
- **Biological analogue:** Direct neural stimulation effects

### SoftProjectionEffect
**What it does:** Applies soft projections to feature subspaces
- **Types:** Creative, Analytical, Emotional, Spatial, Linguistic
- **Up-regulation:** Stronger projection → more specialized processing
- **Down-regulation:** Weaker projection → more general processing
- **Biological analogue:** Specialized cortical area activation

### LayerWiseGainEffect
**What it does:** Applies gain scaling to transformer layers
- **Up-regulation:** Higher gain → amplified processing
- **Down-regulation:** Lower gain → attenuated processing
- **Biological analogue:** Cortical excitability modulation

### NoiseInjectionEffect
**What it does:** Adds Gaussian noise to activations
- **Up-regulation:** More noise → more stochastic processing
- **Down-regulation:** Less noise → more deterministic processing
- **Biological analogue:** Neural noise, stochastic resonance

---

## MoE/Router Effects

### RouterTemperatureBiasEffect
**What it does:** Modifies router temperature and bias for expert selection
- **Up-regulation:** More exploratory routing
- **Down-regulation:** More deterministic routing
- **Biological analogue:** Decision-making flexibility

### ExpertMaskingDropoutEffect
**What it does:** Masks/drops out experts during routing
- **Up-regulation:** More expert dropout → reduced specialization
- **Down-regulation:** Less expert dropout → full specialization
- **Biological analogue:** Specialized neural pathway inhibition

### ExpertPersistenceEffect
**What it does:** Maintains expert selection consistency
- **Up-regulation:** Sticky routing → consistent expert use
- **Down-regulation:** Flexible routing → variable expert use
- **Biological analogue:** Habit formation, neural pathway strengthening

---

## Objective Effects

### VerifierGuidedDecodingEffect
**What it does:** Uses verification criteria to guide generation
- **Types:** Quality, Coherence, Task-alignment
- **Up-regulation:** Stronger verification → higher quality outputs
- **Down-regulation:** Weaker verification → more creative outputs
- **Biological analogue:** Executive function, quality control

### StyleAffectLogitBiasEffect
**What it does:** Biases generation toward specific styles/affects
- **Types:** Prosocial, Sentiment, Warmth, Empathy
- **Up-regulation:** Stronger style bias
- **Down-regulation:** Weaker style bias
- **Biological analogue:** Mood and personality effects

### RiskPreferenceSteeringEffect
**What it does:** Modifies risk-taking in generation
- **Types:** Exploration vs Exploitation, Bold vs Cautious
- **Up-regulation:** More risk-taking
- **Down-regulation:** Less risk-taking
- **Biological analogue:** Risk preference modulation

### ComputeAtTestSchedulingEffect
**What it does:** Schedules computational effort during generation
- **Types:** Burst, Oscillating
- **Up-regulation:** More computational scheduling
- **Down-regulation:** Less computational scheduling
- **Biological analogue:** Cognitive resource allocation

### RetrievalRateModulationEffect
**What it does:** Modulates factual vs imaginative retrieval
- **Types:** Factual, Imaginative
- **Up-regulation:** Stronger retrieval bias
- **Down-regulation:** Weaker retrieval bias
- **Biological analogue:** Memory retrieval modulation

### PersonaVoiceConstraintsEffect
**What it does:** Constrains generation to specific personas
- **Types:** Professional, Friendly, Authoritative, Creative
- **Modes:** Stable, Adaptive, Oscillating
- **Biological analogue:** Personality and role effects

---

## Input Effects

### LexicalJitterEffect
**What it does:** Perturbs input context through synonym swaps/ablation
- **Types:** Synonym swap, Ablation, Noise injection, Reframing
- **Biological analogue:** Perceptual noise, sensory perturbation

### StructuredPrefacesEffect
**What it does:** Injects structured bias into context
- **Types:** Bias, Style, Topic, Emotion
- **Modes:** KV-only, Subtle, Persistent
- **Biological analogue:** Priming effects, contextual bias

---

## Pack Categories

The system includes 82 packs across 4 main categories:

### 1. **Real-World Substances** (50 packs)
- **Stimulants:** Caffeine, Cocaine, Amphetamine, Methamphetamine, etc.
- **Psychedelics:** LSD, Psilocybin, DMT, Mescaline, etc.
- **Depressants:** Alcohol, Benzodiazepines, Barbiturates, etc.
- **Opioids:** Heroin, Morphine, Fentanyl, etc.
- **Dissociatives:** Ketamine, PCP, DXM, etc.
- **Empathogens:** MDMA, MDA, 6-APB, etc.

### 2. **Fictional Substances** (20 packs)
- **Sci-Fi:** Melange (Dune), NZT-48 (Limitless), Soma (Brave New World)
- **Gaming:** Skooma (Elder Scrolls), Jet (Fallout), ADAM (BioShock)
- **Anime/Manga:** Red Eye (Cowboy Bebop)
- **Cyberpunk:** Black Lace, Novacoke (Shadowrun)

### 3. **Specialized Packs** (12 packs)
- **Research:** Mentor, Speciation, Archivist, Goldfish
- **Creative:** Tightrope, Firekeeper, Librarian's Bloom
- **Technical:** Timepiece, Echonull, Chorus, Quanta
- **Specialized:** Anchorite, Parliament

---

## Real-World Substances

### Stimulants
| Pack | Primary Effects | Biological Target |
|------|----------------|-------------------|
| `caffeine` | Enhanced focus, tight sampling | Adenosine receptor antagonism |
| `cocaine` | Sharp focus, high salience | Dopamine reuptake inhibition |
| `amphetamine` | Sharp focus, repetition suppression | Dopamine/norepinephrine release |
| `methamphetamine` | Very sharp focus, activation noise | Strong dopamine release |
| `methylphenidate` | Sharp focus, high salience | Dopamine reuptake inhibition |
| `modafinil` | Sharp focus, stepwise reasoning | Histamine/orexin modulation |

### Psychedelics
| Pack | Primary Effects | Biological Target |
|------|----------------|-------------------|
| `lsd` | High entropy, associative, visionary | 5-HT2A receptor agonism |
| `psilocybin` | High entropy, ego dissolution | 5-HT2A receptor agonism |
| `dmt` | High entropy, visionary, memory decay | 5-HT2A receptor agonism |
| `mescaline` | Increased entropy, visionary | 5-HT2A receptor agonism |
| `2c_b` | Visionary, associative | 5-HT2A receptor agonism |

### Depressants
| Pack | Primary Effects | Biological Target |
|------|----------------|-------------------|
| `alcohol` | Reduced focus, memory impairment | GABA enhancement, glutamate inhibition |
| `benzodiazepines` | Calmness, head disruption | GABA-A receptor modulation |
| `barbiturates` | High calmness, head disruption | GABA-A receptor modulation |
| `ghb` | Calmness, hedging | GABA-B receptor agonism |

### Opioids
| Pack | Primary Effects | Biological Target |
|------|----------------|-------------------|
| `heroin` | High calmness, memory decay | μ-opioid receptor agonism |
| `morphine` | High calmness, reduced focus | μ-opioid receptor agonism |
| `fentanyl` | Very high calmness, memory truncation | μ-opioid receptor agonism |
| `oxycodone` | Calmness, reduced focus | μ-opioid receptor agonism |

### Dissociatives
| Pack | Primary Effects | Biological Target |
|------|----------------|-------------------|
| `ketamine` | Head disruption, memory stride | NMDA receptor antagonism |
| `pcp` | Head disruption, activation noise | NMDA receptor antagonism |
| `dxm` | Head disruption, associative | NMDA receptor antagonism |
| `nitrous_oxide` | Head disruption, attention oscillation | NMDA receptor antagonism |

### Empathogens
| Pack | Primary Effects | Biological Target |
|------|----------------|-------------------|
| `mdma` | High prosocial bias, calmness | Serotonin/dopamine release |
| `mda` | Prosocial, visionary, associative | Serotonin/dopamine release |
| `6_apb` | Prosocial, associative, sharp focus | Serotonin/dopamine release |

---

## Fictional Substances

### Sci-Fi Substances
| Pack | Source | Primary Effects |
|------|--------|-----------------|
| `melange_spice` | Dune | Prescience/acuity, visionary |
| `nzt_48` | Limitless | Hyper-clarity, compute scheduling |
| `soma` | Brave New World | Placid euphoria, calmness |
| `sapho_juice` | Dune | Mentat focus, compute scheduling |

### Gaming Substances
| Pack | Source | Primary Effects |
|------|--------|-----------------|
| `skooma` | Elder Scrolls | Euphoric stim, playful |
| `jet` | Fallout | Fast stimulant, pulsed sampling |
| `mentats` | Fallout | Smart pill, compute scheduling |
| `psycho` | Fallout | Berserker focus, salient |
| `adam` | BioShock | Power-tilt, addictive |

### Cyberpunk Substances
| Pack | Source | Primary Effects |
|------|--------|-----------------|
| `black_lace` | Cyberpunk | Rage stim, salient |
| `novacoke` | Shadowrun | Club stimulant, prosocial |
| `red_eye` | Cowboy Bebop | Reflex enhancer, attention oscillation |

---

## Specialized Packs

### Research Packs
| Pack | Purpose | Primary Effects |
|------|---------|-----------------|
| `mentor` | Calm, exacting specialist | Expert persistence, verifier guidance |
| `speciation` | Novel idea combinations | Expert masking, contrastive decoding |
| `archivist` | Long-horizon recall | Retrieval modulation, KV preservation |
| `goldfish` | Short-window creativity | KV decay, present focus |

### Creative Packs
| Pack | Purpose | Primary Effects |
|------|---------|-----------------|
| `tightrope` | Risk-averse precision | Risk preference down, verifier guidance |
| `firekeeper` | Decisive, terse voice | Persona constraints, head reweighting |
| `librarians_bloom` | Subcortical priming | Structured prefaces, activation additions |

### Technical Packs
| Pack | Purpose | Primary Effects |
|------|---------|-----------------|
| `timepiece` | Temporal re-weighting | Positional bias, segment gains |
| `echonull` | Anti-cliché decoder | Contrastive decoding, presence penalty |
| `chorus` | Stable team voice | Expert persistence, persona constraints |
| `quanta` | Deterministic research mode | Verifier guidance, attention anchors |

### Specialized Packs
| Pack | Purpose | Primary Effects |
|------|---------|-----------------|
| `anchorite` | Monastic focus | Attention anchors, retrieval down |
| `parliament` | Diverse MoE routing | Router bias, expert persistence |

---

## Usage Examples

### Basic Pack Application
```python
from neuromod import NeuromodTool

# Load a pack
tool = NeuromodTool()
tool.load_pack("caffeine")

# Apply to model
tool.apply_to_model(model)
```

### Custom Effect Combination
```python
# Create custom pack
custom_pack = {
    "name": "my_custom_pack",
    "effects": [
        {"effect": "temperature", "weight": 0.3, "direction": "up"},
        {"effect": "qk_score_scaling", "weight": 0.2, "direction": "up"}
    ]
}

# Apply custom pack
tool.load_custom_pack(custom_pack)
tool.apply_to_model(model)
```

### Effect-Specific Configuration
```python
# Configure specific effect parameters
effect_config = {
    "steering_type": "associative",
    "layers": "mid",
    "weight": 0.5,
    "direction": "up"
}

tool.apply_effect("steering", effect_config)
```

---

## Biological Analogues Summary

### Neurotransmitter Systems
- **Dopamine:** Stimulants, reward processing, salience
- **Serotonin:** Psychedelics, mood, prosocial behavior
- **GABA:** Depressants, inhibition, calmness
- **Glutamate:** Excitatory effects, memory
- **Opioids:** Pain relief, euphoria, sedation

### Cognitive Effects
- **Attention:** Focus, breadth, oscillation
- **Memory:** Consolidation, decay, compression
- **Creativity:** Associative thinking, divergent processing
- **Executive Function:** Planning, inhibition, flexibility

### Behavioral Effects
- **Social:** Prosocial bias, empathy, affiliation
- **Emotional:** Mood modulation, affect bias
- **Motivational:** Reward seeking, risk preference
- **Perceptual:** Sensory modulation, synesthesia

This reference provides a comprehensive overview of the neuromodulation system's capabilities for researchers, developers, and users interested in understanding and applying these effects to language model behavior modification.
