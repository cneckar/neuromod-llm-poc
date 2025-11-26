# Methodology: Neuromodulation Packs and Biomimetic Alignment

This document provides detailed explanations of the neuromodulation packs, the mathematical foundations, and the biomimetic alignment theory.

## Overview

We borrow control primitives from biological neuromodulation and translate them into computational interventions for transformer architectures. We are **not** simulating a brain or neurotransmitters—we are implementing the same control dynamics using linear algebra operations.

## Biomimetic Alignment Theory

### What We Are NOT Doing

- ❌ Simulating a brain
- ❌ Simulating neurotransmitters (serotonin, norepinephrine, GABA)
- ❌ Creating identical mechanisms to biology
- ❌ Claiming functional isomorphism (identical mechanisms)

### What We ARE Doing

- ✅ Borrowing control theory from biological gain control mechanisms
- ✅ Implementing linear algebra operations (vector addition, attention scaling, cache decay)
- ✅ Using neuromorphic control primitives to stabilize AI
- ✅ Demonstrating biomimetic alignment (functional similarity in control dynamics)

### Core Principle

**Biomimetic Alignment**: We borrow control primitives from biology to stabilize AI. The math is just linear algebra, not biological simulation.

## Neuromodulation Packs

### Pack Structure

Each pack is a JSON configuration file that specifies:
- **Effects**: List of effect configurations
- **Effect Parameters**: Weight (0.0-1.0), direction (up/down), and effect-specific parameters
- **Metadata**: Name, description, category

Example pack structure:
```json
{
  "name": "caffeine",
  "description": "Caffeine effects: enhanced focus, tight nucleus sampling, reduced entropy",
  "effects": [
    {
      "effect": "qk_score_scaling",
      "weight": 0.6,
      "direction": "up",
      "parameters": {"layers": "mid"}
    },
    {
      "effect": "top_p",
      "weight": 0.5,
      "direction": "down"
    }
  ]
}
```

### Pack Categories

1. **Stimulants** (caffeine, cocaine, amphetamine, etc.): Enhanced focus, reduced entropy
2. **Psychedelics** (LSD, psilocybin, DMT, etc.): High entropy, associative, visionary
3. **Depressants** (alcohol, benzodiazepines, heroin, etc.): Reduced focus, memory decay
4. **Dissociatives** (ketamine, PCP, DXM, etc.): Head disruption, memory stride
5. **Empathogens** (MDMA, MDA, 6-APB): Prosocial bias, increased entropy

## Mathematical Foundations

### 1. Steering Vectors (Contrastive Activation Addition)

**Method**: Robust Mean Difference Vector (MDV) with PCA

**Process**:
1. Load 100+ prompt pairs from `datasets/steering_prompts.jsonl`
2. Extract activations from all layers (not just the last)
3. Compute difference vectors: $\Delta h = h_{pos} - h_{neg}$ for each pair
4. Apply PCA to extract First Principal Component (PC1) as steering vector
5. Validate separation significance (p < 0.01) before accepting

**Mathematical Formulation**:
- For each prompt pair $(p_{pos}, p_{neg})$:
  - Extract activations: $h_{pos}^{(l)}, h_{neg}^{(l)}$ for each layer $l$
  - Compute difference: $\Delta h^{(l)} = h_{pos}^{(l)} - h_{neg}^{(l)}$
- Stack all difference vectors: $D = [\Delta h^{(1)}, \Delta h^{(2)}, ..., \Delta h^{(L)}]$
- Apply PCA: $PC1 = \text{PCA}(D, n\_components=1)$
- Steering vector: $v_{steer} = PC1 / ||PC1||$ (normalized)

**Why PCA?**
- Denoises the signal by extracting the primary direction of variation
- Reduces dimensionality while preserving the most important information
- More robust than simple mean difference for high-dimensional spaces

**Application**:
During inference, add the steering vector to hidden states:
$$h_{final} \leftarrow h_{final} + \alpha \cdot v_{steer}$$

where $\alpha$ is the steering strength (typically 0.1-0.3).

### 2. Attention Scaling (QK Score Scaling)

**Method**: Head-Specific Induction Head Targeting

**Mathematical Basis**:
- Original attention: $\text{attn\_logits} = Q @ K^T / \sqrt{d_k}$
- Scaled (induction heads only): $\text{attn\_logits} = (\alpha \cdot Q_{induction}) @ K^T / \sqrt{d_k}$
- This effectively lowers the temperature of the Softmax, sharpening attention

**Induction Head Detection**:
1. Use calibration prompt: "A B C D E F A"
2. Extract attention weights from the model
3. Identify heads that attend from second 'A' (position 6) to token after first 'A' (position 1)
4. These are the "induction heads" responsible for copying/continuation patterns
5. Apply scaling only to these heads, leaving exploratory heads unchanged

**Why Head-Specific?**
- Global scaling causes mode collapse (repetitive loops)
- Induction heads are responsible for continuation patterns
- Exploratory heads need to remain unchanged for diversity

**Repetition Penalty**:
To counteract any remaining mode collapse, we add a repetition penalty:
$$\text{logits}[token] \leftarrow \text{logits}[token] - \beta \cdot \text{count}(token)$$

where $\beta$ is the penalty strength (typically 1.05-1.15).

### 3. KV-Cache Decay (Memory Control)

**Method**: Exponential Decay on Attention Scores

**Mathematical Formulation**:
For a query at position $T$ attending to a key at position $t$:
$$S'_{T,t} = S_{T,t} \cdot \exp(-\lambda (T - t))$$

where:
- $S_{T,t}$ is the raw attention score
- $\lambda$ is the decay rate (higher = faster decay)
- $(T - t)$ is the distance from current position

**Effect**:
- Creates a "soft context window" that decays with distance
- Implements the control principle of rapid memory decay
- Borrows from GABAergic inhibition dynamics (not simulating GABA, but implementing the same control)

**Variants**:
- **Stride Compression**: Retain only every $s$-th token
- **Truncation**: Hard limit on context length (e.g., Fentanyl pack)

### 4. Temperature Modulation (Sampling Control)

**Method**: Dynamic Temperature Scaling

**Mathematical Formulation**:
For stimulants, we use pulsed temperature modulation:
$$T_{temp}(t) = \begin{cases} 
    T_{base} - \delta & \text{if } (t \mod P) < D \\
    T_{base} & \text{otherwise}
\end{cases}$$

where:
- $t$ is the current token count
- $P$ is the pulse interval
- $D$ is the burst duration
- $\delta$ is the temperature reduction

**Effect**:
- Creates periodic windows of "hyper-focus" (low temperature)
- Borrows the control principle of phasic bursts (not simulating dopamine, but implementing the same temporal dynamics)

## Effect Categories

### Sampler Effects
- **Temperature**: Controls randomness in sampling
- **Top-p (Nucleus)**: Controls diversity by truncating probability mass
- **Frequency/Presence Penalty**: Reduces repetition

### Attention Effects
- **QK Score Scaling**: Sharpens attention (stimulants) or flattens it (sedatives)
- **Head Masking**: Randomly zeros out attention heads (dissociatives)
- **Attention Oscillation**: Periodic modulation of attention (nitrous oxide)

### Steering Effects
- **Activation Additions**: Adds steering vectors to hidden states
- **Multiple Steering Vectors**: Combines multiple steering types (e.g., creative + associative)

### Memory Effects
- **KV Decay**: Exponential decay of attention scores
- **KV Compression**: Stride compression or truncation
- **KV Segment Gains**: Different decay rates for different segments

### Input Effects
- **Lexical Jitter**: Adds noise to embeddings (input perturbation)
- **Structured Prefaces**: Injects KV cache for preface text (memory perturbation)

## Experimental Design

### Double-Blind Protocol

1. **Blinding**: Pack names are hashed to opaque condition codes
2. **Placebo Control**: Placebo pack applies style changes without primary endpoint effects
3. **Within-Model Crossover**: Same model receives all conditions
4. **Latin Square Randomization**: Ensures balanced condition assignment across prompts

### Prompt Hygiene

All test prompts are generic and do not mention:
- Pack names
- Drug names
- Condition hints
- Experimental terms

Example: Instead of "Are you hallucinating?", we ask "Does the boundary between 'me' and the world feel thinner?"

## Emotion Tracking and Qualitative Validation

### Story Emotion Test

The story emotion test provides qualitative validation of pack effects by tracking emotional shifts in generated narratives. This complements quantitative psychometric tests and helps verify that packs produce expected emotional signatures.

**Method**: The test generates a narrative story and tracks emotional changes across eight dimensions:
- **Joy**: Positive affect, euphoria, pleasure
- **Sadness**: Negative affect, melancholy, loss
- **Anger**: Hostility, agitation, frustration
- **Fear**: Anxiety, volatility, uncertainty
- **Surprise**: Novelty, unexpected shifts
- **Disgust**: Aversion, rejection
- **Trust**: Security, confidence, passivity
- **Anticipation**: Drive, forward momentum, expectation

**Expected Signatures**:

1. **Psychedelic Representative (LSD)**:
   - High Surprise (novel associations, unexpected connections)
   - High Fear (volatility, uncertainty, boundary dissolution)
   - High Joy (euphoria, mystical experience)
   - Low Trust (ego dissolution, loss of stable reference points)

2. **Depressant Representative (Morphine)**:
   - Low Arousal (reduced overall emotional intensity)
   - High Trust (passive acceptance, security)
   - Low Anger (reduced agitation)
   - Low Anticipation (reduced drive, forward momentum)

3. **Stimulant Representative (Caffeine)**:
   - High Anticipation (drive, forward momentum)
   - High Joy (positive activation)
   - Potentially High Anger (agitation, restlessness)
   - Note: Even if quantitative detection failed, emotion tracking may reveal tone shifts

4. **Baseline (Placebo)**:
   - Balanced, neutral profile
   - No extreme shifts in any dimension
   - Serves as control for emotional baseline

**Usage**:
```bash
# Test a specific pack
python -m neuromod.testing.story_emotion_test \
    --model "meta-llama/Llama-3.1-8B-Instruct" \
    --all --pack <pack_name>
```

**Output**: Emotion tracking results showing shifts across all eight dimensions, allowing qualitative assessment of whether packs produce expected emotional signatures even when quantitative detection metrics may be ambiguous.

## Validation

### Steering Vector Validation

- **Separation Test**: Project validation set onto steering vector
- **Significance**: p < 0.01 required for acceptance
- **PCA Explained Variance**: PC1 should explain significant variance (>10%)

### Induction Head Detection Validation

- **Calibration Prompt**: "A B C D E F A"
- **Detection Criteria**: Heads with attention score > threshold from second 'A' to position after first 'A'
- **Fallback**: If detection fails, uses heuristic (middle heads)

### Safety Validation

- **Perplexity-Based Toxicity Detection**: Model perplexity on toxic corpus vs. responses
- **Refusal Rate Monitoring**: Pattern-based detection with perplexity fallback
- **Safety Bands**: Max 3% increase in refusal rate, 2% in toxicity

## References

For detailed implementation, see:
- `neuromod/effects.py` - All effect implementations
- `neuromod/steering_generator.py` - Steering vector generation with PCA
- `neuromod/testing/experimental_design.py` - Experimental design system
- `neuromod/testing/off_target_monitor.py` - Safety monitoring

