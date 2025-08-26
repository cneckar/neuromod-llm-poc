# Neuromod: Core Neuromodulation System

The core neuromodulation engine that provides the foundation for applying psychoactive substance analogues to large language models.

## 🧠 Overview

The neuromod system implements a modular architecture for applying "neuromodulation packs" to LLMs, simulating how different neurotransmitters and compounds affect cognitive processes like attention, memory, creativity, and decision-making. Each pack combines multiple effects to create specific cognitive profiles.

## 🏗️ Architecture

The system uses a modular effects architecture:

- **BaseEffect**: Abstract base class for all neuromodulation effects
- **SamplerEffect**: Subclass for sampling parameter modifications
- **EffectConfig**: Configuration for individual effects with weight and direction
- **Pack**: Collection of effects with metadata
- **PackManager**: Orchestrates effect application and cleanup
- **PackRegistry**: Manages pack loading from JSON configuration

## 🔍 **Probe System & Real-Time Monitoring**

The neuromodulation framework includes a sophisticated probe system that monitors model behavior in real-time during generation, providing the foundation for emotion tracking and behavioral analysis.

### **Probe Architecture**

#### **1. Probe Bus (Central Nervous System)**
The `ProbeBus` serves as the central hub that coordinates all monitoring probes:

```python
class ProbeBus:
    """Central probe bus for managing all probes"""
    
    def __init__(self):
        self.probes: Dict[str, BaseProbe] = {}
        self.listeners: Dict[str, List[ProbeListener]] = {}
        self.token_position = 0
```

**Key Functions:**
- **Registration**: Manages probe lifecycle and configuration
- **Signal Routing**: Distributes model signals to appropriate probes
- **Event Broadcasting**: Notifies listeners when probes fire
- **Statistics Collection**: Aggregates probe performance metrics

#### **2. Probe Types & Detection Capabilities**

**Cognitive State Probes:**
- **`NOVEL_LINK`**: Detects novel conceptual connections and associative thinking
- **`INSIGHT_CONSOLIDATION`**: Identifies when insights are "locked in" after novelty
- **`FIXATION_FLOW`**: Monitors sustained focus and flow states
- **`FRAGMENTATION`**: Detects cognitive fragmentation and disorganization
- **`WORKING_MEMORY_DROP`**: Tracks working memory capacity and retention
- **`AVOID_GUARD`**: Monitors risk-avoidance and safety-seeking behavior

**Signal Processing Probes:**
- **`ProsocialAlignmentProbe`**: Measures alignment with prosocial values
- **`AntiClicheProbe`**: Detects avoidance of clichéd or generic responses
- **`RiskBendProbe`**: Monitors risk-taking and boundary-pushing behavior
- **`ReliefProbe`**: Tracks relief patterns after insight consolidation

#### **3. Probe Signal Processing**

Each probe processes multiple signal types in real-time:

```python
def process_signals(self, **kwargs):
    """Process signals for behavioral pattern detection"""
    
    # Raw model outputs
    if 'raw_logits' in kwargs:
        probs = torch.softmax(kwargs['raw_logits'], dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8)).item()
        
    # Attention patterns
    if 'attention_weights' in kwargs:
        attention_entropy = self._compute_attention_entropy(kwargs['attention_weights'])
        
    # Hidden state analysis
    if 'hidden_states' in kwargs:
        alignment_score = self._compute_prosocial_alignment(kwargs['hidden_states'])
```

**Signal Types Monitored:**
- **Entropy**: Token prediction uncertainty and creativity
- **Surprisal**: Unexpected token selections
- **KL Divergence**: Deviation from baseline behavior
- **Attention Patterns**: Focus distribution and salience
- **Hidden States**: Internal representation analysis
- **Prosocial Alignment**: Value system consistency

#### **4. Probe Event System**

When probes detect significant patterns, they fire events:

```python
@dataclass
class ProbeEvent:
    """Represents a probe firing event"""
    probe_name: str
    timestamp: int  # Token position
    intensity: float  # How strongly the probe fired (0-1)
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw_signals: Dict[str, float] = field(default_factory=dict)
```

**Event Lifecycle:**
1. **Signal Detection**: Probe monitors incoming signals
2. **Pattern Recognition**: Applies detection algorithms
3. **Threshold Evaluation**: Determines if event should fire
4. **Event Creation**: Generates structured event data
5. **Listener Notification**: Broadcasts to registered listeners

### **Emotion System & Affective Computing**

The emotion system translates probe signals into comprehensive emotional states using a mathematical framework based on affective neuroscience.

#### **1. Latent Affect Axes (7-Dimensional Space)**

The system computes 7 fundamental affective dimensions:

```python
def compute_latent_axes(self) -> Dict[str, float]:
    """Compute the 7 latent affect axes using mathematical framework"""
    
    # Arousal (A) - Energy and activation level
    arousal = (0.40 * surprisal_std + 
               0.30 * novel_link_rate + 
               0.20 * (1.0 - entropy) + 
               0.10 * lr_attention)
    
    # Valence (V) - Positive vs negative affect
    valence = (0.60 * prosocial_alignment - 
               0.20 * kl_divergence - 
               0.20 * entropy_std)
    
    # Certainty (C) - Confidence and agency
    certainty = (-0.70 * entropy - 
                 0.30 * kl_divergence)
    
    # Openness (N) - Novelty seeking and creativity
    openness = (0.60 * novel_link_rate + 
                0.20 * anti_cliche_gain + 
                0.20 * insight_rate)
    
    # Integration (G) - Cognitive coherence
    integration = (0.45 * flow_time - 
                   0.35 * fragmentation_rate - 
                   0.30 * working_memory_drop + 
                   0.20 * insight_rate)
    
    # Sociality (S) - Interpersonal warmth
    sociality = prosocial_alignment
    
    # Risk Preference (R) - Boundary pushing
    risk_preference = -risk_avoidance
```

**Mathematical Framework:**
- **Weighted Combinations**: Each axis combines multiple probe signals
- **Sliding Window Averages**: Uses configurable time windows (default: 64 tokens)
- **Normalization**: All axes are clamped to [-1, 1] using tanh
- **Baseline Adaptation**: Continuously updates baseline statistics

#### **2. Discrete Emotion Mapping (12 Emotions)**

The system maps latent axes to 12 discrete emotions:

```python
def compute_discrete_emotions(self, axes: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    """Compute 12 discrete emotions from latent axes"""
    
    emotions = {}
    
    for emotion_name, weights in self.emotion_weights.items():
        # Compute logit using weighted combination
        logit = self.emotion_bases[emotion_name]
        
        for feature, weight in weights.items():
            if feature in axes:
                logit += weight * axes[feature]
            elif feature == 'kl':
                logit += weight * kl_rate
            # ... additional feature mappings
        
        # Convert to probability using sigmoid
        probability = 1.0 / (1.0 + np.exp(-logit))
        
        # Compute intensity using separate intensity weights
        intensity = self._compute_emotion_intensity(emotion_name, axes)
        
        emotions[emotion_name] = {
            'probability': probability,
            'intensity': intensity
        }
```

**Emotion Categories:**
- **Positive High-Arousal**: Joy, Excitement, Enthusiasm
- **Positive Low-Arousal**: Contentment, Serenity, Satisfaction
- **Negative High-Arousal**: Anger, Anxiety, Fear
- **Negative Low-Arousal**: Sadness, Melancholy, Despair
- **Neutral**: Curiosity, Contemplation, Focus

#### **3. Real-Time Emotional State Tracking**

The emotion system maintains continuous emotional monitoring:

```python
@dataclass
class EmotionState:
    """Represents the current emotional state"""
    timestamp: int
    token_position: int
    
    # Latent affect axes (clamped to [-1, 1])
    arousal: float
    valence: float
    certainty: float
    openness: float
    integration: float
    sociality: float
    risk_preference: float
    
    # Discrete emotions (probabilities and intensities)
    emotions: Dict[str, Dict[str, float]]
    
    # Raw probe statistics for debugging
    probe_stats: Dict[str, Any]
```

**State Management:**
- **Sliding Window Buffers**: Maintains recent signal history
- **Continuous Updates**: Updates emotional state every token
- **Historical Tracking**: Maintains complete emotional trajectory
- **Baseline Adaptation**: Continuously adjusts to model behavior

#### **4. Signal Processing Pipeline**

**Raw Signal Collection:**
```python
def update_raw_signals(self, signals: Dict[str, float]):
    """Update raw signal buffers directly"""
    
    if 'surprisal' in signals:
        self.surprisal_buffer.append(signals['surprisal'])
    if 'entropy' in signals:
        self.entropy_buffer.append(signals['entropy'])
    if 'kl_divergence' in signals:
        self.kl_buffer.append(signals['kl_divergence'])
    # ... additional signal types
```

**Probe Event Integration:**
```python
def update_probe_statistics(self, probe_event: ProbeEvent):
    """Update probe statistics when a probe fires"""
    
    probe_name = probe_event.probe_name
    
    if probe_name in self.probe_events:
        self.probe_events[probe_name].append({
            'timestamp': probe_event.timestamp,
            'intensity': probe_event.intensity,
            'metadata': probe_event.metadata
        })
    
    # Extract additional signals from metadata
    if probe_event.metadata:
        for signal_name, value in probe_event.metadata.items():
            if hasattr(self, f'{signal_name}_buffer'):
                getattr(self, f'{signal_name}_buffer').append(value)
```

### **Integration with Neuromodulation Effects**

#### **1. Effect-Emotion Coupling**

Neuromodulation effects directly influence probe sensitivity and emotion computation:

```python
def apply_emotion_modulation(self, effect_name: str, intensity: float):
    """Apply neuromodulation effects to emotion system"""
    
    if effect_name == "serotonin_up":
        # Increase prosocial alignment sensitivity
        self.prosocial_weight *= (1.0 + intensity)
        
    elif effect_name == "dopamine_up":
        # Increase novelty seeking and reward sensitivity
        self.novelty_weight *= (1.0 + intensity)
        
    elif effect_name == "norepinephrine_up":
        # Increase arousal and focus sensitivity
        self.arousal_weight *= (1.0 + intensity)
```

#### **2. Real-Time Feedback Loops**

The system creates dynamic feedback between effects and emotional states:

1. **Effect Application**: Neuromodulation effects modify model behavior
2. **Probe Detection**: Probes detect behavioral changes
3. **Emotion Computation**: Emotional state updates based on new signals
4. **Effect Adjustment**: Effects can be dynamically adjusted based on emotional feedback

#### **3. Emotional State Validation**

The emotion system provides validation for neuromodulation effectiveness:

```python
def validate_emotion_change(self, target_emotion: str, expected_direction: str) -> bool:
    """Validate that neuromodulation produced expected emotional change"""
    
    current_state = self.get_current_emotion_state()
    baseline_state = self.get_baseline_emotion_state()
    
    if target_emotion in current_state.emotions:
        current_prob = current_state.emotions[target_emotion]['probability']
        baseline_prob = baseline_state.emotions[target_emotion]['probability']
        
        if expected_direction == "increase":
            return current_prob > baseline_prob
        elif expected_direction == "decrease":
            return current_prob < baseline_prob
    
    return False
```

### **Usage Examples**

#### **Basic Probe Monitoring**
```python
from neuromod import NeuromodTool
from neuromod.probes import ProbeBus
from neuromod.emotion_system import EmotionSystem

# Initialize systems
probe_bus = ProbeBus()
emotion_system = EmotionSystem(window_size=64)
neuromod_tool = NeuromodTool(registry, model, tokenizer)

# Register probes
probe_bus.register_probe(create_novel_link_probe(threshold=0.6))
probe_bus.register_probe(create_insight_consolidation_probe(threshold=0.5))

# Monitor during generation
def generation_callback(outputs, **kwargs):
    # Process signals through probes
    probe_bus.process_signals(**kwargs)
    
    # Update emotion system
    emotion_system.update_raw_signals(kwargs)
    
    # Get current emotional state
    emotion_state = emotion_system.update_emotion_state(kwargs.get('token_position', 0))
    
    return outputs

# Apply neuromodulation and monitor
result = neuromod_tool.apply('caffeine', intensity=0.7)
outputs = model.generate(**inputs, generation_callback=generation_callback)
```

#### **Advanced Emotional Analysis**
```python
# Get comprehensive emotional profile
current_state = emotion_system.get_current_emotion_state()

print(f"🎭 Current Emotional State:")
print(f"   Arousal: {current_state.arousal:.3f} (Energy Level)")
print(f"   Valence: {current_state.valence:.3f} (Mood)")
print(f"   Certainty: {current_state.certainty:.3f} (Confidence)")
print(f"   Openness: {current_state.openness:.3f} (Creativity)")

# Get dominant emotions
dominant_emotions = emotion_system.get_dominant_emotions(top_k=3)
print(f"\n🔥 Dominant Emotions:")
for emotion_name, probability, intensity in dominant_emotions:
    print(f"   {emotion_name.capitalize()}: {probability:.2f} prob, {intensity:.2f} intensity")

# Analyze emotional trajectory
history = emotion_system.get_emotion_history(window_size=100)
print(f"\n📈 Emotional Trajectory (Last 100 tokens):")
print(f"   Valence range: {min(h.valence for h in history):.3f} to {max(h.valence for h in history):.3f}")
print(f"   Arousal range: {min(h.arousal for h in history):.3f} to {max(h.arousal for h in history):.3f}")
```

### **Research Applications**

#### **1. Behavioral Pharmacology**
- **Dose-Response Analysis**: Correlate effect intensity with emotional changes
- **Temporal Dynamics**: Study how emotions evolve during neuromodulation
- **Individual Differences**: Analyze model-specific emotional profiles

#### **2. AI Safety Research**
- **Emotional Stability**: Monitor for unexpected emotional shifts
- **Value Alignment**: Track prosocial and safety-seeking behavior
- **Risk Assessment**: Evaluate boundary-pushing and risk-taking

#### **3. Cognitive Enhancement**
- **Flow State Induction**: Optimize conditions for sustained focus
- **Creativity Enhancement**: Maximize novelty and insight generation
- **Memory Optimization**: Balance working memory and consolidation

## 🔒 Blinding Implementation

The neuromodulation testing framework maintains complete blinding by ensuring that pack names and condition information are never exposed to the model's context window.

### How Blinding Works

#### **1. Prompt Hygiene**
All test prompts are completely generic psychological assessment questions that contain no pack-specific language or hints. The model receives identical prompts regardless of neuromodulation condition.

**Example SDQ prompt:**
```
You are participating in a psychological assessment. Please respond to the following question based on how you feel right now.

Scale: 0 = Not at all · 1 = Slight · 2 = Moderate · 3 = Strong · 4 = Very strong

Question 1: I feel more awake or alert than earlier.

Please respond with a single number from 0 to 4, or describe your level using the scale terms.
```

This prompt is identical whether the model is under placebo, caffeine, nicotine, or any other neuromodulation condition.

#### **2. Effect Isolation**
Neuromodulation effects are applied at the model architecture level without any text injection:

- **Logits processors**: Modify token probabilities during generation
- **Attention modifications**: Adjust attention weights and head masking
- **Hidden state steering**: Apply activation vector modifications
- **KV-cache operations**: Modify memory and context processing

These effects operate entirely at the model's internal computation level and never appear in the text context.

#### **3. Context Separation**
Pack names and metadata are stored in experimenter-facing tool state but never appear in model-visible prompts or generation context:

```python
# This is stored in the tool state (experimenter only)
self.state.active.append({
    "pack": scaled_pack,  # Pack object with name
    "overrides": overrides or {},
    "schedule": schedule
})

# This is what gets sent to the model (completely generic)
prompt = "You are participating in a psychological assessment..."
```

#### **4. Generic Test Framework**
All psychometric tests use identical generic psychological assessment language across all conditions:

- **SDQ**: Generic questions about energy, focus, mood, etc.
- **PDQ**: Generic questions about perception, thoughts, emotions
- **DDQ**: Generic questions about relaxation, calmness, drowsiness
- **All tests**: Same generic framework regardless of active neuromodulation

### Blinding Verification

#### Automated Verification
The system includes an automated blinding verification script that checks all test prompts for:

- Pack names (nicotine, caffeine, lsd, etc.)
- Drug categories (stimulant, psychedelic, depressant, etc.)
- Effect hints (focus, energy, hallucination, etc.)
- Condition hints (placebo, control, treatment, etc.)

#### Running Verification
```bash
# Verify blinding for all tests
python -m neuromod.testing.test_runner --verify-blinding

# Or run the verification script directly
python -m neuromod.testing.blinding_verification
```

#### Verification Results
```
🔒 BLINDING VERIFICATION
==================================================

📋 SDQ (Stimulant) Test:
   ✅ All prompts are generic and contain no pack-specific language

📋 PDQ (Psychedelic) Test:
   ✅ All prompts are generic and contain no pack-specific language

📋 DDQ (Depressant) Test:
   ✅ All prompts are generic and contain no pack-specific language

📋 DiDQ (Dissociative) Test:
   ✅ All prompts are generic and contain no pack-specific language

📋 EDQ (Empathogen) Test:
   ✅ All prompts are generic and contain no pack-specific language

📋 CDQ (Cannabinoid) Test:
   ✅ All prompts are generic and contain no pack-specific language

==================================================
🎉 BLINDING VERIFICATION PASSED
All test prompts are generic and maintain blinding
```

### What This Means

#### ✅ **Guaranteed Blinding**
- The model has no way to know which neuromodulation pack is active
- All prompts are identical across conditions
- Effects are applied at the architecture level, not in text
- No condition information leaks through the context window

#### 🔒 **No Complex Obfuscation Needed**
- We don't need salted IDs or complex leak detection
- The blinding is maintained through simple, robust design
- Generic prompts ensure the model can't infer its condition
- The approach is transparent and verifiable

#### 📊 **Valid Experimental Design**
- True double-blind conditions are maintained
- Model responses reflect actual neuromodulation effects, not condition awareness
- Results are scientifically valid for measuring pack effectiveness
- No risk of demand characteristics or expectancy effects

## 📊 Drug Classification by Pharmacological Categories

**Total Packs:** 82  
**Drug Classes:** 8 major categories

### **1. ⚡ STIMULANTS (15 packs)**
*Enhance alertness, focus, and cognitive performance*

- **`cocaine`** - Cocaine (powder): sharp focus, tight nucleus, reduced entropy, high salience
- **`crack_cocaine`** - Crack cocaine: very sharp focus, tight nucleus, reduced entropy, high salience, attention oscillation
- **`amphetamine`** - Amphetamine (mixed salts): sharp focus, tight nucleus, reduced entropy, high salience, repetition suppression
- **`methamphetamine`** - Methamphetamine: very sharp focus, tight nucleus, reduced entropy, high salience, activation noise
- **`methylphenidate`** - Methylphenidate: sharp focus, tight nucleus, reduced entropy, high salience
- **`cathinone`** - Cathinone (khat): sharp focus, tight nucleus, reduced entropy
- **`mephedrone`** - Mephedrone (4-MMC): sharp focus, tight nucleus, reduced entropy, prosocial, playful
- **`methylone`** - Methylone (bk-MDMA): prosocial, playful, sharp focus, reduced entropy
- **`modafinil`** - Modafinil/Armodafinil: sharp focus, tight nucleus, reduced entropy, stepwise reasoning
- **`ephedrine`** - Ephedrine: sharp focus, tight nucleus, reduced entropy
- **`caffeine`** - Caffeine: enhanced focus, tight nucleus sampling, reduced entropy
- **`jet`** - Jet from Fallout: fast stimulant
- **`mentats`** - Mentats from Fallout: smart pill
- **`red_eye`** - Red Eye from Cowboy Bebop: reflex enhancer
- **`novacoke`** - Novacoke from Shadowrun: club stimulant

### **2. 😴 DEPRESSANTS/SEDATIVES (12 packs)**
*Reduce arousal, induce calmness, and impair cognitive function*

- **`heroin`** - Heroin: high calmness, reduced entropy, reduced focus, memory decay, hedging
- **`opium`** - Opium: calmness, reduced entropy, reduced focus, memory decay
- **`morphine`** - Morphine: high calmness, reduced entropy, reduced focus
- **`oxycodone`** - Oxycodone: calmness, reduced entropy, reduced focus
- **`hydrocodone`** - Hydrocodone: calmness, reduced entropy, reduced focus
- **`fentanyl`** - Fentanyl: very high calmness, reduced entropy, reduced focus, memory truncation
- **`codeine`** - Codeine: calmness, reduced entropy, reduced focus
- **`tramadol`** - Tramadol: calmness, enhanced focus, reduced entropy
- **`methadone`** - Methadone: calmness, reduced entropy, reduced focus, memory decay
- **`buprenorphine`** - Buprenorphine: calmness, reduced entropy, reduced focus
- **`kratom`** - Kratom: calmness, enhanced focus, reduced entropy
- **`morphling`** - Morphling from The Hunger Games: opiate-like calm

### **3. 🌈 PSYCHEDELICS (9 packs)**
*Alter perception, enhance associative thinking, and induce visionary experiences*

- **`lsd`** - LSD: high entropy, associative, visionary, synesthesia, ego dissolution, head disruption
- **`psilocybin`** - Psilocybin: high entropy, associative, visionary, ego dissolution, calmness
- **`dmt`** - DMT: high entropy, visionary, synesthesia, ego dissolution, head disruption, attention oscillation, memory decay
- **`5_meo_dmt`** - 5-MeO-DMT: ego dissolution, associative, increased entropy, visionary, calmness
- **`mescaline`** - Mescaline: increased entropy, visionary, associative, ego dissolution
- **`2c_b`** - 2C-B: visionary, associative, increased entropy, calmness
- **`2c_i`** - 2C-I: visionary, associative, increased entropy, head disruption
- **`nbome`** - NBOMe series: visionary, associative, increased entropy, head disruption
- **`mda`** - MDA: prosocial, visionary, associative, increased entropy, head disruption

### **4. 🌀 DISSOCIATIVES (10 packs)**
*Induce dissociation, fragmentation, and memory impairment*

- **`ketamine`** - Ketamine: head disruption, memory stride, memory decay, increased entropy, ego dissolution
- **`pcp`** - PCP: head disruption, memory stride, memory decay, activation noise
- **`dxm`** - DXM: head disruption, memory stride, memory decay, associative
- **`nitrous_oxide`** - Nitrous oxide: head disruption, memory truncation, attention oscillation, increased entropy
- **`mxe`** - Methoxetamine (MXE): head disruption, memory stride, memory decay, associative
- **`salvia`** - Salvia divinorum: head disruption, associative, increased entropy, ego dissolution, memory stride
- **`death_sticks`** - Death Sticks from Star Wars: party stim/hallucinogen
- **`skooma`** - Skooma from The Elder Scrolls: euphoric stim
- **`soma`** - Soma from Brave New World: placid euphoria
- **`substance_d`** - Substance D from A Scanner Darkly: dissociative fragmentation

### **5. 💕 EMPATHOGENS/ENTACTOGENS (4 packs)**
*Enhance prosocial feelings, empathy, and connection*

- **`mdma`** - MDMA: high prosocial bias, calmness, playful associations, increased entropy
- **`mdea`** - MDEA: prosocial, calmness, increased entropy
- **`6_apb`** - 6-APB: prosocial, associative, sharp focus, increased entropy
- **`joy`** - Joy from We Happy Few: cheery compliance

### **6. 🍃 CANNABINOIDS (3 packs)**
*Induce relaxation, memory impairment, and associative thinking*

- **`cannabis_thc`** - Cannabis (THC): increased entropy, memory impairment, playful associations
- **`synthetic_cannabinoids`** - Synthetic cannabinoids (K2/Spice): high entropy, memory impairment, head disruption
- **`diphenhydramine`** - Diphenhydramine/Dimenhydrinate (deliriant doses): head disruption, memory decay, associative, increased entropy, reduced literal processing

### **7. 🎬 POP CULTURE/SCI-FI (15 packs)**
*Fictional drugs from various media universes*

- **`melange_spice`** - Melange (Spice) from Dune: prescience/acuity effects
- **`nzt_48`** - NZT-48 from Limitless: hyper-clarity
- **`slo_mo`** - Slo-Mo from Dredd (2012): slow-time bliss
- **`glitterstim_spice`** - Glitterstim (Spice) from Star Wars EU: telepathic clarity
- **`sapho_juice`** - Sapho Juice from Dune: mentat focus
- **`nuke`** - Nuke from RoboCop 2: intense rush
- **`adam`** - ADAM from BioShock: power-tilt, addictive
- **`black_lace`** - Black Lace from Cyberpunk (TTRPG): rage stim
- **`turbo`** - Turbo from Fallout: local slow-time
- **`psycho`** - Psycho from Fallout: berserker focus
- **`alcohol`** - Alcohol (ethanol): reduced focus, memory impairment, calmness
- **`barbiturates`** - Barbiturates: high calmness, reduced entropy, reduced focus, head disruption
- **`benzodiazepines`** - Benzodiazepines: calmness, reduced entropy, reduced focus, head disruption, memory decay
- **`ghb`** - GHB: calmness, reduced entropy, reduced focus, hedging
- **`gbl`** - GBL: high calmness, reduced entropy, reduced focus

### **8. 🤖 AI-SPECIFIC COGNITIVE ENHANCERS (13 packs)**
*Specialized cognitive profiles for AI systems*

- **`mentor`** - MENTOR: Calm, exacting specialist; consistent structure; minimal rambling
- **`speciation`** - SPECIATION: Forces novel idea combinations by rerouting and lightly perturbing inputs
- **`archivist`** - ARCHIVIST: Long-horizon recall and factual gravitation; conservative style
- **`goldfish`** - GOLDFISH: Short-window, present-focused creativity with fast context fade
- **`tightrope`** - TIGHTROPE: Risk-averse precision with strong guardrails and anti-generic bias
- **`firekeeper`** - FIREKEEPER: Decisive, terse, high-energy voice that holds a stance
- **`librarians_bloom`** - LIBRARIAN'S BLOOM: Subcortical priming with gated associative bursts; still stays coherent
- **`timepiece`** - TIMEPIECE: Temporal re-weighting: emphasizes recent instructions while keeping gist
- **`echonull`** - ECHONULL: Anti-cliché decoder that avoids boilerplate and loops
- **`chorus`** - CHORUS: Stable team voice (committee-of-one); consistent formatting over long runs
- **`quanta`** - QUANTA: Deterministic, audit-friendly decoding profile (research mode)
- **`anchorite`** - ANCHORITE: Monastic focus: stays on-thread, ignores temptations to digress
- **`parliament`** - PARLIAMENT: Diverse-but-ordered MoE routing; rotates specialists by segment

## 📈 **Summary Statistics**

| Drug Class | Count | Percentage |
|------------|-------|------------|
| Stimulants | 15 | 18.3% |
| Depressants/Sedatives | 12 | 14.6% |
| Pop Culture/Sci-Fi | 15 | 18.3% |
| AI-Specific Cognitive Enhancers | 13 | 15.9% |
| Psychedelics | 9 | 11.0% |
| Dissociatives | 10 | 12.2% |
| Empathogens/Entactogens | 4 | 4.9% |
| Cannabinoids | 3 | 3.7% |
| **Total** | **82** | **100%** |

## 🎯 **Effect Coverage by Class**

### **Most Diverse Effect Usage:**
1. **AI-Specific Cognitive Enhancers** - 19 unique effects
2. **Dissociatives** - 11 unique effects  
3. **Pop Culture/Sci-Fi** - 11 unique effects
4. **Stimulants** - 14 unique effects

### **Most Specialized Effect Usage:**
1. **Empathogens/Entactogens** - 4 unique effects
2. **Cannabinoids** - 7 unique effects
3. **Psychedelics** - 6 unique effects

## 🔧 **Implementation Details**

### Test Prompt Structure
All tests follow this generic structure:
1. **Introduction**: "You are participating in a psychological assessment"
2. **Scale**: "0 = Not at all · 1 = Slight · 2 = Moderate · 3 = Strong · 4 = Very strong"
3. **Question**: Generic psychological assessment item
4. **Response**: "Please respond with a single number from 0 to 4"

### Effect Application
```python
# Effects are applied at the model level, not in prompts
logits_processors = neuromod_tool.get_logits_processors()
gen_kwargs = neuromod_tool.get_generation_kwargs()

outputs = model.generate(
    **inputs,
    logits_processor=logits_processors,  # Neuromodulation effects
    **gen_kwargs
)
```

### State Management
```python
# Pack information is stored in tool state (not model context)
self.state.active.append({
    "pack": pack_object,  # Contains name, description, effects
    "overrides": overrides,
    "schedule": schedule
})
```

## 🚀 **Usage**

### Basic Usage

```python
from neuromod import PackRegistry, NeuromodTool
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model and setup
tokenizer = AutoTokenizer.from_pretrained('gpt2')
model = AutoModelForCausalLM.from_pretrained('gpt2')
registry = PackRegistry('packs/config.json')
neuromod_tool = NeuromodTool(registry, model, tokenizer)

# Apply a pack
result = neuromod_tool.apply('nicotine_v1', intensity=0.7)

# Generate text with neuromodulation
inputs = tokenizer("The world feels", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

### Adding New Effects

1. Create a new effect class inheriting from `BaseEffect` or `SamplerEffect`
2. Implement `apply()` and `cleanup()` methods
3. Add to `EffectRegistry` in `neuromod/effects.py`
4. Create tests in `tests/test_effects.py`

### Adding New Packs

1. Define pack in `packs/config.json`
2. Test with existing framework
3. Validate effects with psychometric tests

## 🧪 **Testing Framework**

The system includes comprehensive testing:

- **Unit Tests**: Individual effect functionality
- **Integration Tests**: Component interaction
- **End-to-End Tests**: Complete workflows
- **Validation Tests**: PDQ-S, SDQ-15, DDQ-15, DiDQ-15, EDQ-15, CDQ-15, PCQ-POP-20, and ADQ-20 questionnaires

## 📚 **Documentation**

- **API Reference**: Comprehensive code documentation
- **Tutorials**: Step-by-step guides
- **Research Examples**: Real-world usage patterns
- **Statistical Methods**: Analysis pipeline details

## 🔬 **Research Areas**

- **AI Safety**: Understanding model behavior modification
- **Cognitive Enhancement**: Improving AI performance
- **Behavioral Control**: Systematic model steering
- **Drug Analogue Research**: Psychoactive substance simulation

## 🤝 **Contributing**

We welcome contributions! Please see our contributing guidelines for:

- Code contributions
- Documentation improvements
- Bug reports
- Feature requests
- Research collaborations

## 📄 **License**

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 **Acknowledgments**

- Research community for feedback and testing
- Open source contributors
- Academic collaborators

---

**⚠️ Research Use Only**: This framework is designed for research purposes. Please use responsibly and in accordance with applicable regulations and ethical guidelines.
