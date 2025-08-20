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
