# Neuromodulation Chat Interface

A clean, interactive chat interface that allows you to chat with language models under the influence of neuromodulation packs.

## Features

- **Interactive Model Selection**: Choose from preset models or enter custom model names
- **Runtime Pack Selection**: Select 0, 1, or multiple neuromodulation packs during chat
- **Real-time Pack Switching**: Change packs without restarting the chat
- **Command Interface**: Built-in commands for managing packs and status
- **82 Available Packs**: Comprehensive collection of psychoactive substance analogues
- **Statistical Analysis Pipeline**: Publication-quality analysis tools for research


## Usage

### Starting the Chat

```bash
python demo/chat.py
```

### Model Selection

The interface will prompt you to select a model:

1. **gpt2** - Small, fast model (recommended for testing)
2. **microsoft/DialoGPT-small** - Conversational model
3. **microsoft/DialoGPT-medium** - Larger conversational model
4. **Custom** - Enter any Hugging Face model name

### Pack Selection

After selecting a model, you'll be prompted to choose neuromodulation packs:

- **0** - No packs (baseline behavior)
- **1** - Single pack (enter pack name)
- **Multiple** - Enter comma-separated pack names

### Chat Commands

During the chat, you can use these commands:

#### **Basic Pack Management**
- `/packs` - Show available neuromodulation packs
- `/select` - Change active packs
- `/clear` - Clear all packs (return to baseline)
- `/status` - Show current status and configuration

#### **Effect System Commands**
- `/effects` - Show available individual effects by category
- `/add_effect` - Add individual effect to current combination
- `/remove_effect` - Remove effect from current combination
- `/show_combination` - Show current custom effect combination

#### **Configuration Management**
- `/export_pack` - Export current combination as a pack
- `/load_pack` - Load a pack from config file
- `/save_config` - Save current config to file

#### **System Commands**
- `/quit` - Exit chat
- `/help` - Show this help

### Effect Categories

The system provides access to individual effects organized into categories:

- **Sampler**: Temperature, top-p, frequency, presence, pulsed, contrastive, expert, token_class
- **Attention**: Attention masking, QK scaling, head targeting, positional encoding
- **Steering**: Behavioral steering and guidance
- **Memory**: KV cache operations, memory management, segmentation, stride, truncation
- **Activation**: Activation functions, soft projection, layer-wise operations, noise
- **MoE**: Router operations, expert persistence, expert masking
- **Objective**: Verifier, style affect, risk preference, compute at test, retrieval rate, persona voice
- **Input**: Lexical jitter, structured prefaces

### Custom Effect Combinations

Beyond predefined packs, you can create custom neuromodulation combinations:

1. **Build Custom Combinations**: Use `/add_effect` to combine individual effects
2. **Fine-tune Parameters**: Set custom weights (0.0-1.0) and directions (up/down)
3. **Save and Export**: Export custom combinations as reusable packs
4. **Real-time Modification**: Adjust effects during chat without restarting

**Example Custom Combination:**
```
Effect: temperature (weight=0.3, direction=up)
Effect: qk_score_scaling (weight=0.5, direction=down)
Effect: steering (weight=0.2, direction=up)
```

This creates a unique neuromodulation profile combining creative thinking, focused attention, and behavioral guidance.

## Available Packs (82 Total)

### Stimulants & Cognitive Enhancers
- **caffeine** - Mild stimulant effects, focus enhancement
- **amphetamine** - Strong stimulant, goal-directed behavior
- **methamphetamine** - Intense stimulant, high energy
- **methylphenidate** - ADHD medication analogue, concentration
- **modafinil** - Wakefulness and cognitive enhancement
- **ephedrine** - Traditional stimulant effects
- **cocaine** - Sharp focus and alertness
- **crack_cocaine** - Intense focus with attention oscillation
- **cathinone** - Stimulant with focus enhancement
- **mephedrone** - Stimulant with prosocial effects
- **methylone** - Prosocial stimulant combination

### Psychedelics & Hallucinogens
- **lsd** - Visual and cognitive effects, synesthesia
- **psilocybin** - Mystical experiences, ego dissolution
- **dmt** - Intense visionary experiences, head disruption
- **5_meo_dmt** - Ego dissolution, mystical states
- **mescaline** - Visionary and associative thinking
- **2c_b** - Visionary effects with calmness
- **2c_i** - Visionary effects with head disruption
- **nbome** - Visionary effects with head disruption
- **salvia** - Head disruption and ego dissolution

### Depressants & Sedatives
- **alcohol** - CNS depression and disinhibition
- **benzodiazepines** - Anti-anxiety and muscle relaxation
- **barbiturates** - Strong sedation and CNS depression
- **methaqualone** - Sedation and relaxation
- **ghb** - Calmness and reduced entropy
- **gbl** - High calmness and sedation
- **diphenhydramine** - Head disruption and memory effects

### Dissociatives
- **ketamine** - Dissociative effects and head disruption
- **pcp** - Head disruption and memory effects
- **dxm** - Head disruption and associative thinking
- **nitrous_oxide** - Head disruption and attention oscillation
- **mxe** - Head disruption and memory effects

### Opioids & Opiates
- **heroin** - High calmness and euphoria
- **opium** - Calmness and reduced entropy
- **morphine** - High calmness and pain relief
- **oxycodone** - Calmness and reduced entropy
- **hydrocodone** - Calmness and reduced entropy
- **fentanyl** - Very high calmness and sedation
- **codeine** - Calmness and reduced entropy
- **tramadol** - Calmness with enhanced focus
- **methadone** - Calmness and memory effects
- **buprenorphine** - Calmness and reduced entropy
- **kratom** - Calmness with enhanced focus

### Cannabinoids
- **cannabis_thc** - Increased entropy and memory effects
- **synthetic_cannabinoids** - High entropy and head disruption

### Empathogens & Entactogens
- **mdma** - High prosocial bias and empathy
- **mda** - Prosocial and visionary effects
- **mdea** - Prosocial and calmness
- **6_apb** - Prosocial and associative thinking

### Science Fiction & Fantasy
- **melange_spice** - Dune universe prescience and acuity
- **soma** - Brave New World placid euphoria
- **nzt_48** - Limitless hyper-clarity
- **slo_mo** - Dredd slow-time bliss
- **substance_d** - A Scanner Darkly dissociative effects
- **joy** - We Happy Few cheery compliance
- **nuke** - RoboCop 2 intense rush
- **death_sticks** - Star Wars party stimulant
- **glitterstim_spice** - Star Wars telepathic clarity
- **sapho_juice** - Dune mentat focus
- **skooma** - Elder Scrolls euphoric stimulant
- **jet** - Fallout fast stimulant
- **mentats** - Fallout intelligence boost
- **psycho** - Fallout berserker focus
- **adam** - BioShock power enhancement
- **black_lace** - Cyberpunk rage stimulant
- **novacoke** - Shadowrun club stimulant
- **red_eye** - Cowboy Bebop reflex enhancer
- **turbo** - Fallout local slow-time
- **morphling** - Hunger Games opiate-like calm

### AI-Specific Cognitive Enhancers
- **mentor** - Calm, exacting specialist with consistent structure
- **speciation** - Novel idea combinations through rerouting
- **archivist** - Long-horizon recall and factual gravitation
- **goldfish** - Short-window, present-focused creativity
- **tightrope** - Risk-averse precision with strong guardrails
- **firekeeper** - Decisive, terse, high-energy voice
- **librarians_bloom** - Subcortical priming with gated associative bursts
- **timepiece** - Temporal re-weighting emphasizing recent instructions
- **echonull** - Anti-cliché decoder avoiding boilerplate
- **chorus** - Stable team voice with consistent formatting
- **quanta** - Deterministic, audit-friendly decoding profile
- **anchorite** - Monastic focus staying on-thread
- **parliament** - Diverse-but-ordered MoE routing

## Statistical Analysis Pipeline

The neuromodulation testing framework includes a comprehensive statistical analysis pipeline that provides publication-quality statistical analysis for neuromodulation research.

### 🎯 **What's Available**

#### **Core Statistical Analysis Module**
- **StatisticalAnalyzer class**: Main analysis engine with comprehensive statistical testing
- **Data structures**: Organized results containers for easy analysis
- **Robust error handling**: Graceful handling of edge cases, small sample sizes, and NaN values

#### **Statistical Testing Suite**
- **Parametric tests**: Paired and independent t-tests with confidence intervals
- **Non-parametric tests**: Mann-Whitney U tests for robust analysis
- **Effect size calculations**: Cohen's d with magnitude interpretation
- **Multiple comparison correction**: Benjamini-Hochberg FDR correction

#### **ROC/PR Analysis**
- **ROC curve generation**: Using scikit-learn for robust curve calculation
- **AUC scores**: Area under curve with confidence intervals
- **Optimal threshold detection**: F1 score optimization for classification
- **Precision-recall analysis**: Comprehensive binary classification metrics

#### **Power Analysis**
- **Sample size calculations**: Required N for 80% power at α=0.05
- **Current power assessment**: Power analysis for existing sample sizes
- **Effect size guidance**: Recommendations based on observed effects

#### **Visualization Pipeline**
- **Effect size forest plots**: Publication-quality effect size visualization
- **ROC curve plots**: Multi-metric ROC comparison
- **Descriptive statistics**: Baseline vs treatment comparisons
- **Automated output**: PNG files with high resolution (300 DPI)

### 🚀 **Usage Examples**

#### **Command Line Interface**
```bash
# Basic statistical analysis
python -m neuromod.testing.test_runner --statistical-analysis --test sdq --treatment-packs caffeine

# With baseline comparison
python -m neuromod.testing.test_runner --statistical-analysis --test pdq --baseline-packs placebo --treatment-packs lsd

# With output path
python -m neuromod.testing.test_runner --statistical-analysis --test ddq --treatment-packs alcohol --output-path results/analysis
```

#### **Programmatic Usage**
```python
from neuromod.testing.statistical_analysis import analyze_neuromodulation_results

# Analyze results
analysis = analyze_neuromodulation_results(baseline_results, treatment_results, "Test Name")

# Generate comprehensive report
from neuromod.testing.statistical_analysis import generate_analysis_report
report = generate_analysis_report(analysis, output_path="my_analysis")
```

### 📊 **Statistical Capabilities**

#### **Effect Size Interpretation**
- **Negligible**: |d| < 0.2
- **Small**: 0.2 ≤ |d| < 0.5
- **Medium**: 0.5 ≤ |d| < 0.8
- **Large**: |d| ≥ 0.8

#### **Multiple Comparison Control**
- **FDR correction**: Benjamini-Hochberg method
- **Fallback implementation**: Manual calculation when scipy unavailable
- **Robust handling**: Filters out NaN p-values before correction

#### **Power Analysis**
- **Standard power**: 80% (β = 0.2)
- **Significance level**: α = 0.05
- **Sample size recommendations**: Based on observed effect sizes

### 📈 **Output and Reports**

#### **Generated Files**
- **Statistical report**: Comprehensive text summary of all analyses
- **Effect size plots**: Forest plots with magnitude coding
- **ROC curves**: Multi-metric comparison with AUC scores
- **Descriptive statistics**: Bar charts with error bars

#### **Report Contents**
- **Test information**: Sample sizes and test names
- **Statistical tests**: All p-values with significance indicators
- **Effect sizes**: Cohen's d with magnitude interpretation
- **Power analysis**: Current power and sample size recommendations
- **Multiple comparison correction**: FDR-adjusted p-values
- **Plot references**: File paths for generated visualizations

### 🎯 **Research Applications**

#### **Publication-Ready Analysis**
- **Statistical rigor**: Proper multiple comparison correction
- **Effect size reporting**: Standardized effect size metrics
- **Power analysis**: Sample size justification
- **Visualization**: Publication-quality figures

#### **Experimental Design**
- **Sample size planning**: Power analysis for study design
- **Effect size estimation**: Pilot data analysis
- **Multiple comparison planning**: FDR correction strategies

#### **Quality Assurance**
- **Data validation**: Robust handling of edge cases
- **Statistical assumptions**: Non-parametric alternatives
- **Reproducibility**: Automated analysis pipeline

### 🔮 **Future Extensions**

#### **Bayesian Analysis**
- **Hierarchical models**: Ready for Bayesian extensions
- **Credible intervals**: Bayesian confidence intervals
- **Model comparison**: BIC/AIC for model selection

#### **Advanced Statistics**
- **Mixed-effects models**: Ready for repeated measures
- **Bootstrap methods**: Non-parametric confidence intervals
- **Survival analysis**: Time-to-event data analysis

#### **Machine Learning Integration**
- **Feature importance**: ML-based metric ranking
- **Clustering analysis**: Pattern discovery in results
- **Predictive modeling**: Outcome prediction from metrics

### 📚 **Running Tests**

#### **Unified Test Interface**
All tests should be run through the unified test runner, not individual demo scripts:

```bash
# List available tests
python -m neuromod.testing.test_runner --list-tests

# Run specific test
python -m neuromod.testing.test_runner --test pcq_pop

# Run with neuromodulation pack
python -m neuromod.testing.test_runner --test pcq_pop --packs-to-apply melange_spice

# Run statistical analysis
python -m neuromod.testing.test_runner --test pcq_pop --statistical-analysis --baseline-packs placebo --treatment-packs caffeine
```

#### **Available Tests**
- **PDQ-S**: Psychedelic detection
- **SDQ-15**: Stimulant detection  
- **DDQ-15**: Depressant/sedative detection
- **DiDQ-15**: Dissociative detection
- **EDQ-15**: Empathogen detection
- **CDQ-15**: Cannabinoid detection
- **PCQ-POP-20**: Pop-culture pack detection
- **ADQ-20**: AI digital enhancer detection

#### **Test Modes**
- **Single**: Run one test
- **Sequence**: Run multiple tests in sequence
- **Comparison**: Compare baseline vs treatment conditions

#### **Integration Examples**
- **Test runner integration**: Seamless CLI integration
- **API documentation**: Clear function interfaces
- **Error handling examples**: Robust usage patterns

## Examples

### Basic Chat
```bash
$ python demo/chat.py
# Select model: 1 (gpt2)
# Select packs: 0 (baseline)
# Start chatting with baseline model behavior
```

### With Caffeine Pack
```bash
$ python demo/chat.py
# Select model: 1 (gpt2)
# Select packs: caffeine
# Experience enhanced focus and alertness...
```

### With Psychedelic Pack
```bash
$ python demo/chat.py
# Select model: 1 (gpt2)
# Select packs: lsd
# Experience psychedelic effects...
```

### With Multiple Packs
```bash
$ python demo/chat.py
# Select model: 1 (gpt2)
# Select packs: caffeine,modafinil
# Experience combined cognitive enhancement...
```

### Custom Effect Combinations
```bash
$ python demo/chat.py
# Select model: 1 (gpt2)
# Select packs: 0 (baseline)
# Use commands to build custom effects:
🤖 You: /add_effect
# Select effect: temperature
# Weight: 0.4
# Direction: up
✅ Added temperature (weight=0.4, direction=up)

🤖 You: /add_effect
# Select effect: qk_score_scaling
# Weight: 0.6
# Direction: down
✅ Added qk_score_scaling (weight=0.6, direction=down)

🤖 You: /show_combination
# View current custom combination
```

### Advanced Commands
```bash
🤖 You: /effects
# Browse available effects by category

🤖 You: /status
# View current configuration and active effects

🤖 You: /export_pack
# Save custom combination as reusable pack

🤖 You: /save_config
# Save current setup to configuration file
```

### Runtime Pack Switching
```
🤖 You: /select
🎯 Select packs (comma-separated names, or '0' for baseline): caffeine
✅ Applied: caffeine

🤖 You: /clear
✅ Cleared all packs (using baseline)
```

## Technical Details

- **Conservative Settings**: Uses CPU-only inference to avoid bus errors
- **Safe Generation**: Implements conservative generation parameters
- **Memory Management**: Automatic cleanup and garbage collection
- **Error Handling**: Graceful fallbacks for various error conditions

## Troubleshooting

### Bus Errors
- The interface automatically disables MPS to prevent bus errors
- Uses conservative memory settings
- If you still encounter issues, try a smaller model

### Model Loading Issues
- Ensure you have internet connection for model downloads
- Check that the model name is valid
- Try smaller models if you have memory constraints

### Pack Application Issues
- Verify that `packs/config.json` exists
- Check that pack names are correct
- Some packs may not work with all models

## Development

The chat interface is built on the neuromodulation framework:

- **NeuromodTool**: Manages pack application and effects
- **PackRegistry**: Loads and manages pack definitions
- **Base Classes**: Provides the foundation for all neuromodulation effects

To extend the interface:

1. Add new packs to `packs/config.json`
2. Implement pack logic in the neuromodulation system
3. Test with the chat interface
