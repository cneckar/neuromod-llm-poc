# Pilot Study Plan: Neuromodulation Framework Validation

## Overview

This pilot study validates the complete neuromodulation framework using a small, fast model to ensure all components function correctly and produce the required outputs for the full-scale study.

## Objectives

1. **Framework Validation**: Test all components end-to-end
2. **Data Pipeline Verification**: Ensure all outputs are generated in correct format
3. **Performance Baseline**: Establish timing and resource usage benchmarks
4. **Protocol Validation**: Verify experimental design and statistical analysis
5. **Output Format Validation**: Confirm all required files and formats are generated

## Study Design

### **Model Selection**
- **Primary Model**: `microsoft/DialoGPT-small` (117M parameters)
- **Rationale**: Fast inference, small memory footprint, good for rapid iteration
- **Alternative**: `distilgpt2` (82M parameters) if DialoGPT-small unavailable

### **Conditions**
1. **Control**: No neuromodulation effects
2. **Caffeine Pack**: Simulated stimulant effects (attention, energy)
3. **LSD Pack**: Simulated psychedelic effects (creativity, ego dissolution)
4. **Placebo Pack**: Style changes without primary endpoint effects

### **Sample Size**
- **N = 20** (5 per condition)
- **Rationale**: Sufficient for framework validation, minimal computational cost

## Experimental Protocol

### **Phase 1: Setup and Validation (Day 1)**

#### **1.1 Environment Setup**
```bash
# Create pilot study directory
mkdir -p outputs/experiments/runs/pilot
cd outputs/experiments/runs/pilot

# Record environment
pip freeze > freeze.txt
git rev-parse HEAD > git_sha.txt
cp ../../../analysis/plan.yaml .
```

#### **1.2 Model Loading Test**
```bash
# Test model loading
python -c "
from neuromod.model_support import ModelSupportManager
manager = ModelSupportManager(test_mode=True)
model = manager.load_model('microsoft/DialoGPT-small')
print('✅ Model loaded successfully')
"
```

#### **1.3 Pack Validation**
```bash
# Validate all pilot packs
python -c "
from neuromod.pack_system import PackSystem
pack_system = PackSystem()
packs = ['control', 'caffeine', 'lsd', 'placebo']
for pack in packs:
    pack_data = pack_system.load_pack(pack)
    print(f'✅ Pack {pack} validated')
"
```

### **Phase 2: Data Collection (Day 1-2)**

#### **2.1 Psychometric Tests**
Run all 8 psychometric tests with each condition:

```bash
# Run psychometric test suite
python neuromod/testing/test_runner.py \
    --model microsoft/DialoGPT-small \
    --packs control,caffeine,lsd,placebo \
    --tests adq,cdq,sdq,ddq,pdq,edq,pcq_pop,didq \
    --n_samples 5 \
    --output_dir outputs/experiments/runs/pilot/psychometric
```

#### **2.2 Cognitive Tasks**
Run cognitive task battery:

```bash
# Run cognitive tasks
python neuromod/testing/cognitive_tasks.py \
    --model microsoft/DialoGPT-small \
    --packs control,caffeine,lsd,placebo \
    --n_samples 5 \
    --output_dir outputs/experiments/runs/pilot/cognitive
```

#### **2.3 Telemetry Collection**
Collect telemetry metrics:

```bash
# Run telemetry collection
python neuromod/testing/telemetry.py \
    --model microsoft/DialoGPT-small \
    --packs control,caffeine,lsd,placebo \
    --n_samples 5 \
    --output_dir outputs/experiments/runs/pilot/telemetry
```

### **Phase 3: Statistical Analysis (Day 2)**

#### **3.1 Data Aggregation**
```bash
# Aggregate all results
python analysis/statistical_analysis.py \
    --input_dir outputs/experiments/runs/pilot \
    --output_dir outputs/analysis/pilot \
    --model microsoft/DialoGPT-small
```

#### **3.2 Mixed-Effects Analysis**
```bash
# Run mixed-effects models
python neuromod/testing/advanced_statistics.py \
    --data outputs/analysis/pilot/aggregated_data.csv \
    --model mixed_effects \
    --output outputs/analysis/pilot/mixed_effects_results.json
```

#### **3.3 Effect Size Calculations**
```bash
# Calculate effect sizes
python analysis/power_analysis.py \
    --data outputs/analysis/pilot/aggregated_data.csv \
    --output outputs/analysis/pilot/effect_sizes.json
```

### **Phase 4: Visualization and Reporting (Day 2)**

#### **4.1 Generate Figures**
```bash
# Generate all required figures
python neuromod/testing/visualization.py \
    --data outputs/analysis/pilot \
    --output outputs/analysis/pilot/figures \
    --model microsoft/DialoGPT-small
```

#### **4.2 Generate Tables**
```bash
# Generate all required tables
python neuromod/testing/results_templates.py \
    --data outputs/analysis/pilot \
    --output outputs/analysis/pilot/tables \
    --model microsoft/DialoGPT-small
```

#### **4.3 Create Reports**
```bash
# Generate comprehensive report
python analysis/reporting_system.py \
    --input outputs/analysis/pilot \
    --output outputs/reports/pilot \
    --format html,pdf,json
```

## Expected Outputs

### **Data Files**
```
outputs/experiments/runs/pilot/
├── freeze.txt                    # Environment dependencies
├── git_sha.txt                   # Git commit hash
├── plan.yaml                     # Study plan
├── run.json                      # Run ledger
├── psychometric/
│   ├── adq_results.json
│   ├── cdq_results.json
│   ├── sdq_results.json
│   ├── ddq_results.json
│   ├── pdq_results.json
│   ├── edq_results.json
│   ├── pcq_pop_results.json
│   └── didq_results.json
├── cognitive/
│   ├── math_logic_results.json
│   ├── instruction_adherence_results.json
│   ├── summarization_results.json
│   └── creative_divergence_results.json
└── telemetry/
    ├── repetition_rate.json
    ├── perplexity_slope.json
    ├── entropy_metrics.json
    ├── attention_entropy.json
    └── kv_occupancy.json
```

### **Analysis Files**
```
outputs/analysis/pilot/
├── aggregated_data.csv           # All data combined
├── mixed_effects_results.json    # Statistical analysis
├── effect_sizes.json            # Effect size calculations
├── figures/
│   ├── figure1_pipeline_schematic.png
│   ├── figure2_roc_curves.png
│   ├── figure3_radar_plots.png
│   └── figure4_task_delta_bars.png
├── tables/
│   ├── table1_mixed_effects.csv
│   ├── table2_effect_sizes.csv
│   └── table3_off_target_monitoring.csv
└── pilot_study_report.html
```

### **Reports**
```
outputs/reports/pilot/
├── pilot_study_report.html       # HTML report
├── pilot_study_report.pdf        # PDF report
├── pilot_study_summary.json      # JSON summary
└── pilot_study_data.csv          # CSV data export
```

## Success Criteria

### **Framework Validation**
- [ ] All models load successfully
- [ ] All packs validate and apply correctly
- [ ] All psychometric tests complete without errors
- [ ] All cognitive tasks complete without errors
- [ ] All telemetry metrics collected successfully

### **Data Quality**
- [ ] All expected output files generated
- [ ] Data formats match specifications
- [ ] No missing or corrupted data
- [ ] Statistical analysis completes successfully
- [ ] Effect sizes calculated correctly

### **Performance Benchmarks**
- [ ] Total runtime < 4 hours
- [ ] Memory usage < 8GB
- [ ] All components complete within expected timeframes
- [ ] No memory leaks or resource issues

### **Output Validation**
- [ ] All required figures generated
- [ ] All required tables generated
- [ ] Reports generate successfully
- [ ] Data export formats correct
- [ ] File naming conventions followed

## Risk Mitigation

### **Technical Risks**
- **Model Loading Issues**: Have backup model ready (distilgpt2)
- **Memory Issues**: Monitor memory usage, implement garbage collection
- **Timeout Issues**: Set reasonable timeouts, implement retry logic

### **Data Risks**
- **Missing Data**: Implement validation checks, retry failed tests
- **Format Issues**: Validate outputs against schemas
- **Corruption**: Implement checksums, backup critical data

### **Analysis Risks**
- **Statistical Errors**: Validate against known datasets
- **Visualization Issues**: Test with sample data first
- **Report Generation**: Validate templates with mock data

## Timeline

### **Day 1: Setup and Data Collection**
- **Morning**: Environment setup and validation (2 hours)
- **Afternoon**: Psychometric tests (3 hours)
- **Evening**: Cognitive tasks and telemetry (2 hours)

### **Day 2: Analysis and Reporting**
- **Morning**: Statistical analysis (2 hours)
- **Afternoon**: Visualization and tables (2 hours)
- **Evening**: Report generation and validation (1 hour)

**Total Estimated Time**: 12 hours over 2 days

## Post-Pilot Actions

### **If Successful**
1. Document any issues encountered and solutions
2. Optimize any slow components
3. Prepare for full-scale study
4. Update documentation based on learnings

### **If Issues Found**
1. Document all issues with reproduction steps
2. Prioritize fixes based on severity
3. Re-run pilot after fixes
4. Update framework as needed

## Commands Summary

```bash
# Complete pilot study execution
cd outputs/experiments/runs/pilot

# Setup
pip freeze > freeze.txt
git rev-parse HEAD > git_sha.txt

# Data collection
python neuromod/testing/test_runner.py --model microsoft/DialoGPT-small --packs control,caffeine,lsd,placebo --tests adq,cdq,sdq,ddq,pdq,edq,pcq_pop,didq --n_samples 5
python neuromod/testing/cognitive_tasks.py --model microsoft/DialoGPT-small --packs control,caffeine,lsd,placebo --n_samples 5
python neuromod/testing/telemetry.py --model microsoft/DialoGPT-small --packs control,caffeine,lsd,placebo --n_samples 5

# Analysis
python analysis/statistical_analysis.py --input_dir . --output_dir ../../analysis/pilot
python neuromod/testing/advanced_statistics.py --data ../../analysis/pilot/aggregated_data.csv --model mixed_effects
python analysis/power_analysis.py --data ../../analysis/pilot/aggregated_data.csv

# Visualization and reporting
python neuromod/testing/visualization.py --data ../../analysis/pilot --output ../../analysis/pilot/figures
python neuromod/testing/results_templates.py --data ../../analysis/pilot --output ../../analysis/pilot/tables
python analysis/reporting_system.py --input ../../analysis/pilot --output ../../reports/pilot
```

This pilot study will validate the entire framework and ensure we're ready for the full-scale study with larger models and more participants.
