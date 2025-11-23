# Statistical Analysis Validation Report

**Section**: 4.7 - Statistical Analysis
**Date**: 2025-11-18 22:17:45

## Summary

- [OK] Basic statistical tests: 3 passed, 0 failed
- [FAIL] FDR correction: Not verified
- [OK] Mixed-effects models: Available
- [WARN] Bayesian models: Optional (PyMC/ArviZ not installed)
- [OK] Canonical correlation: Available
- [OK] Power analysis: Available

## Test Results

### Basic Statistical Functions

- **stimulant_detection (placebo)**:
  - Paired t-test: p=0.0012, d=0.295
  - Wilcoxon: p=0.0011, d=0.207

- **stimulant_detection (caffeine)**:
  - Paired t-test: p=0.0000, d=1.531
  - Wilcoxon: p=0.0000, d=0.790

- **cognitive_performance (caffeine)**:
  - Paired t-test: p=0.0000, d=-0.747
  - Wilcoxon: p=0.0000, d=-0.351

### FDR Correction

- Total tests: 0
- Raw significant: 0
- FDR significant: 0

### Advanced Statistics

- **Mixed-effects model**: passed
  - AIC: nan
  - BIC: nan

- **Bayesian model**: optional
- **Canonical correlation**: passed
  - Correlations: [0.9799048706356935, 0.9734608471131077, 0.9620762978624334]

### Power Analysis

- Required sample size (d=0.25, power=0.80): 126
- Power for n=126: 0.801

## Conclusion

[WARN] **Section 4.7 validation INCOMPLETE**

Some statistical functions need attention.
