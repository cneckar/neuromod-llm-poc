#!/usr/bin/env python3
"""Simple test script to debug mixed-effects and Bayesian models"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from neuromod.testing.advanced_statistics import AdvancedStatisticalAnalyzer, create_sample_data

print("="*70)
print("Testing Advanced Statistics")
print("="*70)

# Create sample data
print("\n[*] Creating sample data...")
data = create_sample_data()
print(f"[OK] Created {len(data)} observations")
print(f"     Columns: {list(data.columns)}")
print(f"     condition dtype: {data['condition'].dtype}")
print(f"     prompt_set dtype: {data['prompt_set'].dtype}")

analyzer = AdvancedStatisticalAnalyzer()

# Test 1: Mixed-effects model
print("\n" + "="*70)
print("Test 1: Mixed-Effects Model")
print("="*70)

try:
    # Try with numeric condition
    data_me = data.copy()
    
    # Convert condition to numeric
    condition_map = {val: idx for idx, val in enumerate(sorted(data_me['condition'].unique()))}
    data_me['condition_num'] = data_me['condition'].map(condition_map)
    
    # Ensure prompt_set is string
    data_me['prompt_set'] = data_me['prompt_set'].astype(str)
    
    print(f"[*] Data prepared:")
    print(f"     condition_num dtype: {data_me['condition_num'].dtype}")
    print(f"     prompt_set dtype: {data_me['prompt_set'].dtype}")
    print(f"     Unique prompt_sets: {data_me['prompt_set'].nunique()}")
    
    formula = "score ~ condition_num + (1|prompt_set)"
    print(f"[*] Formula: {formula}")
    
    me_result = analyzer.fit_mixed_effects_model(
        data=data_me,
        formula=formula,
        group_var="prompt_set",
        model_name="test_mixed_effects"
    )
    
    print(f"[OK] Mixed-effects model fitted!")
    print(f"     AIC: {me_result.aic:.3f}")
    print(f"     BIC: {me_result.bic:.3f}")
    print(f"     Fixed effects: {me_result.fixed_effects}")
    
except Exception as e:
    print(f"[ERROR] Mixed-effects model failed: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Bayesian model
print("\n" + "="*70)
print("Test 2: Bayesian Hierarchical Model")
print("="*70)

try:
    # Check if Bayesian is available
    from neuromod.testing.advanced_statistics import BAYESIAN_AVAILABLE
    print(f"[*] BAYESIAN_AVAILABLE: {BAYESIAN_AVAILABLE}")
    
    if not BAYESIAN_AVAILABLE:
        print("[WARN] Bayesian libraries not available")
    else:
        # Prepare data
        data_bayes = data.copy()
        
        # Convert condition to numeric
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        data_bayes['condition_encoded'] = le.fit_transform(data_bayes['condition'])
        
        print(f"[*] Data prepared:")
        print(f"     condition_encoded dtype: {data_bayes['condition_encoded'].dtype}")
        print(f"     prompt_set dtype: {data_bayes['prompt_set'].dtype}")
        
        bayes_result = analyzer.fit_bayesian_hierarchical_model(
            data=data_bayes,
            y_var="score",
            x_vars=["condition_encoded"],
            group_var="prompt_set",
            model_name="test_bayesian"
        )
        
        if bayes_result:
            print(f"[OK] Bayesian model fitted!")
            print(f"     WAIC: {bayes_result.waic:.3f}")
            print(f"     LOO: {bayes_result.loo:.3f}")
        else:
            print(f"[WARN] Bayesian model returned None")
            
except Exception as e:
    print(f"[ERROR] Bayesian model failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("Test Complete")
print("="*70)

