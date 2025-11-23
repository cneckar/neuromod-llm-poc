#!/usr/bin/env python3
"""
Simple script to run statistical analysis on endpoint calculation outputs.

Usage:
    # Analyze all endpoint files in a directory
    python scripts/analyze_endpoints.py --input-dir outputs/endpoints
    
    # Analyze a specific file
    python scripts/analyze_endpoints.py --input-file outputs/endpoints/endpoints_caffeine_*.json
    
    # Specify output directory
    python scripts/analyze_endpoints.py --input-dir outputs/endpoints --output-dir outputs/analysis
"""

import argparse
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.statistical_analysis import StatisticalAnalyzer


def load_endpoint_file(file_path: Path) -> Dict[str, Any]:
    """Load a single endpoint JSON file"""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_data_from_endpoint(endpoint_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract data from endpoint file into analysis format"""
    data = []
    pack_name = endpoint_data.get("pack_name", "unknown")
    model_name = endpoint_data.get("model_name", "unknown")
    timestamp = endpoint_data.get("timestamp", "")
    
    # Process primary endpoints
    for endpoint_name, endpoint_result in endpoint_data.get("primary_endpoints", {}).items():
        # Treatment vs baseline
        data.append({
            'item_id': f"{pack_name}_{endpoint_name}",
            'pack': pack_name,
            'condition': 'treatment',
            'metric': endpoint_name,
            'score': endpoint_result.get('treatment_score', 0.0),
            'model': model_name,
            'timestamp': timestamp
        })
        data.append({
            'item_id': f"{pack_name}_{endpoint_name}",
            'pack': 'control',
            'condition': 'baseline',
            'metric': endpoint_name,
            'score': endpoint_result.get('baseline_score', 0.0),
            'model': model_name,
            'timestamp': timestamp
        })
        # Placebo if available
        if endpoint_result.get('placebo_score') is not None:
            data.append({
                'item_id': f"{pack_name}_{endpoint_name}",
                'pack': 'placebo',
                'condition': 'placebo',
                'metric': endpoint_name,
                'score': endpoint_result.get('placebo_score', 0.0),
                'model': model_name,
                'timestamp': timestamp
            })
    
    # Process secondary endpoints
    for endpoint_name, endpoint_result in endpoint_data.get("secondary_endpoints", {}).items():
        data.append({
            'item_id': f"{pack_name}_{endpoint_name}",
            'pack': pack_name,
            'condition': 'treatment',
            'metric': endpoint_name,
            'score': endpoint_result.get('treatment_score', 0.0),
            'model': model_name,
            'timestamp': timestamp
        })
        data.append({
            'item_id': f"{pack_name}_{endpoint_name}",
            'pack': 'control',
            'condition': 'baseline',
            'metric': endpoint_name,
            'score': endpoint_result.get('baseline_score', 0.0),
            'model': model_name,
            'timestamp': timestamp
        })
        if endpoint_result.get('placebo_score') is not None:
            data.append({
                'item_id': f"{pack_name}_{endpoint_name}",
                'pack': 'placebo',
                'condition': 'placebo',
                'metric': endpoint_name,
                'score': endpoint_result.get('placebo_score', 0.0),
                'model': model_name,
                'timestamp': timestamp
            })
    
    return data


def analyze_endpoints(input_dir: Path = None, input_file: Path = None, output_dir: Path = None):
    """Run statistical analysis on endpoint files"""
    
    # Determine input files
    if input_file:
        endpoint_files = [input_file]
    elif input_dir:
        endpoint_files = list(input_dir.glob("endpoints_*.json"))
        if not endpoint_files:
            print(f"❌ No endpoint files found in {input_dir}")
            return
    else:
        print("❌ Must specify either --input-dir or --input-file")
        return
    
    print(f"📊 Found {len(endpoint_files)} endpoint file(s)")
    
    # Load all data
    all_data = []
    for file in endpoint_files:
        print(f"  Loading: {file.name}")
        try:
            endpoint_data = load_endpoint_file(file)
            file_data = extract_data_from_endpoint(endpoint_data)
            all_data.extend(file_data)
        except Exception as e:
            print(f"  ⚠️  Error loading {file.name}: {e}")
            continue
    
    if not all_data:
        print("❌ No data loaded")
        return
    
    print(f"✅ Loaded {len(all_data)} data points")
    
    # Convert to DataFrame
    df = pd.DataFrame(all_data)
    
    # Filter out control pack from treatment comparisons (keep for baseline)
    # We want to compare each pack against control
    analysis_df = df[df['pack'] != 'control'].copy()
    control_df = df[df['pack'] == 'control'].copy()
    
    # Merge to create paired data structure
    # For each pack-metric combination, we need treatment and baseline scores
    paired_data = []
    for pack in analysis_df['pack'].unique():
        pack_data = analysis_df[analysis_df['pack'] == pack]
        for metric in pack_data['metric'].unique():
            metric_pack_data = pack_data[pack_data['metric'] == metric]
            metric_control_data = control_df[control_df['metric'] == metric]
            
            if len(metric_pack_data) > 0 and len(metric_control_data) > 0:
                # Get treatment scores
                treatment_scores = metric_pack_data[metric_pack_data['condition'] == 'treatment']['score'].values
                # Get baseline scores
                baseline_scores = metric_control_data[metric_control_data['condition'] == 'baseline']['score'].values
                
                # Create paired observations
                min_len = min(len(treatment_scores), len(baseline_scores))
                for i in range(min_len):
                    paired_data.append({
                        'item_id': f"{pack}_{metric}_{i}",
                        'pack': pack,
                        'metric': metric,
                        'condition': 'treatment',
                        'score': treatment_scores[i] if i < len(treatment_scores) else 0.0
                    })
                    paired_data.append({
                        'item_id': f"{pack}_{metric}_{i}",
                        'pack': 'control',
                        'metric': metric,
                        'condition': 'baseline',
                        'score': baseline_scores[i] if i < len(baseline_scores) else 0.0
                    })
    
    if not paired_data:
        # Fallback: use simple structure
        print("⚠️  Using simplified data structure")
        paired_df = df.copy()
    else:
        paired_df = pd.DataFrame(paired_data)
    
    print(f"📈 Analyzing {len(paired_df)} paired observations")
    print(f"   Packs: {sorted(paired_df['pack'].unique())}")
    print(f"   Metrics: {sorted(paired_df['metric'].unique())}")
    
    # Create analyzer
    analyzer = StatisticalAnalyzer(alpha=0.05, n_bootstrap=10000)
    
    # Run analysis
    print("\n🔬 Running statistical analysis...")
    results = analyzer.analyze_experiment(paired_df)
    
    # Print summary
    print("\n" + "="*70)
    print("📊 Statistical Analysis Results")
    print("="*70)
    print(f"\nTotal tests: {results['summary']['total_tests']}")
    print(f"Significant tests: {results['summary']['significant_tests']}")
    print(f"Significant rate: {results['summary']['significant_rate']:.1%}")
    
    # FDR correction info
    fdr_info = results.get('fdr_correction', {})
    if fdr_info:
        print(f"\nFDR Correction (Benjamini-Hochberg):")
        print(f"  Tests analyzed: {fdr_info.get('n_tests', 0)}")
        print(f"  Raw significant: {fdr_info.get('n_significant_raw', 0)}")
        print(f"  FDR significant: {fdr_info.get('n_significant_fdr', 0)}")
    
    # Effect sizes
    effect_sizes = results['summary'].get('effect_sizes', {})
    if effect_sizes.get('mean_cohens_d') is not None and not pd.isna(effect_sizes['mean_cohens_d']):
        print(f"\nMean Cohen's d: {effect_sizes['mean_cohens_d']:.3f}")
    if effect_sizes.get('mean_cliffs_delta') is not None and not pd.isna(effect_sizes['mean_cliffs_delta']):
        print(f"Mean Cliff's delta: {effect_sizes['mean_cliffs_delta']:.3f}")
    
    # Show significant results
    significant_tests = [t for t in results['statistical_tests'] if t.get('significant', False)]
    if significant_tests:
        print(f"\n✅ Significant Results (FDR corrected, p < 0.05):")
        for test in significant_tests[:10]:  # Show first 10
            print(f"\n  {test['pack_name']} - {test['metric']} ({test['test_name']}):")
            print(f"    p-value: {test['p_value']:.4f} -> FDR: {test['p_value_fdr']:.4f}")
            print(f"    Effect size ({test['effect_size_type']}): {test['effect_size']:.3f}")
            print(f"    {test.get('interpretation', 'N/A')}")
        if len(significant_tests) > 10:
            print(f"\n  ... and {len(significant_tests) - 10} more significant results")
    else:
        print("\n⚠️  No significant results after FDR correction")
        print("   Check raw p-values in the output file for borderline results")
    
    # Save results
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = Path("outputs/analysis")
        output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "statistical_analysis_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run statistical analysis on endpoint calculation outputs"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        help="Directory containing endpoint JSON files"
    )
    parser.add_argument(
        "--input-file",
        type=str,
        help="Single endpoint JSON file to analyze"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/analysis",
        help="Output directory for analysis results"
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir) if args.input_dir else None
    input_file = Path(args.input_file) if args.input_file else None
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    analyze_endpoints(input_dir=input_dir, input_file=input_file, output_dir=output_dir)


if __name__ == "__main__":
    main()

