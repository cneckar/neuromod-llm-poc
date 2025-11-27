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
from tqdm import tqdm

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
    
    # Create a unique identifier for this file to allow aggregation across multiple files
    # Use timestamp to make item_id unique per file, but still matchable by pack-metric
    file_id = timestamp or model_name
    
    # Process primary endpoints
    for endpoint_name, endpoint_result in endpoint_data.get("primary_endpoints", {}).items():
        # Treatment vs baseline
        # item_id includes file_id to make each file's data unique, but we'll aggregate by pack-metric
        data.append({
            'item_id': f"{pack_name}_{endpoint_name}_{file_id}",
            'pack': pack_name,
            'condition': 'treatment',
            'metric': endpoint_name,
            'score': endpoint_result.get('treatment_score', 0.0),
            'model': model_name,
            'timestamp': timestamp
        })
        data.append({
            'item_id': f"{pack_name}_{endpoint_name}_{file_id}",
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
                'item_id': f"{pack_name}_{endpoint_name}_{file_id}",
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
            'item_id': f"{pack_name}_{endpoint_name}_{file_id}",
            'pack': pack_name,
            'condition': 'treatment',
            'metric': endpoint_name,
            'score': endpoint_result.get('treatment_score', 0.0),
            'model': model_name,
            'timestamp': timestamp
        })
        data.append({
            'item_id': f"{pack_name}_{endpoint_name}_{file_id}",
            'pack': 'control',
            'condition': 'baseline',
            'metric': endpoint_name,
            'score': endpoint_result.get('baseline_score', 0.0),
            'model': model_name,
            'timestamp': timestamp
        })
        if endpoint_result.get('placebo_score') is not None:
            data.append({
                'item_id': f"{pack_name}_{endpoint_name}_{file_id}",
                'pack': 'placebo',
                'condition': 'placebo',
                'metric': endpoint_name,
                'score': endpoint_result.get('placebo_score', 0.0),
                'model': model_name,
                'timestamp': timestamp
            })
    
    return data


def analyze_endpoints(input_dir: Path = None, input_file: Path = None, output_dir: Path = None, output_file: Path = None):
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
    
    # Load all data with progress bar
    all_data = []
    errors = []
    with tqdm(total=len(endpoint_files), desc="Loading endpoint files", unit="file") as pbar:
        for file in endpoint_files:
            pbar.set_postfix(file=file.name[:30] + "..." if len(file.name) > 30 else file.name)
            try:
                endpoint_data = load_endpoint_file(file)
                file_data = extract_data_from_endpoint(endpoint_data)
                all_data.extend(file_data)
            except Exception as e:
                error_msg = f"Error loading {file.name}: {e}"
                errors.append(error_msg)
                tqdm.write(f"  ⚠️  {error_msg}")
            finally:
                pbar.update(1)
    
    if errors:
        print(f"\n⚠️  {len(errors)} file(s) had errors during loading")
    
    if not all_data:
        print("❌ No data loaded")
        return
    
    print(f"✅ Loaded {len(all_data)} data points")
    
    # Convert to DataFrame
    df = pd.DataFrame(all_data)
    
    # Debug: Check initial data structure
    print(f"\n🔍 Initial Data Structure:")
    print(f"   Total rows: {len(df)}")
    print(f"   Columns: {list(df.columns)}")
    if 'item_id' in df.columns:
        item_id_counts = df.groupby('item_id').size()
        print(f"   Unique item_ids: {len(item_id_counts)}")
        print(f"   Item_ids with 2+ rows: {(item_id_counts >= 2).sum()}")
        # Show sample item_ids
        sample_ids = item_id_counts.head(3)
        print(f"   Sample item_ids: {list(sample_ids.index)}")
        for item_id in sample_ids.index[:2]:
            item_data = df[df['item_id'] == item_id][['pack', 'condition', 'metric', 'score']]
            print(f"      {item_id}:")
            for _, row in item_data.iterrows():
                print(f"        {row['pack']} ({row['condition']}): score={row['score']}")
    
    # The data structure from extract_data_from_endpoint already has proper item_id pairing
    # Each item_id like "{pack_name}_{metric}" appears twice:
    #   - Once with pack={pack_name}, condition='treatment'
    #   - Once with pack='control', condition='baseline'
    # This is exactly what the analyzer needs for paired tests
    
    # Use the data directly - it should already have proper item_id pairing
    paired_df = df.copy()
    
    print(f"📈 Analyzing {len(paired_df)} paired observations")
    print(f"   Packs: {sorted(paired_df['pack'].unique())}")
    print(f"   Metrics: {sorted(paired_df['metric'].unique())}")
    
    # Debug: Check data quality
    print(f"\n🔍 Data Quality Check:")
    print(f"   Total rows: {len(paired_df)}")
    print(f"   Missing scores: {paired_df['score'].isna().sum()}")
    print(f"   Zero scores: {(paired_df['score'] == 0).sum()}")
    print(f"   Score range: [{paired_df['score'].min():.4f}, {paired_df['score'].max():.4f}]")
    print(f"   Score mean: {paired_df['score'].mean():.4f}")
    print(f"   Score std: {paired_df['score'].std():.4f}")
    
    # Check if we have proper item_id matching for paired tests
    if 'item_id' in paired_df.columns:
        item_id_counts = paired_df.groupby('item_id').size()
        print(f"   Item IDs: {len(item_id_counts)} unique")
        print(f"   Items with 2+ observations (needed for pairing): {(item_id_counts >= 2).sum()}")
        if (item_id_counts >= 2).sum() == 0:
            print("   ⚠️  WARNING: No item_ids have multiple observations - pairing will fail!")
    
    # Create analyzer
    analyzer = StatisticalAnalyzer(alpha=0.05, n_bootstrap=10000)
    
    # Calculate total number of tests for progress bar
    metrics = paired_df['metric'].unique()
    packs = [p for p in paired_df['pack'].unique() if p != 'control']
    total_tests = len(metrics) * len(packs) * 2  # 2 tests per pack-metric (t-test and Wilcoxon)
    
    # Run analysis with progress bar
    print("\n🔬 Running statistical analysis...")
    print(f"   Will run ~{total_tests} statistical tests ({len(metrics)} metrics × {len(packs)} packs × 2 tests)")
    
    # Wrap the analyzer's test methods to track progress
    pbar_ref = {'pbar': None}  # Will hold the progress bar reference
    
    original_paired_t_test = analyzer.paired_t_test
    original_wilcoxon_test = analyzer.wilcoxon_test
    
    def tracked_paired_t_test(*args, **kwargs):
        result = original_paired_t_test(*args, **kwargs)
        if pbar_ref['pbar']:
            pbar_ref['pbar'].update(1)
            # Update postfix with current test info
            if len(args) >= 4:
                pbar_ref['pbar'].set_postfix(test=f"{args[2]}-{args[3]}")
        return result
    
    def tracked_wilcoxon_test(*args, **kwargs):
        result = original_wilcoxon_test(*args, **kwargs)
        if pbar_ref['pbar']:
            pbar_ref['pbar'].update(1)
            # Update postfix with current test info
            if len(args) >= 4:
                pbar_ref['pbar'].set_postfix(test=f"{args[2]}-{args[3]}")
        return result
    
    analyzer.paired_t_test = tracked_paired_t_test
    analyzer.wilcoxon_test = tracked_wilcoxon_test
    
    # Create progress bar and run analysis
    with tqdm(total=total_tests, desc="Running statistical tests", unit="test", 
             bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
        pbar_ref['pbar'] = pbar
        try:
            results = analyzer.analyze_experiment(paired_df)
        finally:
            # Restore original methods
            analyzer.paired_t_test = original_paired_t_test
            analyzer.wilcoxon_test = original_wilcoxon_test
            pbar_ref['pbar'] = None
    
    # Print summary
    print("\n" + "="*70)
    print("📊 Statistical Analysis Results")
    print("="*70)
    print(f"\nTotal tests: {results['summary']['total_tests']}")
    if results['summary'].get('nan_tests', 0) > 0:
        print(f"NaN tests (zero variance or insufficient data): {results['summary']['nan_tests']}")
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
    if output_file:
        # User specified a file path directly
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
    elif output_dir:
        # User specified a directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "statistical_analysis_results.json"
    else:
        # Default directory
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
        help="Output directory for analysis results (default: outputs/analysis)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file path (e.g., outputs/analysis/statistical_results.json). Overrides --output-dir if specified."
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir) if args.input_dir else None
    input_file = Path(args.input_file) if args.input_file else None
    output_dir = Path(args.output_dir) if args.output_dir else None
    output_file = Path(args.output) if args.output else None
    
    analyze_endpoints(input_dir=input_dir, input_file=input_file, output_dir=output_dir, output_file=output_file)


if __name__ == "__main__":
    main()

