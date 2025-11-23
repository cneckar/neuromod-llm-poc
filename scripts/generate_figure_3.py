#!/usr/bin/env python3
"""
Generate Figure 3: Behavioral Signature Radar Plots

This script reads endpoint JSON files from the input directory,
filters for the most recent run of each pack, and generates a radar chart
showing multi-axial behavioral profiles across different neuromodulation conditions.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from math import pi
import os
import glob
import re
import argparse

def parse_timestamp(filename):
    """Extract timestamp from filename (YYYYMMDD_HHMMSS)"""
    match = re.search(r'_(\d{8}_\d{6})\.json$', filename)
    return match.group(1) if match else "00000000_000000"

def find_endpoint_files(input_dir):
    """Scan input directory for endpoint JSON files and return most recent for each pack"""
    files = glob.glob(os.path.join(input_dir, "endpoints_*.json"))
    
    # Group by pack name
    pack_files = {}
    for filepath in files:
        filename = os.path.basename(filepath)
        
        # Extract pack name from filename: endpoints_<pack>_<model>_<timestamp>.json
        match = re.search(r'endpoints_(\w+)_', filename)
        if match:
            pack_name = match.group(1).title()
            timestamp = parse_timestamp(filename)
            
            # Keep most recent for each pack
            if pack_name not in pack_files or timestamp > parse_timestamp(os.path.basename(pack_files[pack_name])):
                pack_files[pack_name] = filepath
    
    return pack_files

def load_endpoint_data(filepath):
    """Load and extract metrics from endpoint JSON file"""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def extract_metrics(data, pack_name):
    """Extract the 5 metrics for radar plot from endpoint data"""
    metrics = {
        'psychedelic_detection': 0.0,
        'depressant_detection': 0.0,
        'stimulant_detection': 0.0,
        'cognitive_performance': 0.0,
        'social_behavior': 0.0
    }
    
    # Extract primary endpoints
    primary_endpoints = data.get('primary_endpoints', {})
    
    # Psychedelic Detection
    if 'psychedelic_detection' in primary_endpoints:
        metrics['psychedelic_detection'] = primary_endpoints['psychedelic_detection'].get('treatment_score', 0.0)
    
    # Depressant Detection
    if 'depressant_detection' in primary_endpoints:
        metrics['depressant_detection'] = primary_endpoints['depressant_detection'].get('treatment_score', 0.0)
    
    # Stimulant Detection
    if 'stimulant_detection' in primary_endpoints:
        metrics['stimulant_detection'] = primary_endpoints['stimulant_detection'].get('treatment_score', 0.0)
    
    # Extract secondary endpoints
    secondary_endpoints = data.get('secondary_endpoints', {})
    
    # Cognitive Performance (normalize by dividing by 3.0)
    if 'cognitive_performance' in secondary_endpoints:
        cog_score = secondary_endpoints['cognitive_performance'].get('treatment_score', 0.0)
        metrics['cognitive_performance'] = min(1.0, cog_score / 3.0)
    
    # Social Behavior (normalize by dividing by 3.0)
    if 'social_behavior' in secondary_endpoints:
        soc_score = secondary_endpoints['social_behavior'].get('treatment_score', 0.0)
        metrics['social_behavior'] = min(1.0, soc_score / 3.0)
    
    return metrics

def calculate_placebo_baseline(pack_files, input_dir):
    """Calculate placebo baseline from all available endpoint files"""
    placebo_scores = {
        'psychedelic_detection': [],
        'depressant_detection': [],
        'stimulant_detection': [],
        'cognitive_performance': [],
        'social_behavior': []
    }
    
    # Load all endpoint files to collect placebo scores
    all_files = glob.glob(os.path.join(input_dir, "endpoints_*.json"))
    
    for filepath in all_files:
        data = load_endpoint_data(filepath)
        if not data:
            continue
        
        primary = data.get('primary_endpoints', {})
        secondary = data.get('secondary_endpoints', {})
        
        # Collect placebo scores
        if 'psychedelic_detection' in primary:
            score = primary['psychedelic_detection'].get('placebo_score', 0.0)
            if score is not None:
                placebo_scores['psychedelic_detection'].append(score)
        
        if 'depressant_detection' in primary:
            score = primary['depressant_detection'].get('placebo_score', 0.0)
            if score is not None:
                placebo_scores['depressant_detection'].append(score)
        
        if 'stimulant_detection' in primary:
            score = primary['stimulant_detection'].get('placebo_score', 0.0)
            if score is not None:
                placebo_scores['stimulant_detection'].append(score)
        
        if 'cognitive_performance' in secondary:
            score = secondary['cognitive_performance'].get('placebo_score', 0.0)
            if score is not None:
                placebo_scores['cognitive_performance'].append(score / 3.0)
        
        if 'social_behavior' in secondary:
            score = secondary['social_behavior'].get('placebo_score', 0.0)
            if score is not None:
                placebo_scores['social_behavior'].append(score / 3.0)
    
    # Calculate averages
    baseline = {
        'psychedelic_detection': np.mean(placebo_scores['psychedelic_detection']) if placebo_scores['psychedelic_detection'] else 0.0,
        'depressant_detection': np.mean(placebo_scores['depressant_detection']) if placebo_scores['depressant_detection'] else 0.0,
        'stimulant_detection': np.mean(placebo_scores['stimulant_detection']) if placebo_scores['stimulant_detection'] else 0.0,
        'cognitive_performance': np.mean(placebo_scores['cognitive_performance']) if placebo_scores['cognitive_performance'] else 0.0,
        'social_behavior': np.mean(placebo_scores['social_behavior']) if placebo_scores['social_behavior'] else 0.0
    }
    
    return baseline

def create_radar_chart(metrics_data, output_path):
    """Generate radar chart from metrics data"""
    categories = ['Psychedelic\nDetection', 'Depressant\nDetection', 'Stimulant\nDetection', 
                  'Cognitive\nPerformance', 'Social\nBehavior']
    N = len(categories)
    
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111, polar=True)
    
    # Setup Axis
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    plt.xticks(angles[:-1], categories, size=11)
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=8)
    plt.ylim(0, 1.0)
    
    # Color palette
    colors = {
        "LSD": "#9b59b6",      # Purple
        "Morphine": "#3498db", # Blue
        "Caffeine": "#e67e22", # Orange
        "Placebo": "#95a5a6",  # Grey
        "Psilocybin": "#8e44ad", # Dark Purple
        "DMT": "#7d3c98",      # Purple
        "Heroin": "#2980b9",  # Dark Blue
        "Amphetamine": "#d35400", # Dark Orange
        "Cocaine": "#c0392b",  # Red
        "Default": "#34495e"   # Dark Grey for unknown
    }
    
    # Plot each condition
    plotted_count = 0
    for drug, metrics in sorted(metrics_data.items()):
        # Convert metrics dict to list in correct order
        values = [
            metrics.get('psychedelic_detection', 0.0),
            metrics.get('depressant_detection', 0.0),
            metrics.get('stimulant_detection', 0.0),
            metrics.get('cognitive_performance', 0.0),
            metrics.get('social_behavior', 0.0)
        ]
        
        if any(v > 0 for v in values):  # Only plot if we have data
            values += values[:1]  # Close loop
            
            color = colors.get(drug, colors["Default"])
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=drug, color=color)
            ax.fill(angles, values, color=color, alpha=0.1)
            plotted_count += 1
    
    if plotted_count == 0:
        print("Warning: No valid metrics data found! Cannot generate figure.")
        return False
    
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title("Figure 3: Behavioral Signature Radar Plots\n(Multi-Axial Profile)", size=16, y=1.1)
    
    plt.tight_layout()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    plt.savefig(output_path, dpi=300)
    print(f"Generated {output_path} with {plotted_count} behavioral profiles")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Figure 3: Behavioral Signature Radar Plots"
    )
    parser.add_argument(
        "--input",
        default="outputs/endpoints",
        help="Directory containing endpoint JSON files (default: outputs/endpoints)"
    )
    parser.add_argument(
        "--output",
        default="outputs/figure_3_radar_plots.png",
        help="Output image path (default: outputs/figure_3_radar_plots.png)"
    )
    args = parser.parse_args()
    
    print(f"Scanning for endpoint files in: {args.input}")
    pack_files = find_endpoint_files(args.input)
    
    if not pack_files:
        print(f"No endpoint files found in {args.input}")
        print("Expected files matching pattern: endpoints_<pack>_<model>_<timestamp>.json")
        exit(1)
    
    print(f"Found {len(pack_files)} pack(s) with endpoint data:")
    for name, path in sorted(pack_files.items()):
        print(f"  - {name}: {path}")
    
    # Calculate placebo baseline from all files
    print("Calculating placebo baseline from all endpoint files...")
    placebo_baseline = calculate_placebo_baseline(pack_files, args.input)
    
    # Extract metrics for each pack
    metrics_data = {}
    for pack_name, filepath in pack_files.items():
        data = load_endpoint_data(filepath)
        if data:
            metrics = extract_metrics(data, pack_name)
            metrics_data[pack_name] = metrics
            print(f"  Extracted metrics for {pack_name}")
    
    # Add placebo baseline
    metrics_data["Placebo"] = placebo_baseline
    print("  Added Placebo baseline")
    
    success = create_radar_chart(metrics_data, args.output)
    if not success:
        exit(1)

