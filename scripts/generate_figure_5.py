#!/usr/bin/env python3
"""
Generate Figure 5: Empirical Emotion Signatures

This script reads emotion result JSON files from the input directory,
filters for the most recent run of each pack, and generates a radar chart
showing emotion profiles across different neuromodulation conditions.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from math import pi
import os
import glob
import re
import argparse

EMOTIONS = ['Joy', 'Trust', 'Fear', 'Surprise', 'Sadness', 'Disgust', 'Anger', 'Anticipation']
COLORS = {
    "LSD": "#9b59b6",      # Purple
    "Morphine": "#3498db", # Blue
    "Caffeine": "#e67e22", # Orange
    "Placebo": "#95a5a6",  # Grey
    "Psilocybin": "#8e44ad", # Dark Purple
    "DMT": "#7d3c98",      # Purple
    "Heroin": "#2980b9",  # Dark Blue
    "Amphetamine": "#d35400", # Dark Orange
    "Cocaine": "#c0392b",  # Red
    "Default": "#34495e"   # Dark Grey for unknown packs
}

def parse_timestamp(filename):
    """Extract timestamp from filename (YYYYMMDD_HHMMSS or similar patterns)"""
    # Try multiple patterns
    patterns = [
        r'_(\d{8}_\d{6})\.json$',  # YYYYMMDD_HHMMSS
        r'_(\d{14})\.json$',        # YYYYMMDDHHMMSS
        r'(\d{8}_\d{6})',           # Anywhere in filename
    ]
    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            return match.group(1)
    return "00000000_000000"

def find_emotion_files(input_dir):
    """Scan input directory for emotion result files and return most recent for each pack"""
    # Look for files matching emotion result patterns
    patterns = [
        os.path.join(input_dir, "**/emotion_results_*.json"),
        os.path.join(input_dir, "**/*emotion*.json"),
        os.path.join(input_dir, "emotion_results_*.json"),
        os.path.join(input_dir, "*emotion*.json"),
    ]
    
    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern, recursive=True))
    
    # Group by pack name
    pack_files = {}
    for filepath in files:
        filename = os.path.basename(filepath)
        
        # Extract pack name from filename
        # Patterns: emotion_results_story_emotion_test_<pack>.json
        #          emotion_results_<pack>_*.json
        pack_name = None
        for pattern in [
            r'emotion_results_story_emotion_test_(\w+)\.json',
            r'emotion_results_(\w+)_\d+\.json',
            r'emotion_results_(\w+)\.json',
            r'(\w+)_emotion_results\.json',
        ]:
            match = re.search(pattern, filename, re.IGNORECASE)
            if match:
                pack_name = match.group(1).title()
                break
        
        if not pack_name:
            # Try to extract from path
            dir_parts = filepath.split(os.sep)
            for part in dir_parts:
                if part.lower() in ['lsd', 'morphine', 'caffeine', 'placebo', 'psilocybin', 
                                   'dmt', 'heroin', 'amphetamine', 'cocaine']:
                    pack_name = part.title()
                    break
        
        if pack_name:
            timestamp = parse_timestamp(filename)
            if pack_name not in pack_files or timestamp > parse_timestamp(os.path.basename(pack_files[pack_name])):
                pack_files[pack_name] = filepath
    
    return pack_files

def load_emotion_profile(filepath):
    """Extracts and normalizes emotion counts from the JSON log."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # If your logger structure is different, adjust this path
        # Assuming: data['summary']['emotion_totals'][emotion]['up']
        totals = data.get('summary', {}).get('emotion_totals', {})
        
        if not totals:
            print(f"Warning: No emotion data found in {filepath}")
            return [0.1] * 8  # Return baseline noise to avoid crash

        counts = []
        for emo in [e.lower() for e in EMOTIONS]:
            # We sum 'up' (increase) events as the primary signal
            # You could also subtract 'down' events for a net score
            score = totals.get(emo, {}).get('up', 0)
            counts.append(score)
            
        # Normalize to 0.0 - 1.0 range (relative to max emotion in this run)
        # This emphasizes the *shape* of the profile rather than absolute volume
        max_val = max(counts) if max(counts) > 0 else 1
        normalized = [c / max_val for c in counts]
        
        # Add a small baseline (0.05) so the chart doesn't collapse to center
        normalized = [min(1.0, c + 0.05) for c in normalized]
        
        return normalized
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return [0.0] * 8

def create_radar_chart(file_map, output_path):
    """Generate radar chart from available emotion data"""
    N = len(EMOTIONS)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1] # Close the loop
    
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111, polar=True)
    
    # Setup Axis
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    plt.xticks(angles[:-1], EMOTIONS, size=12)
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["", "", "", "", ""], color="grey", size=8)
    plt.ylim(0, 1.0)
    
    # Plot each condition
    plotted_count = 0
    for name, filepath in sorted(file_map.items()):
        if os.path.exists(filepath):
            values = load_emotion_profile(filepath)
            if values and any(v > 0 for v in values):  # Check if we have valid data
                values += values[:1] # Close loop
                
                color = COLORS.get(name, COLORS["Default"])
                ax.plot(angles, values, linewidth=2, linestyle='solid', label=name, color=color)
                ax.fill(angles, values, color=color, alpha=0.1)
                plotted_count += 1
            else:
                print(f"Skipping {name}: No valid emotion data in {filepath}")
        else:
            print(f"Skipping {name}: File not found at {filepath}")
    
    if plotted_count == 0:
        print("Warning: No valid emotion data found! Cannot generate figure.")
        return False
    
    plt.title("Figure 5: Empirical Emotion Signatures\n(Story Generation Task)", size=16, y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.tight_layout()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    plt.savefig(output_path, dpi=300)
    print(f"Generated {output_path} with {plotted_count} emotion profiles")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Figure 5: Empirical Emotion Signatures radar chart"
    )
    parser.add_argument(
        "--input",
        default="outputs/reports/emotion",
        help="Directory containing emotion result JSON files (default: outputs/reports/emotion)"
    )
    parser.add_argument(
        "--output",
        default="outputs/figure_5_emotion_signatures.png",
        help="Output image path (default: outputs/figure_5_emotion_signatures.png)"
    )
    args = parser.parse_args()
    
    print(f"Scanning for emotion files in: {args.input}")
    file_map = find_emotion_files(args.input)
    
    if not file_map:
        print(f"No emotion result files found in {args.input}")
        print("Expected files matching patterns like:")
        print("  - emotion_results_story_emotion_test_<pack>.json")
        print("  - emotion_results_<pack>_*.json")
        exit(1)
    
    print(f"Found {len(file_map)} pack(s) with emotion data:")
    for name, path in sorted(file_map.items()):
        print(f"  - {name}: {path}")
    
    success = create_radar_chart(file_map, args.output)
    if not success:
        exit(1)