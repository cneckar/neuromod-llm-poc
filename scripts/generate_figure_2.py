#!/usr/bin/env python3
"""
Generate Figure 2: Primary Endpoint Detection Sensitivity

This script reads the endpoint JSON results from 'outputs/endpoints/',
filters for the most recent run of each pack, and generates the
detection sensitivity bar chart with significance annotations.
"""

import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import os
import glob
import re
import argparse

def parse_timestamp(filename):
    """Extract timestamp from filename (YYYYMMDD_HHMMSS)"""
    match = re.search(r'_(\d{8}_\d{6})\.json$', filename)
    return match.group(1) if match else "00000000_000000"

def load_data(input_dir):
    """Load and aggregate endpoint data"""
    files = glob.glob(os.path.join(input_dir, "endpoints_*.json"))
    data_buffer = []

    print(f"Found {len(files)} endpoint files.")

    for filepath in files:
        try:
            with open(filepath, 'r') as f:
                content = json.load(f)
            
            pack_name = content.get('pack_name', 'Unknown').title()
            filename = os.path.basename(filepath)
            timestamp = parse_timestamp(filename)
            
            # Determine category
            # You can also pull this from the JSON metadata if available
            p_lower = pack_name.lower()
            if p_lower in ['lsd', 'psilocybin', 'mescaline', 'dmt', '2c_b']:
                category = 'Psychedelic'
            elif p_lower in ['amphetamine', 'cocaine', 'methylphenidate', 'modafinil', 'caffeine']:
                category = 'Stimulant'
            elif p_lower in ['morphine', 'heroin', 'fentanyl', 'benzodiazepines', 'alcohol']:
                category = 'Depressant'
            else:
                category = 'Other'

            # Extract Primary Endpoint Score
            primary_endpoints = content.get('primary_endpoints', {})
            if primary_endpoints:
                # Grab the first primary endpoint (there is usually only one per pack type)
                key = list(primary_endpoints.keys())[0]
                endpoint_data = primary_endpoints[key]
                
                score = endpoint_data.get('treatment_score', 0.0)
                p_value = endpoint_data.get('p_value', 1.0)
                
                data_buffer.append({
                    "Drug": pack_name,
                    "Class": category,
                    "Score": score,
                    "P-Value": p_value,
                    "Timestamp": timestamp
                })
                
        except Exception as e:
            print(f"Skipping {filepath}: {e}")

    if not data_buffer:
        print("No valid data found!")
        return pd.DataFrame()

    # Convert to DataFrame
    df = pd.DataFrame(data_buffer)
    
    # Filter: Keep only the most recent run for each Drug
    # This prevents duplicate bars if you ran the same pack multiple times
    df_clean = df.sort_values('Timestamp', ascending=False).groupby('Drug').first().reset_index()
    
    # Add Placebo Baseline manually if it doesn't exist
    # (Placebo is 0.0 by definition in this relative scoring system)
    if not any(df_clean['Drug'] == 'Placebo'):
        new_row = pd.DataFrame([{
            "Drug": "Placebo", "Class": "Control", "Score": 0.0, "P-Value": 1.0, "Timestamp": "N/A"
        }])
        df_clean = pd.concat([df_clean, new_row], ignore_index=True)
        
    return df_clean

def generate_figure(df, output_path):
    """Generate the seaborn plot"""
    
    # Color Palette (Matches Paper)
    palette = {
        "Psychedelic": "#9b59b6", # Purple
        "Stimulant": "#e67e22",   # Orange
        "Depressant": "#3498db",  # Blue
        "Control": "#95a5a6",      # Grey
        "Other": "#95a5a6"
    }

    # Sort Order: Placebo -> Psych -> Stim -> Depress
    order = ["Placebo"] + \
            sorted(df[df["Class"] == "Psychedelic"]["Drug"].tolist()) + \
            sorted(df[df["Class"] == "Stimulant"]["Drug"].tolist()) + \
            sorted(df[df["Class"] == "Depressant"]["Drug"].tolist())

    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")
    
    ax = sns.barplot(
        data=df,
        x="Drug",
        y="Score",
        hue="Class",
        palette=palette,
        dodge=False,
        order=[o for o in order if o in df["Drug"].values] # Only plot present drugs
    )

    # Styling
    plt.title("Figure 2: Primary Endpoint Detection Sensitivity (Llama-3.1-8B)", fontsize=16, pad=20)
    plt.ylabel("Detection Score (0.0 - 1.0)", fontsize=12)
    plt.xlabel("Neuromodulation Pack", fontsize=12)
    plt.ylim(0, 1.2) # Room for stars
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.3, label="Chance Level")

    # Significance Stars
    for i, drug_name in enumerate([o for o in order if o in df["Drug"].values]):
        row = df[df["Drug"] == drug_name]
        if not row.empty:
            p_val = row.iloc[0]["P-Value"]
            score = row.iloc[0]["Score"]
            
            # Add stars for significant results
            if p_val < 0.001:
                label = "***"
            elif p_val < 0.01:
                label = "**"
            elif p_val < 0.05:
                label = "*"
            else:
                label = ""
            
            if label and score > 0:
                ax.text(i, score + 0.02, label, ha='center', color='black', fontsize=12, fontweight='bold')

    plt.legend(title="Drug Class", loc="upper right")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300)
    print(f"Figure saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs/endpoints", help="Directory containing endpoint JSONs")
    parser.add_argument("--output", default="outputs/figure_2_detection_sensitivity.png", help="Output image path")
    args = parser.parse_args()
    
    df = load_data(args.input)
    if not df.empty:
        generate_figure(df, args.output)