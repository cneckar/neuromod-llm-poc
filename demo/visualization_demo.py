#!/usr/bin/env python3
"""
Visualization System Demo

This script demonstrates the visualization and results generation system
for the neuromodulation research project. It generates all required figures
and tables for the paper.
"""

import sys
import os
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from neuromod.testing.visualization import NeuromodVisualizer
from neuromod.testing.results_templates import (
    ResultsTemplateGenerator, 
    create_sample_results
)


def main():
    """Main demonstration function"""
    print("🎨 NEUROMODULATION VISUALIZATION SYSTEM DEMO")
    print("=" * 60)
    
    # Create output directory
    output_dir = project_root / "outputs" / "analysis" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Output directory: {output_dir}")
    
    # Initialize visualizer
    print("\n🔧 Initializing visualizer...")
    visualizer = NeuromodVisualizer(output_dir=str(output_dir))
    
    # Initialize results template generator
    print("📊 Initializing results template generator...")
    generator = ResultsTemplateGenerator(output_dir=str(output_dir.parent))
    
    # Generate sample data
    print("\n📈 Generating sample data...")
    sample_data = visualizer._generate_sample_data()
    sample_results = create_sample_results()
    
    print("✅ Sample data generated")
    print(f"   - ROC curves for {len(sample_data['pdq_s_roc'])} models")
    print(f"   - Signature data for {len(sample_data['signature_data'])} categories")
    print(f"   - Task data for {len(sample_data['task_data'])} packs")
    
    # Generate all figures
    print("\n🎨 Generating figures...")
    try:
        visual_results = visualizer.generate_all_figures_and_tables(sample_data)
        print("✅ All figures and tables generated successfully!")
        
        # List generated files
        figure_files = list(output_dir.glob("figure_*.png"))
        table_files = list(output_dir.glob("table_*.csv"))
        
        print(f"\n📊 Generated {len(figure_files)} figures:")
        for file in figure_files:
            print(f"   - {file.name}")
        
        print(f"\n📋 Generated {len(table_files)} tables:")
        for file in table_files:
            print(f"   - {file.name}")
            
    except Exception as e:
        print(f"❌ Error generating figures: {e}")
        return
    
    # Generate reports
    print("\n📄 Generating reports...")
    try:
        # HTML report
        html_report = generator.create_html_report(sample_results, "demo_report.html")
        print("✅ HTML report generated")
        
        # JSON summary
        json_summary = generator.create_json_summary(sample_results, "demo_summary.json")
        print("✅ JSON summary generated")
        
        # CSV export
        csv_data = generator.export_results_csv(sample_results, "demo_data.csv")
        print("✅ CSV data exported")
        
        print(f"\n📊 Generated reports:")
        print(f"   - demo_report.html ({len(html_report)} characters)")
        print(f"   - demo_summary.json ({len(json_summary)} characters)")
        print(f"   - demo_data.csv ({len(csv_data)} rows)")
        
    except Exception as e:
        print(f"❌ Error generating reports: {e}")
        return
    
    # Show summary statistics
    print("\n📈 Summary Statistics:")
    summary_stats = generator._calculate_summary_stats(sample_results)
    for key, value in summary_stats.items():
        if isinstance(value, float):
            print(f"   - {key}: {value:.3f}")
        else:
            print(f"   - {key}: {value}")
    
    # Show file sizes
    print("\n📁 File Sizes:")
    all_files = list(output_dir.glob("*")) + list(output_dir.parent.glob("demo_*"))
    for file in all_files:
        if file.is_file():
            size_kb = file.stat().st_size / 1024
            print(f"   - {file.name}: {size_kb:.1f} KB")
    
    print("\n🎉 VISUALIZATION DEMO COMPLETE!")
    print("=" * 60)
    print("📂 All outputs saved to:")
    print(f"   - Figures: {output_dir}")
    print(f"   - Reports: {output_dir.parent}")
    print("\n💡 Next steps:")
    print("   1. Review generated figures and tables")
    print("   2. Customize data for your specific results")
    print("   3. Integrate with your analysis pipeline")
    print("   4. Use templates for paper submission")


if __name__ == "__main__":
    main()
