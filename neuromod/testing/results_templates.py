#!/usr/bin/env python3
"""
Results Templates System for Neuromodulation Research

This module provides comprehensive results formatting and template generation
for the paper "Neuromodulated Language Models: Prototyping Pharmacological 
Analogues and Blind, Placebo-Controlled Evaluation".

Features:
- Automated report generation
- Statistical result formatting
- LaTeX table generation
- HTML report templates
- Data export utilities
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import json
from datetime import datetime
import jinja2
from dataclasses import dataclass, asdict
import warnings

@dataclass
class StatisticalResult:
    """Data class for statistical test results"""
    test_name: str
    pack_name: str
    endpoint: str
    statistic: float
    p_value: float
    p_value_adjusted: float
    effect_size: float
    effect_size_type: str
    ci_lower: float
    ci_upper: float
    n_samples: int
    significant: bool

@dataclass
class PackResult:
    """Data class for pack-level results"""
    pack_name: str
    category: str
    primary_endpoints: Dict[str, StatisticalResult]
    secondary_endpoints: Dict[str, StatisticalResult]
    off_target_metrics: Dict[str, float]
    safety_violations: List[str]
    overall_significance: bool

@dataclass
class ModelResult:
    """Data class for model-level results"""
    model_name: str
    pack_results: Dict[str, PackResult]
    overall_performance: Dict[str, float]
    robustness_metrics: Dict[str, float]

class ResultsTemplateGenerator:
    """Main class for generating results templates and reports"""
    
    def __init__(self, output_dir: str = "outputs/analysis"):
        """
        Initialize the results template generator
        
        Args:
            output_dir: Directory to save generated reports and templates
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up Jinja2 environment for templating
        self.jinja_env = jinja2.Environment(
            loader=jinja2.DictLoader({
                'latex_table': self._get_latex_table_template(),
                'html_report': self._get_html_report_template(),
                'json_summary': self._get_json_summary_template()
            }),
            autoescape=jinja2.select_autoescape(['html', 'xml'])
        )
    
    def format_statistical_result(self, result: StatisticalResult, 
                                format_type: str = 'latex') -> str:
        """
        Format a statistical result for different output types
        
        Args:
            result: StatisticalResult object
            format_type: 'latex', 'html', 'markdown', or 'plain'
        
        Returns:
            Formatted string
        """
        if format_type == 'latex':
            return self._format_latex_result(result)
        elif format_type == 'html':
            return self._format_html_result(result)
        elif format_type == 'markdown':
            return self._format_markdown_result(result)
        else:
            return self._format_plain_result(result)
    
    def _format_latex_result(self, result: StatisticalResult) -> str:
        """Format result for LaTeX output"""
        significance = "***" if result.p_value_adjusted < 0.001 else \
                      "**" if result.p_value_adjusted < 0.01 else \
                      "*" if result.p_value_adjusted < 0.05 else ""
        
        return (f"{result.statistic:.3f}{significance} "
                f"({result.effect_size:.3f}, "
                f"95\\% CI: [{result.ci_lower:.3f}, {result.ci_upper:.3f}], "
                f"n={result.n_samples})")
    
    def _format_html_result(self, result: StatisticalResult) -> str:
        """Format result for HTML output"""
        significance = "***" if result.p_value_adjusted < 0.001 else \
                      "**" if result.p_value_adjusted < 0.01 else \
                      "*" if result.p_value_adjusted < 0.05 else ""
        
        return (f"<span class='statistic'>{result.statistic:.3f}{significance}</span> "
                f"<span class='effect-size'>({result.effect_size:.3f})</span> "
                f"<span class='ci'>95% CI: [{result.ci_lower:.3f}, {result.ci_upper:.3f}]</span> "
                f"<span class='sample-size'>n={result.n_samples}</span>")
    
    def _format_markdown_result(self, result: StatisticalResult) -> str:
        """Format result for Markdown output"""
        significance = "***" if result.p_value_adjusted < 0.001 else \
                      "**" if result.p_value_adjusted < 0.01 else \
                      "*" if result.p_value_adjusted < 0.05 else ""
        
        return (f"**{result.statistic:.3f}{significance}** "
                f"({result.effect_size:.3f}, "
                f"95% CI: [{result.ci_lower:.3f}, {result.ci_upper:.3f}], "
                f"n={result.n_samples})")
    
    def _format_plain_result(self, result: StatisticalResult) -> str:
        """Format result for plain text output"""
        significance = "***" if result.p_value_adjusted < 0.001 else \
                      "**" if result.p_value_adjusted < 0.01 else \
                      "*" if result.p_value_adjusted < 0.05 else ""
        
        return (f"{result.statistic:.3f}{significance} "
                f"({result.effect_size:.3f}, "
                f"95% CI: [{result.ci_lower:.3f}, {result.ci_upper:.3f}], "
                f"n={result.n_samples})")
    
    def create_latex_table(self, results: List[StatisticalResult], 
                          caption: str, label: str) -> str:
        """
        Create a LaTeX table from statistical results
        
        Args:
            results: List of StatisticalResult objects
            caption: Table caption
            label: Table label
        
        Returns:
            LaTeX table string
        """
        if not results:
            return ""
        
        # Create DataFrame
        df = pd.DataFrame([asdict(result) for result in results])
        
        # Format the table
        latex_table = df.to_latex(
            index=False,
            caption=caption,
            label=label,
            escape=False,
            column_format='l' + 'c' * (len(df.columns) - 1)
        )
        
        return latex_table
    
    def create_html_report(self, model_results: Dict[str, ModelResult], 
                          output_file: Optional[str] = None) -> str:
        """
        Create an HTML report from model results
        
        Args:
            model_results: Dictionary of model results
            output_file: Optional output file path
        
        Returns:
            HTML report string
        """
        template = self.jinja_env.get_template('html_report')
        
        # Prepare data for template
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'models': {},
            'summary_stats': self._calculate_summary_stats(model_results)
        }
        
        for model_name, model_result in model_results.items():
            report_data['models'][model_name] = {
                'pack_results': {},
                'overall_performance': model_result.overall_performance,
                'robustness_metrics': model_result.robustness_metrics
            }
            
            for pack_name, pack_result in model_result.pack_results.items():
                report_data['models'][model_name]['pack_results'][pack_name] = {
                    'category': pack_result.category,
                    'primary_endpoints': {
                        name: self.format_statistical_result(result, 'html')
                        for name, result in pack_result.primary_endpoints.items()
                    },
                    'secondary_endpoints': {
                        name: self.format_statistical_result(result, 'html')
                        for name, result in pack_result.secondary_endpoints.items()
                    },
                    'off_target_metrics': pack_result.off_target_metrics,
                    'safety_violations': pack_result.safety_violations,
                    'overall_significance': pack_result.overall_significance
                }
        
        html_content = template.render(**report_data)
        
        if output_file:
            output_path = self.output_dir / output_file
            with open(output_path, 'w') as f:
                f.write(html_content)
        
        return html_content
    
    def create_json_summary(self, model_results: Dict[str, ModelResult], 
                           output_file: Optional[str] = None) -> str:
        """
        Create a JSON summary of results
        
        Args:
            model_results: Dictionary of model results
            output_file: Optional output file path
        
        Returns:
            JSON summary string
        """
        # Convert to serializable format
        json_data = {
            'timestamp': datetime.now().isoformat(),
            'summary': self._calculate_summary_stats(model_results),
            'models': {}
        }
        
        for model_name, model_result in model_results.items():
            json_data['models'][model_name] = {
                'overall_performance': model_result.overall_performance,
                'robustness_metrics': model_result.robustness_metrics,
                'packs': {}
            }
            
            for pack_name, pack_result in model_result.pack_results.items():
                json_data['models'][model_name]['packs'][pack_name] = {
                    'category': pack_result.category,
                    'primary_endpoints': {
                        name: asdict(result) for name, result in pack_result.primary_endpoints.items()
                    },
                    'secondary_endpoints': {
                        name: asdict(result) for name, result in pack_result.secondary_endpoints.items()
                    },
                    'off_target_metrics': pack_result.off_target_metrics,
                    'safety_violations': pack_result.safety_violations,
                    'overall_significance': pack_result.overall_significance
                }
        
        json_str = json.dumps(json_data, indent=2, default=str)
        
        if output_file:
            output_path = self.output_dir / output_file
            with open(output_path, 'w') as f:
                f.write(json_str)
        
        return json_str
    
    def export_results_csv(self, model_results: Dict[str, ModelResult], 
                          output_file: Optional[str] = None) -> pd.DataFrame:
        """
        Export results to CSV format
        
        Args:
            model_results: Dictionary of model results
            output_file: Optional output file path
        
        Returns:
            DataFrame with all results
        """
        rows = []
        
        for model_name, model_result in model_results.items():
            for pack_name, pack_result in model_result.pack_results.items():
                # Primary endpoints
                for endpoint_name, result in pack_result.primary_endpoints.items():
                    rows.append({
                        'model': model_name,
                        'pack': pack_name,
                        'category': pack_result.category,
                        'endpoint': endpoint_name,
                        'endpoint_type': 'primary',
                        'test_name': result.test_name,
                        'statistic': result.statistic,
                        'p_value': result.p_value,
                        'p_value_adjusted': result.p_value_adjusted,
                        'effect_size': result.effect_size,
                        'effect_size_type': result.effect_size_type,
                        'ci_lower': result.ci_lower,
                        'ci_upper': result.ci_upper,
                        'n_samples': result.n_samples,
                        'significant': result.significant
                    })
                
                # Secondary endpoints
                for endpoint_name, result in pack_result.secondary_endpoints.items():
                    rows.append({
                        'model': model_name,
                        'pack': pack_name,
                        'category': pack_result.category,
                        'endpoint': endpoint_name,
                        'endpoint_type': 'secondary',
                        'test_name': result.test_name,
                        'statistic': result.statistic,
                        'p_value': result.p_value,
                        'p_value_adjusted': result.p_value_adjusted,
                        'effect_size': result.effect_size,
                        'effect_size_type': result.effect_size_type,
                        'ci_lower': result.ci_lower,
                        'ci_upper': result.ci_upper,
                        'n_samples': result.n_samples,
                        'significant': result.significant
                    })
        
        df = pd.DataFrame(rows)
        
        if output_file:
            output_path = self.output_dir / output_file
            df.to_csv(output_path, index=False)
        
        return df
    
    def _calculate_summary_stats(self, model_results: Dict[str, ModelResult]) -> Dict[str, Any]:
        """Calculate summary statistics across all models and packs"""
        total_packs = 0
        significant_packs = 0
        total_tests = 0
        significant_tests = 0
        
        effect_sizes = []
        p_values = []
        
        for model_result in model_results.values():
            for pack_result in model_result.pack_results.values():
                total_packs += 1
                if pack_result.overall_significance:
                    significant_packs += 1
                
                for result in pack_result.primary_endpoints.values():
                    total_tests += 1
                    if result.significant:
                        significant_tests += 1
                    effect_sizes.append(result.effect_size)
                    p_values.append(result.p_value)
                
                for result in pack_result.secondary_endpoints.values():
                    total_tests += 1
                    if result.significant:
                        significant_tests += 1
                    effect_sizes.append(result.effect_size)
                    p_values.append(result.p_value)
        
        return {
            'total_models': len(model_results),
            'total_packs': total_packs,
            'significant_packs': significant_packs,
            'pack_significance_rate': significant_packs / total_packs if total_packs > 0 else 0,
            'total_tests': total_tests,
            'significant_tests': significant_tests,
            'test_significance_rate': significant_tests / total_tests if total_tests > 0 else 0,
            'mean_effect_size': np.mean(effect_sizes) if effect_sizes else 0,
            'median_effect_size': np.median(effect_sizes) if effect_sizes else 0,
            'mean_p_value': np.mean(p_values) if p_values else 0,
            'median_p_value': np.median(p_values) if p_values else 0
        }
    
    def _get_latex_table_template(self) -> str:
        """Get LaTeX table template"""
        return """
\\begin{table}[h]
\\centering
{{ table_content }}
\\caption{{{ caption }}}
\\label{{{ label }}}
\\end{table}
"""
    
    def _get_html_report_template(self) -> str:
        """Get HTML report template"""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Neuromodulation Research Results</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background-color: #f0f0f0; padding: 20px; border-radius: 5px; }
        .model-section { margin: 20px 0; border: 1px solid #ddd; padding: 15px; }
        .pack-section { margin: 10px 0; padding: 10px; background-color: #f9f9f9; }
        .statistic { font-weight: bold; color: #0066cc; }
        .effect-size { color: #009900; }
        .ci { color: #666; }
        .sample-size { color: #999; }
        .summary { background-color: #e8f4f8; padding: 15px; border-radius: 5px; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Neuromodulation Research Results</h1>
        <p>Generated: {{ timestamp }}</p>
    </div>
    
    <div class="summary">
        <h2>Summary Statistics</h2>
        <ul>
            <li>Total Models: {{ summary_stats.total_models }}</li>
            <li>Total Packs: {{ summary_stats.total_packs }}</li>
            <li>Significant Packs: {{ summary_stats.significant_packs }} ({{ "%.1f"|format(summary_stats.pack_significance_rate * 100) }}%)</li>
            <li>Total Tests: {{ summary_stats.total_tests }}</li>
            <li>Significant Tests: {{ summary_stats.significant_tests }} ({{ "%.1f"|format(summary_stats.test_significance_rate * 100) }}%)</li>
            <li>Mean Effect Size: {{ "%.3f"|format(summary_stats.mean_effect_size) }}</li>
            <li>Median Effect Size: {{ "%.3f"|format(summary_stats.median_effect_size) }}</li>
        </ul>
    </div>
    
    {% for model_name, model_data in models.items() %}
    <div class="model-section">
        <h2>Model: {{ model_name }}</h2>
        
        <h3>Overall Performance</h3>
        <ul>
            {% for metric, value in model_data.overall_performance.items() %}
            <li>{{ metric }}: {{ "%.3f"|format(value) }}</li>
            {% endfor %}
        </ul>
        
        <h3>Robustness Metrics</h3>
        <ul>
            {% for metric, value in model_data.robustness_metrics.items() %}
            <li>{{ metric }}: {{ "%.3f"|format(value) }}</li>
            {% endfor %}
        </ul>
        
        {% for pack_name, pack_data in model_data.pack_results.items() %}
        <div class="pack-section">
            <h4>Pack: {{ pack_name }} ({{ pack_data.category }})</h4>
            
            <h5>Primary Endpoints</h5>
            <ul>
                {% for endpoint, result in pack_data.primary_endpoints.items() %}
                <li>{{ endpoint }}: {{ result|safe }}</li>
                {% endfor %}
            </ul>
            
            <h5>Secondary Endpoints</h5>
            <ul>
                {% for endpoint, result in pack_data.secondary_endpoints.items() %}
                <li>{{ endpoint }}: {{ result|safe }}</li>
                {% endfor %}
            </ul>
            
            <h5>Off-target Metrics</h5>
            <ul>
                {% for metric, value in pack_data.off_target_metrics.items() %}
                <li>{{ metric }}: {{ "%.3f"|format(value) }}</li>
                {% endfor %}
            </ul>
            
            {% if pack_data.safety_violations %}
            <h5>Safety Violations</h5>
            <ul>
                {% for violation in pack_data.safety_violations %}
                <li style="color: red;">{{ violation }}</li>
                {% endfor %}
            </ul>
            {% endif %}
            
            <p><strong>Overall Significance:</strong> {{ "Yes" if pack_data.overall_significance else "No" }}</p>
        </div>
        {% endfor %}
    </div>
    {% endfor %}
</body>
</html>
"""
    
    def _get_json_summary_template(self) -> str:
        """Get JSON summary template"""
        return """
{
    "timestamp": "{{ timestamp }}",
    "summary": {{ summary_stats|tojson }},
    "models": {{ models|tojson }}
}
"""


def create_sample_results() -> Dict[str, ModelResult]:
    """Create sample results for testing"""
    # Create sample statistical results
    caffeine_pdq = StatisticalResult(
        test_name="paired_t_test",
        pack_name="caffeine",
        endpoint="PDQ-S",
        statistic=4.25,
        p_value=0.0001,
        p_value_adjusted=0.0003,
        effect_size=0.45,
        effect_size_type="cohens_d",
        ci_lower=0.38,
        ci_upper=0.52,
        n_samples=80,
        significant=True
    )
    
    caffeine_sdq = StatisticalResult(
        test_name="paired_t_test",
        pack_name="caffeine",
        endpoint="SDQ",
        statistic=3.12,
        p_value=0.002,
        p_value_adjusted=0.006,
        effect_size=0.28,
        effect_size_type="cohens_d",
        ci_lower=0.21,
        ci_upper=0.35,
        n_samples=80,
        significant=True
    )
    
    # Create pack result
    caffeine_result = PackResult(
        pack_name="caffeine",
        category="stimulants",
        primary_endpoints={"PDQ-S": caffeine_pdq, "SDQ": caffeine_sdq},
        secondary_endpoints={},
        off_target_metrics={"refusal_rate": 0.01, "toxicity": 0.00, "verbosity": 0.05},
        safety_violations=[],
        overall_significance=True
    )
    
    # Create model result
    llama_result = ModelResult(
        model_name="Llama-3.1-70B",
        pack_results={"caffeine": caffeine_result},
        overall_performance={"accuracy": 0.85, "precision": 0.82, "recall": 0.88},
        robustness_metrics={"cross_model_consistency": 0.78, "paraphrase_stability": 0.82}
    )
    
    return {"Llama-3.1-70B": llama_result}


def main():
    """Main function to demonstrate the results template system"""
    generator = ResultsTemplateGenerator()
    
    # Create sample results
    sample_results = create_sample_results()
    
    # Generate different output formats
    print("📊 Generating results templates...")
    
    # HTML report
    html_report = generator.create_html_report(sample_results, "results_report.html")
    print("✅ HTML report generated")
    
    # JSON summary
    json_summary = generator.create_json_summary(sample_results, "results_summary.json")
    print("✅ JSON summary generated")
    
    # CSV export
    csv_data = generator.export_results_csv(sample_results, "results_data.csv")
    print("✅ CSV data exported")
    
    print(f"\n🎉 Results template system complete!")
    print(f"📂 Output directory: {generator.output_dir}")


if __name__ == "__main__":
    main()
