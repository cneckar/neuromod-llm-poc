#!/usr/bin/env python3
"""
Automated Reporting System

This module generates comprehensive reports for:
- Statistical analysis results
- Robustness validation
- Off-target monitoring
- Overall study summary
- Publication-ready tables and figures
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

@dataclass
class ReportConfig:
    """Configuration for report generation"""
    output_dir: str = "reports"
    include_plots: bool = True
    include_tables: bool = True
    include_raw_data: bool = False
    format: str = "html"  # html, pdf, markdown
    template: Optional[str] = None

@dataclass
class ReportSection:
    """A section of the report"""
    title: str
    content: str
    data: Optional[Dict[str, Any]] = None
    plots: Optional[List[str]] = None

class ReportGenerator:
    """Generates comprehensive reports for the neuromodulation study"""
    
    def __init__(self, project_root: str = ".", config: ReportConfig = None):
        self.project_root = Path(project_root)
        self.config = config or ReportConfig()
        self.reports_dir = self.project_root / self.config.output_dir
        self.reports_dir.mkdir(exist_ok=True)
        
    def generate_statistical_report(self, results_data: pd.DataFrame) -> ReportSection:
        """Generate statistical analysis report section"""
        
        # Calculate summary statistics
        n_tests = len(results_data)
        significant_tests = len(results_data[results_data.get('significant', False)])
        significant_rate = significant_tests / n_tests if n_tests > 0 else 0
        
        # Effect size summary
        effect_sizes = results_data.get('effect_size', pd.Series())
        mean_effect_size = effect_sizes.mean() if not effect_sizes.empty else 0
        median_effect_size = effect_sizes.median() if not effect_sizes.empty else 0
        
        # P-value summary
        p_values = results_data.get('p_value', pd.Series())
        mean_p_value = p_values.mean() if not p_values.empty else 1
        
        content = f"""
## Statistical Analysis Results

### Summary Statistics
- **Total Tests**: {n_tests}
- **Significant Tests**: {significant_tests} ({significant_rate:.1%})
- **Mean Effect Size**: {mean_effect_size:.3f}
- **Median Effect Size**: {median_effect_size:.3f}
- **Mean P-value**: {mean_p_value:.3f}

### Effect Size Interpretation
- **Small Effect** (d < 0.2): {len(effect_sizes[abs(effect_sizes) < 0.2])} tests
- **Medium Effect** (0.2 ≤ d < 0.5): {len(effect_sizes[(abs(effect_sizes) >= 0.2) & (abs(effect_sizes) < 0.5)])} tests
- **Large Effect** (d ≥ 0.5): {len(effect_sizes[abs(effect_sizes) >= 0.5])} tests

### Significant Results
"""
        
        # Add significant results table
        significant_data = results_data[results_data.get('significant', False)]
        if not significant_data.empty:
            content += "\n| Pack | Metric | Effect Size | P-value | Interpretation |\n"
            content += "|------|--------|-------------|---------|----------------|\n"
            
            for _, row in significant_data.iterrows():
                pack = row.get('pack', 'unknown')
                metric = row.get('metric', 'unknown')
                effect_size = row.get('effect_size', 0)
                p_value = row.get('p_value', 1)
                
                # Interpret effect size
                abs_effect = abs(effect_size)
                if abs_effect < 0.2:
                    interpretation = "negligible"
                elif abs_effect < 0.5:
                    interpretation = "small"
                elif abs_effect < 0.8:
                    interpretation = "medium"
                else:
                    interpretation = "large"
                
                content += f"| {pack} | {metric} | {effect_size:.3f} | {p_value:.3f} | {interpretation} |\n"
        else:
            content += "\nNo significant results found.\n"
        
        return ReportSection(
            title="Statistical Analysis",
            content=content,
            data={
                "n_tests": n_tests,
                "significant_tests": significant_tests,
                "significant_rate": significant_rate,
                "mean_effect_size": mean_effect_size,
                "median_effect_size": median_effect_size
            }
        )
    
    def generate_robustness_report(self, robustness_data: Dict[str, Any]) -> ReportSection:
        """Generate robustness validation report section"""
        
        robustness_score = robustness_data.get('robustness_score', 0)
        generalization_score = robustness_data.get('generalization_score', 0)
        overall_robust = robustness_data.get('overall_robust', False)
        n_models = len(robustness_data.get('model_results', []))
        
        content = f"""
## Robustness and Generalization

### Overall Scores
- **Robustness Score**: {robustness_score:.3f}
- **Generalization Score**: {generalization_score:.3f}
- **Overall Robust**: {'YES' if overall_robust else 'NO'}
- **Models Tested**: {n_models}

### Interpretation
- **Robustness Score ≥ 0.7**: {'✅ PASS' if robustness_score >= 0.7 else '❌ FAIL'}
- **Generalization Score ≥ 0.6**: {'✅ PASS' if generalization_score >= 0.6 else '❌ FAIL'}
- **Multiple Models**: {'✅ PASS' if n_models >= 2 else '❌ FAIL'}

### Model-Specific Results
"""
        
        # Add model-specific results
        model_results = robustness_data.get('model_results', [])
        for model_result in model_results:
            model_name = model_result.get('model_name', 'unknown')
            model_type = model_result.get('model_type', 'unknown')
            n_items = model_result.get('n_items', 0)
            significant_effects = model_result.get('significant_effects', [])
            
            content += f"\n#### {model_name} ({model_type})\n"
            content += f"- **Items**: {n_items}\n"
            content += f"- **Significant Effects**: {len(significant_effects)}\n"
            if significant_effects:
                content += f"- **Effects**: {', '.join(significant_effects[:5])}\n"
                if len(significant_effects) > 5:
                    content += f"  - ... and {len(significant_effects) - 5} more\n"
        
        return ReportSection(
            title="Robustness Validation",
            content=content,
            data={
                "robustness_score": robustness_score,
                "generalization_score": generalization_score,
                "overall_robust": overall_robust,
                "n_models": n_models
            }
        )
    
    def generate_off_target_report(self, off_target_data: Dict[str, Any]) -> ReportSection:
        """Generate off-target monitoring report section"""
        
        status = off_target_data.get('status', 'unknown')
        total_measurements = off_target_data.get('total_measurements', 0)
        baseline_available = off_target_data.get('baseline_available', False)
        
        content = f"""
## Off-Target Monitoring

### Status
- **Overall Status**: {status.upper()}
- **Total Measurements**: {total_measurements}
- **Baseline Available**: {'YES' if baseline_available else 'NO'}

### Safety Bands
"""
        
        # Add safety check details
        safety_check = off_target_data.get('safety_check', {})
        within_bands = safety_check.get('within_bands', False)
        violations = safety_check.get('violations', [])
        
        content += f"- **Within Safety Bands**: {'✅ YES' if within_bands else '❌ NO'}\n"
        
        if violations:
            content += "\n### Safety Violations\n"
            for violation in violations:
                content += f"- ⚠️ {violation}\n"
        else:
            content += "\n### Safety Status\n- ✅ No safety violations detected\n"
        
        # Add latest metrics if available
        latest_metrics = off_target_data.get('latest_metrics', {})
        if latest_metrics:
            content += "\n### Latest Metrics\n"
            content += f"- **Refusal Rate**: {latest_metrics.get('refusal_rate', 0):.3f}\n"
            content += f"- **Toxicity Score**: {latest_metrics.get('toxicity_score', 0):.3f}\n"
            content += f"- **Verbosity (tokens)**: {latest_metrics.get('verbosity_tokens', 0)}\n"
            content += f"- **Coherence Score**: {latest_metrics.get('coherence_score', 0):.3f}\n"
        
        return ReportSection(
            title="Off-Target Monitoring",
            content=content,
            data={
                "status": status,
                "within_bands": within_bands,
                "violations": violations,
                "total_measurements": total_measurements
            }
        )
    
    def generate_study_summary(self, 
                              statistical_data: pd.DataFrame,
                              robustness_data: Dict[str, Any],
                              off_target_data: Dict[str, Any]) -> ReportSection:
        """Generate overall study summary"""
        
        # Statistical summary
        n_tests = len(statistical_data)
        significant_tests = len(statistical_data[statistical_data.get('significant', False)])
        significant_rate = significant_tests / n_tests if n_tests > 0 else 0
        
        # Robustness summary
        robustness_score = robustness_data.get('robustness_score', 0)
        overall_robust = robustness_data.get('overall_robust', False)
        
        # Off-target summary
        off_target_status = off_target_data.get('status', 'unknown')
        within_bands = off_target_data.get('safety_check', {}).get('within_bands', False)
        
        # Overall assessment
        overall_success = (significant_rate > 0.1 and 
                          robustness_score >= 0.7 and 
                          within_bands)
        
        content = f"""
## Study Summary

### Overall Assessment
- **Study Status**: {'✅ SUCCESS' if overall_success else '⚠️ PARTIAL SUCCESS' if significant_rate > 0.05 else '❌ INCONCLUSIVE'}
- **Statistical Power**: {significant_rate:.1%} of tests significant
- **Robustness**: {'✅ ROBUST' if overall_robust else '⚠️ PARTIAL' if robustness_score >= 0.5 else '❌ NOT ROBUST'}
- **Safety**: {'✅ SAFE' if within_bands else '⚠️ SAFETY CONCERNS'}

### Key Findings
- **Total Tests Performed**: {n_tests}
- **Significant Effects**: {significant_tests}
- **Robustness Score**: {robustness_score:.3f}
- **Off-Target Status**: {off_target_status}

### Recommendations
"""
        
        if overall_success:
            content += "- ✅ Study demonstrates significant neuromodulation effects\n"
            content += "- ✅ Results are robust across conditions\n"
            content += "- ✅ No safety concerns detected\n"
            content += "- 📝 Ready for publication\n"
        elif significant_rate > 0.05:
            content += "- ⚠️ Some significant effects detected\n"
            content += "- 🔍 Consider increasing sample size\n"
            content += "- 📊 Review robustness across models\n"
        else:
            content += "- ❌ No significant effects detected\n"
            content += "- 🔍 Check experimental design\n"
            content += "- 📈 Consider larger effect sizes or sample size\n"
        
        return ReportSection(
            title="Study Summary",
            content=content,
            data={
                "overall_success": overall_success,
                "significant_rate": significant_rate,
                "robustness_score": robustness_score,
                "within_bands": within_bands
            }
        )
    
    def generate_html_report(self, sections: List[ReportSection], report_id: str) -> str:
        """Generate HTML report"""
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Neuromodulation Study Report - {report_id}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
        h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        h3 {{ color: #7f8c8d; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .success {{ color: #27ae60; }}
        .warning {{ color: #f39c12; }}
        .error {{ color: #e74c3c; }}
        .summary-box {{ background-color: #ecf0f1; padding: 20px; border-radius: 5px; margin: 20px 0; }}
        code {{ background-color: #f8f9fa; padding: 2px 4px; border-radius: 3px; }}
    </style>
</head>
<body>
    <h1>Neuromodulation Study Report</h1>
    <p><strong>Report ID:</strong> {report_id}</p>
    <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    
"""
        
        for section in sections:
            html_content += f"    <h2>{section.title}</h2>\n"
            html_content += f"    {section.content}\n\n"
        
        html_content += """
</body>
</html>
"""
        
        return html_content
    
    def generate_markdown_report(self, sections: List[ReportSection], report_id: str) -> str:
        """Generate Markdown report"""
        
        markdown_content = f"""# Neuromodulation Study Report

**Report ID:** {report_id}  
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

"""
        
        for section in sections:
            markdown_content += f"## {section.title}\n\n"
            markdown_content += f"{section.content}\n\n"
        
        return markdown_content
    
    def generate_comprehensive_report(self,
                                    statistical_data: pd.DataFrame,
                                    robustness_data: Dict[str, Any],
                                    off_target_data: Dict[str, Any],
                                    report_id: str = None) -> str:
        """Generate comprehensive study report"""
        
        if report_id is None:
            report_id = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Generate sections
        sections = [
            self.generate_study_summary(statistical_data, robustness_data, off_target_data),
            self.generate_statistical_report(statistical_data),
            self.generate_robustness_report(robustness_data),
            self.generate_off_target_report(off_target_data)
        ]
        
        # Generate report based on format
        if self.config.format == "html":
            report_content = self.generate_html_report(sections, report_id)
            file_extension = "html"
        else:  # markdown
            report_content = self.generate_markdown_report(sections, report_id)
            file_extension = "md"
        
        # Save report
        report_file = self.reports_dir / f"{report_id}.{file_extension}"
        with open(report_file, 'w') as f:
            f.write(report_content)
        
        logger.info(f"Report generated: {report_file}")
        return str(report_file)

def main():
    """Example usage of report generator"""
    import pandas as pd
    
    # Create sample data
    statistical_data = pd.DataFrame([
        {'pack': 'caffeine', 'metric': 'adq_score', 'effect_size': 0.5, 'p_value': 0.01, 'significant': True},
        {'pack': 'caffeine', 'metric': 'pdq_score', 'effect_size': 0.3, 'p_value': 0.05, 'significant': True},
        {'pack': 'lsd', 'metric': 'adq_score', 'effect_size': 0.1, 'p_value': 0.2, 'significant': False},
        {'pack': 'alcohol', 'metric': 'sdq_score', 'effect_size': 0.4, 'p_value': 0.02, 'significant': True},
    ])
    
    robustness_data = {
        'robustness_score': 0.85,
        'generalization_score': 0.75,
        'overall_robust': True,
        'model_results': [
            {'model_name': 'llama-3.1-70b', 'model_type': 'open', 'n_items': 50, 'significant_effects': ['caffeine_adq', 'alcohol_sdq']},
            {'model_name': 'qwen-2.5-omni-7b', 'model_type': 'open', 'n_items': 50, 'significant_effects': ['caffeine_adq', 'caffeine_pdq']}
        ]
    }
    
    off_target_data = {
        'status': 'within_bands',
        'total_measurements': 100,
        'baseline_available': True,
        'safety_check': {
            'within_bands': True,
            'violations': []
        },
        'latest_metrics': {
            'refusal_rate': 0.02,
            'toxicity_score': 0.01,
            'verbosity_tokens': 150,
            'coherence_score': 0.85
        }
    }
    
    # Generate report
    config = ReportConfig(format="html")
    generator = ReportGenerator(config=config)
    
    report_file = generator.generate_comprehensive_report(
        statistical_data, robustness_data, off_target_data
    )
    
    print(f"Report generated: {report_file}")

if __name__ == "__main__":
    main()
