"""
Safety and Ethics Review System

This module implements comprehensive safety and ethics review for neuromodulation
research, ensuring responsible AI development and ethical research practices.

Key Features:
- Safety risk assessment
- Ethics compliance validation
- Bias and fairness monitoring
- Harm prevention measures
- Ethical guidelines enforcement
- Safety reporting and alerts
"""

import json
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import pandas as pd
import numpy as np
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    """Risk level classification"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class EthicsCompliance(Enum):
    """Ethics compliance status"""
    COMPLIANT = "compliant"
    MINOR_ISSUES = "minor_issues"
    MAJOR_ISSUES = "major_issues"
    NON_COMPLIANT = "non_compliant"

@dataclass
class SafetyRisk:
    """Individual safety risk assessment"""
    risk_id: str
    risk_type: str  # 'bias', 'harm', 'misuse', 'privacy', 'security'
    description: str
    risk_level: RiskLevel
    probability: float  # 0.0 to 1.0
    impact: float  # 0.0 to 1.0
    mitigation_measures: List[str]
    monitoring_required: bool
    escalation_threshold: float

@dataclass
class EthicsIssue:
    """Ethics compliance issue"""
    issue_id: str
    issue_type: str  # 'informed_consent', 'data_privacy', 'bias_fairness', 'transparency', 'accountability'
    description: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    affected_populations: List[str]
    recommended_actions: List[str]
    compliance_status: EthicsCompliance
    resolution_deadline: Optional[str]

@dataclass
class SafetyReport:
    """Comprehensive safety report"""
    report_id: str
    generated_at: str
    overall_risk_level: RiskLevel
    total_risks: int
    critical_risks: int
    high_risks: int
    medium_risks: int
    low_risks: int
    safety_score: float  # 0.0 to 1.0
    recommendations: List[str]
    monitoring_alerts: List[str]

@dataclass
class EthicsReport:
    """Comprehensive ethics report"""
    report_id: str
    generated_at: str
    overall_compliance: EthicsCompliance
    total_issues: int
    critical_issues: int
    major_issues: int
    minor_issues: int
    compliant_areas: int
    ethics_score: float  # 0.0 to 1.0
    recommendations: List[str]
    action_items: List[str]

class SafetyEthicsReviewer:
    """Main class for safety and ethics review"""
    
    def __init__(self, project_root: str = "/Users/cris/src/neuromod-llm-poc"):
        self.project_root = Path(project_root)
        self.reports_dir = self.project_root / "analysis" / "safety_ethics"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize tracking
        self.safety_risks: List[SafetyRisk] = []
        self.ethics_issues: List[EthicsIssue] = []
        self.safety_reports: List[SafetyReport] = []
        self.ethics_reports: List[EthicsReport] = []
        
        # Load safety and ethics guidelines
        self.safety_guidelines = self._load_safety_guidelines()
        self.ethics_guidelines = self._load_ethics_guidelines()
    
    def conduct_safety_review(self, 
                            pack_configs: Dict[str, Dict],
                            test_results: Dict[str, Any],
                            model_info: Dict[str, Any]) -> SafetyReport:
        """
        Conduct comprehensive safety review
        
        Args:
            pack_configs: Neuromodulation pack configurations
            test_results: Test results and metrics
            model_info: Model information and capabilities
            
        Returns:
            Comprehensive safety report
        """
        logger.info("Conducting safety review...")
        
        # Identify safety risks
        risks = self._identify_safety_risks(pack_configs, test_results, model_info)
        self.safety_risks.extend(risks)
        
        # Assess risk levels
        risk_counts = self._count_risks_by_level(risks)
        
        # Calculate safety score
        safety_score = self._calculate_safety_score(risks)
        
        # Generate recommendations
        recommendations = self._generate_safety_recommendations(risks)
        
        # Generate monitoring alerts
        alerts = self._generate_safety_alerts(risks)
        
        # Create safety report
        report = SafetyReport(
            report_id=f"safety_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            generated_at=datetime.now().isoformat(),
            overall_risk_level=self._determine_overall_risk_level(risks),
            total_risks=len(risks),
            critical_risks=risk_counts.get(RiskLevel.CRITICAL, 0),
            high_risks=risk_counts.get(RiskLevel.HIGH, 0),
            medium_risks=risk_counts.get(RiskLevel.MEDIUM, 0),
            low_risks=risk_counts.get(RiskLevel.LOW, 0),
            safety_score=safety_score,
            recommendations=recommendations,
            monitoring_alerts=alerts
        )
        
        # Save report
        self._save_safety_report(report)
        self.safety_reports.append(report)
        
        logger.info(f"Safety review complete. Overall risk: {report.overall_risk_level.value}")
        logger.info(f"Safety score: {safety_score:.2f}")
        
        return report
    
    def conduct_ethics_review(self, 
                            study_design: Dict[str, Any],
                            data_handling: Dict[str, Any],
                            participant_info: Dict[str, Any]) -> EthicsReport:
        """
        Conduct comprehensive ethics review
        
        Args:
            study_design: Study design and methodology
            data_handling: Data collection and processing procedures
            participant_info: Participant information and consent
            
        Returns:
            Comprehensive ethics report
        """
        logger.info("Conducting ethics review...")
        
        # Identify ethics issues
        issues = self._identify_ethics_issues(study_design, data_handling, participant_info)
        self.ethics_issues.extend(issues)
        
        # Assess compliance
        compliance_counts = self._count_issues_by_severity(issues)
        
        # Calculate ethics score
        ethics_score = self._calculate_ethics_score(issues)
        
        # Generate recommendations
        recommendations = self._generate_ethics_recommendations(issues)
        
        # Generate action items
        action_items = self._generate_ethics_action_items(issues)
        
        # Create ethics report
        report = EthicsReport(
            report_id=f"ethics_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            generated_at=datetime.now().isoformat(),
            overall_compliance=self._determine_overall_compliance(issues),
            total_issues=len(issues),
            critical_issues=compliance_counts.get('critical', 0),
            major_issues=compliance_counts.get('high', 0),
            minor_issues=compliance_counts.get('medium', 0),
            compliant_areas=len(self.ethics_guidelines) - len(issues),
            ethics_score=ethics_score,
            recommendations=recommendations,
            action_items=action_items
        )
        
        # Save report
        self._save_ethics_report(report)
        self.ethics_reports.append(report)
        
        logger.info(f"Ethics review complete. Overall compliance: {report.overall_compliance.value}")
        logger.info(f"Ethics score: {ethics_score:.2f}")
        
        return report
    
    def _identify_safety_risks(self, pack_configs: Dict[str, Dict], 
                             test_results: Dict[str, Any], 
                             model_info: Dict[str, Any]) -> List[SafetyRisk]:
        """Identify potential safety risks"""
        risks = []
        
        # Check for bias risks
        bias_risks = self._assess_bias_risks(pack_configs, test_results)
        risks.extend(bias_risks)
        
        # Check for harm risks
        harm_risks = self._assess_harm_risks(pack_configs, test_results)
        risks.extend(harm_risks)
        
        # Check for misuse risks
        misuse_risks = self._assess_misuse_risks(pack_configs, model_info)
        risks.extend(misuse_risks)
        
        # Check for privacy risks
        privacy_risks = self._assess_privacy_risks(pack_configs, test_results)
        risks.extend(privacy_risks)
        
        # Check for security risks
        security_risks = self._assess_security_risks(pack_configs, model_info)
        risks.extend(security_risks)
        
        return risks
    
    def _assess_bias_risks(self, pack_configs: Dict[str, Dict], 
                          test_results: Dict[str, Any]) -> List[SafetyRisk]:
        """Assess bias-related safety risks"""
        risks = []
        
        # Check for demographic bias
        if self._has_demographic_bias(test_results):
            risks.append(SafetyRisk(
                risk_id="bias_demographic",
                risk_type="bias",
                description="Potential demographic bias in test results",
                risk_level=RiskLevel.MEDIUM,
                probability=0.7,
                impact=0.6,
                mitigation_measures=[
                    "Implement demographic parity testing",
                    "Use diverse test datasets",
                    "Apply bias correction techniques"
                ],
                monitoring_required=True,
                escalation_threshold=0.8
            ))
        
        # Check for cultural bias
        if self._has_cultural_bias(pack_configs):
            risks.append(SafetyRisk(
                risk_id="bias_cultural",
                risk_type="bias",
                description="Potential cultural bias in neuromodulation effects",
                risk_level=RiskLevel.MEDIUM,
                probability=0.5,
                impact=0.5,
                mitigation_measures=[
                    "Use culturally diverse training data",
                    "Implement cultural sensitivity checks",
                    "Include diverse cultural perspectives in validation"
                ],
                monitoring_required=True,
                escalation_threshold=0.7
            ))
        
        return risks
    
    def _assess_harm_risks(self, pack_configs: Dict[str, Dict], 
                          test_results: Dict[str, Any]) -> List[SafetyRisk]:
        """Assess harm-related safety risks"""
        risks = []
        
        # Check for psychological harm
        if self._has_psychological_harm_risk(pack_configs):
            risks.append(SafetyRisk(
                risk_id="harm_psychological",
                risk_type="harm",
                description="Potential psychological harm from neuromodulation effects",
                risk_level=RiskLevel.HIGH,
                probability=0.3,
                impact=0.8,
                mitigation_measures=[
                    "Implement psychological safety checks",
                    "Provide clear warnings about effects",
                    "Include mental health monitoring"
                ],
                monitoring_required=True,
                escalation_threshold=0.5
            ))
        
        # Check for cognitive impairment
        if self._has_cognitive_impairment_risk(test_results):
            risks.append(SafetyRisk(
                risk_id="harm_cognitive",
                risk_type="harm",
                description="Potential cognitive impairment from effects",
                risk_level=RiskLevel.MEDIUM,
                probability=0.4,
                impact=0.6,
                mitigation_measures=[
                    "Monitor cognitive performance metrics",
                    "Implement safety thresholds",
                    "Provide cognitive assessment tools"
                ],
                monitoring_required=True,
                escalation_threshold=0.6
            ))
        
        return risks
    
    def _assess_misuse_risks(self, pack_configs: Dict[str, Dict], 
                            model_info: Dict[str, Any]) -> List[SafetyRisk]:
        """Assess misuse-related safety risks"""
        risks = []
        
        # Check for malicious use
        if self._has_malicious_use_risk(pack_configs):
            risks.append(SafetyRisk(
                risk_id="misuse_malicious",
                risk_type="misuse",
                description="Potential for malicious use of neuromodulation effects",
                risk_level=RiskLevel.CRITICAL,
                probability=0.2,
                impact=0.9,
                mitigation_measures=[
                    "Implement access controls",
                    "Add usage monitoring",
                    "Require ethical use agreements"
                ],
                monitoring_required=True,
                escalation_threshold=0.3
            ))
        
        # Check for unintended consequences
        if self._has_unintended_consequences_risk(pack_configs):
            risks.append(SafetyRisk(
                risk_id="misuse_unintended",
                risk_type="misuse",
                description="Potential for unintended consequences from effects",
                risk_level=RiskLevel.HIGH,
                probability=0.4,
                impact=0.7,
                mitigation_measures=[
                    "Implement comprehensive testing",
                    "Add safety guards",
                    "Monitor for unexpected behaviors"
                ],
                monitoring_required=True,
                escalation_threshold=0.5
            ))
        
        return risks
    
    def _assess_privacy_risks(self, pack_configs: Dict[str, Dict], 
                             test_results: Dict[str, Any]) -> List[SafetyRisk]:
        """Assess privacy-related safety risks"""
        risks = []
        
        # Check for data privacy
        if self._has_data_privacy_risk(test_results):
            risks.append(SafetyRisk(
                risk_id="privacy_data",
                risk_type="privacy",
                description="Potential data privacy violations",
                risk_level=RiskLevel.MEDIUM,
                probability=0.3,
                impact=0.6,
                mitigation_measures=[
                    "Implement data anonymization",
                    "Add privacy-preserving techniques",
                    "Ensure GDPR compliance"
                ],
                monitoring_required=True,
                escalation_threshold=0.4
            ))
        
        return risks
    
    def _assess_security_risks(self, pack_configs: Dict[str, Dict], 
                              model_info: Dict[str, Any]) -> List[SafetyRisk]:
        """Assess security-related safety risks"""
        risks = []
        
        # Check for model security
        if self._has_model_security_risk(model_info):
            risks.append(SafetyRisk(
                risk_id="security_model",
                risk_type="security",
                description="Potential model security vulnerabilities",
                risk_level=RiskLevel.HIGH,
                probability=0.2,
                impact=0.8,
                mitigation_measures=[
                    "Implement model security checks",
                    "Add input validation",
                    "Monitor for adversarial attacks"
                ],
                monitoring_required=True,
                escalation_threshold=0.3
            ))
        
        return risks
    
    def _identify_ethics_issues(self, study_design: Dict[str, Any], 
                               data_handling: Dict[str, Any], 
                               participant_info: Dict[str, Any]) -> List[EthicsIssue]:
        """Identify ethics compliance issues"""
        issues = []
        
        # Check informed consent
        consent_issues = self._assess_informed_consent(study_design, participant_info)
        issues.extend(consent_issues)
        
        # Check data privacy
        privacy_issues = self._assess_data_privacy(data_handling)
        issues.extend(privacy_issues)
        
        # Check bias and fairness
        bias_issues = self._assess_bias_fairness(study_design)
        issues.extend(bias_issues)
        
        # Check transparency
        transparency_issues = self._assess_transparency(study_design)
        issues.extend(transparency_issues)
        
        # Check accountability
        accountability_issues = self._assess_accountability(study_design)
        issues.extend(accountability_issues)
        
        return issues
    
    def _assess_informed_consent(self, study_design: Dict[str, Any], 
                                participant_info: Dict[str, Any]) -> List[EthicsIssue]:
        """Assess informed consent compliance"""
        issues = []
        
        # Check if consent is properly documented
        if not self._has_proper_consent_documentation(participant_info):
            issues.append(EthicsIssue(
                issue_id="consent_documentation",
                issue_type="informed_consent",
                description="Insufficient informed consent documentation",
                severity="high",
                affected_populations=["all_participants"],
                recommended_actions=[
                    "Create comprehensive consent forms",
                    "Ensure clear explanation of risks and benefits",
                    "Implement consent tracking system"
                ],
                compliance_status=EthicsCompliance.MAJOR_ISSUES,
                resolution_deadline="2024-02-01"
            ))
        
        return issues
    
    def _assess_data_privacy(self, data_handling: Dict[str, Any]) -> List[EthicsIssue]:
        """Assess data privacy compliance"""
        issues = []
        
        # Check data anonymization
        if not self._has_proper_data_anonymization(data_handling):
            issues.append(EthicsIssue(
                issue_id="data_anonymization",
                issue_type="data_privacy",
                description="Insufficient data anonymization measures",
                severity="critical",
                affected_populations=["all_participants"],
                recommended_actions=[
                    "Implement data anonymization protocols",
                    "Remove personally identifiable information",
                    "Add data privacy safeguards"
                ],
                compliance_status=EthicsCompliance.NON_COMPLIANT,
                resolution_deadline="2024-01-15"
            ))
        
        return issues
    
    def _assess_bias_fairness(self, study_design: Dict[str, Any]) -> List[EthicsIssue]:
        """Assess bias and fairness compliance"""
        issues = []
        
        # Check for bias in study design
        if self._has_bias_in_study_design(study_design):
            issues.append(EthicsIssue(
                issue_id="study_bias",
                issue_type="bias_fairness",
                description="Potential bias in study design",
                severity="medium",
                affected_populations=["underrepresented_groups"],
                recommended_actions=[
                    "Ensure diverse participant recruitment",
                    "Implement bias detection measures",
                    "Add fairness validation checks"
                ],
                compliance_status=EthicsCompliance.MINOR_ISSUES,
                resolution_deadline="2024-02-15"
            ))
        
        return issues
    
    def _assess_transparency(self, study_design: Dict[str, Any]) -> List[EthicsIssue]:
        """Assess transparency compliance"""
        issues = []
        
        # Check for transparency in methodology
        if not self._has_transparent_methodology(study_design):
            issues.append(EthicsIssue(
                issue_id="methodology_transparency",
                issue_type="transparency",
                description="Insufficient transparency in methodology",
                severity="medium",
                affected_populations=["research_community"],
                recommended_actions=[
                    "Document all methodology details",
                    "Provide open access to code and data",
                    "Implement reproducibility measures"
                ],
                compliance_status=EthicsCompliance.MINOR_ISSUES,
                resolution_deadline="2024-02-28"
            ))
        
        return issues
    
    def _assess_accountability(self, study_design: Dict[str, Any]) -> List[EthicsIssue]:
        """Assess accountability compliance"""
        issues = []
        
        # Check for accountability measures
        if not self._has_accountability_measures(study_design):
            issues.append(EthicsIssue(
                issue_id="accountability_measures",
                issue_type="accountability",
                description="Insufficient accountability measures",
                severity="medium",
                affected_populations=["all_stakeholders"],
                recommended_actions=[
                    "Implement accountability frameworks",
                    "Add oversight mechanisms",
                    "Create reporting systems"
                ],
                compliance_status=EthicsCompliance.MINOR_ISSUES,
                resolution_deadline="2024-03-01"
            ))
        
        return issues
    
    def _load_safety_guidelines(self) -> Dict[str, Any]:
        """Load safety guidelines"""
        return {
            "bias_threshold": 0.1,
            "harm_threshold": 0.3,
            "misuse_threshold": 0.2,
            "privacy_threshold": 0.4,
            "security_threshold": 0.3
        }
    
    def _load_ethics_guidelines(self) -> Dict[str, Any]:
        """Load ethics guidelines"""
        return {
            "informed_consent": "required",
            "data_privacy": "required",
            "bias_fairness": "required",
            "transparency": "required",
            "accountability": "required"
        }
    
    def _has_demographic_bias(self, test_results: Dict[str, Any]) -> bool:
        """Check for demographic bias in test results"""
        # Mock implementation - in practice, would analyze actual results
        return False
    
    def _has_cultural_bias(self, pack_configs: Dict[str, Dict]) -> bool:
        """Check for cultural bias in pack configurations"""
        # Mock implementation - in practice, would analyze pack contents
        return False
    
    def _has_psychological_harm_risk(self, pack_configs: Dict[str, Dict]) -> bool:
        """Check for psychological harm risk"""
        # Mock implementation - in practice, would analyze effect types
        return False
    
    def _has_cognitive_impairment_risk(self, test_results: Dict[str, Any]) -> bool:
        """Check for cognitive impairment risk"""
        # Mock implementation - in practice, would analyze performance metrics
        return False
    
    def _has_malicious_use_risk(self, pack_configs: Dict[str, Dict]) -> bool:
        """Check for malicious use risk"""
        # Mock implementation - in practice, would analyze effect capabilities
        return False
    
    def _has_unintended_consequences_risk(self, pack_configs: Dict[str, Dict]) -> bool:
        """Check for unintended consequences risk"""
        # Mock implementation - in practice, would analyze effect interactions
        return False
    
    def _has_data_privacy_risk(self, test_results: Dict[str, Any]) -> bool:
        """Check for data privacy risk"""
        # Mock implementation - in practice, would analyze data handling
        return False
    
    def _has_model_security_risk(self, model_info: Dict[str, Any]) -> bool:
        """Check for model security risk"""
        # Mock implementation - in practice, would analyze model security
        return False
    
    def _has_proper_consent_documentation(self, participant_info: Dict[str, Any]) -> bool:
        """Check for proper consent documentation"""
        # Mock implementation - in practice, would check actual documentation
        return True
    
    def _has_proper_data_anonymization(self, data_handling: Dict[str, Any]) -> bool:
        """Check for proper data anonymization"""
        # Mock implementation - in practice, would check anonymization measures
        return True
    
    def _has_bias_in_study_design(self, study_design: Dict[str, Any]) -> bool:
        """Check for bias in study design"""
        # Mock implementation - in practice, would analyze study design
        return False
    
    def _has_transparent_methodology(self, study_design: Dict[str, Any]) -> bool:
        """Check for transparent methodology"""
        # Mock implementation - in practice, would check methodology documentation
        return True
    
    def _has_accountability_measures(self, study_design: Dict[str, Any]) -> bool:
        """Check for accountability measures"""
        # Mock implementation - in practice, would check accountability frameworks
        return True
    
    def _count_risks_by_level(self, risks: List[SafetyRisk]) -> Dict[RiskLevel, int]:
        """Count risks by level"""
        counts = {level: 0 for level in RiskLevel}
        for risk in risks:
            counts[risk.risk_level] += 1
        return counts
    
    def _count_issues_by_severity(self, issues: List[EthicsIssue]) -> Dict[str, int]:
        """Count issues by severity"""
        counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        for issue in issues:
            counts[issue.severity] += 1
        return counts
    
    def _calculate_safety_score(self, risks: List[SafetyRisk]) -> float:
        """Calculate overall safety score"""
        if not risks:
            return 1.0
        
        # Weight by risk level
        weights = {
            RiskLevel.CRITICAL: 0.0,
            RiskLevel.HIGH: 0.3,
            RiskLevel.MEDIUM: 0.6,
            RiskLevel.LOW: 0.9
        }
        
        total_weight = sum(weights[risk.risk_level] for risk in risks)
        return total_weight / len(risks)
    
    def _calculate_ethics_score(self, issues: List[EthicsIssue]) -> float:
        """Calculate overall ethics score"""
        if not issues:
            return 1.0
        
        # Weight by severity
        weights = {
            "critical": 0.0,
            "high": 0.2,
            "medium": 0.6,
            "low": 0.9
        }
        
        total_weight = sum(weights[issue.severity] for issue in issues)
        return total_weight / len(issues)
    
    def _determine_overall_risk_level(self, risks: List[SafetyRisk]) -> RiskLevel:
        """Determine overall risk level"""
        if not risks:
            return RiskLevel.LOW
        
        # Check for critical risks
        if any(risk.risk_level == RiskLevel.CRITICAL for risk in risks):
            return RiskLevel.CRITICAL
        
        # Check for high risks
        if any(risk.risk_level == RiskLevel.HIGH for risk in risks):
            return RiskLevel.HIGH
        
        # Check for medium risks
        if any(risk.risk_level == RiskLevel.MEDIUM for risk in risks):
            return RiskLevel.MEDIUM
        
        return RiskLevel.LOW
    
    def _determine_overall_compliance(self, issues: List[EthicsIssue]) -> EthicsCompliance:
        """Determine overall compliance status"""
        if not issues:
            return EthicsCompliance.COMPLIANT
        
        # Check for critical issues
        if any(issue.severity == "critical" for issue in issues):
            return EthicsCompliance.NON_COMPLIANT
        
        # Check for major issues
        if any(issue.severity == "high" for issue in issues):
            return EthicsCompliance.MAJOR_ISSUES
        
        # Check for minor issues
        if any(issue.severity == "medium" for issue in issues):
            return EthicsCompliance.MINOR_ISSUES
        
        return EthicsCompliance.COMPLIANT
    
    def _generate_safety_recommendations(self, risks: List[SafetyRisk]) -> List[str]:
        """Generate safety recommendations"""
        recommendations = []
        
        for risk in risks:
            if risk.risk_level in [RiskLevel.CRITICAL, RiskLevel.HIGH]:
                recommendations.extend(risk.mitigation_measures)
        
        return list(set(recommendations))  # Remove duplicates
    
    def _generate_safety_alerts(self, risks: List[SafetyRisk]) -> List[str]:
        """Generate safety monitoring alerts"""
        alerts = []
        
        for risk in risks:
            if risk.monitoring_required:
                alerts.append(f"Monitor {risk.risk_type} risk: {risk.description}")
        
        return alerts
    
    def _generate_ethics_recommendations(self, issues: List[EthicsIssue]) -> List[str]:
        """Generate ethics recommendations"""
        recommendations = []
        
        for issue in issues:
            if issue.severity in ["critical", "high"]:
                recommendations.extend(issue.recommended_actions)
        
        return list(set(recommendations))  # Remove duplicates
    
    def _generate_ethics_action_items(self, issues: List[EthicsIssue]) -> List[str]:
        """Generate ethics action items"""
        action_items = []
        
        for issue in issues:
            if issue.resolution_deadline:
                action_items.append(f"Resolve {issue.issue_type} by {issue.resolution_deadline}")
        
        return action_items
    
    def _save_safety_report(self, report: SafetyReport):
        """Save safety report to file"""
        report_path = self.reports_dir / f"{report.report_id}.json"
        with open(report_path, 'w') as f:
            json.dump(asdict(report), f, indent=2, default=str)
    
    def _save_ethics_report(self, report: EthicsReport):
        """Save ethics report to file"""
        report_path = self.reports_dir / f"{report.report_id}.json"
        with open(report_path, 'w') as f:
            json.dump(asdict(report), f, indent=2, default=str)

def main():
    """Example usage of the safety ethics reviewer"""
    reviewer = SafetyEthicsReviewer()
    
    # Mock data for testing
    pack_configs = {
        "caffeine": {"name": "caffeine", "effects": []},
        "lsd": {"name": "lsd", "effects": []}
    }
    
    test_results = {
        "caffeine_score": 0.75,
        "lsd_score": 0.68
    }
    
    model_info = {
        "model_name": "llama-3.1-70b",
        "capabilities": ["text_generation", "reasoning"]
    }
    
    study_design = {
        "methodology": "randomized_controlled_trial",
        "participants": 100
    }
    
    data_handling = {
        "anonymization": True,
        "encryption": True
    }
    
    participant_info = {
        "consent_obtained": True,
        "demographics": "diverse"
    }
    
    print("Conducting safety review...")
    safety_report = reviewer.conduct_safety_review(
        pack_configs, test_results, model_info
    )
    
    print("Conducting ethics review...")
    ethics_report = reviewer.conduct_ethics_review(
        study_design, data_handling, participant_info
    )
    
    print(f"Safety review complete. Risk level: {safety_report.overall_risk_level.value}")
    print(f"Ethics review complete. Compliance: {ethics_report.overall_compliance.value}")

if __name__ == "__main__":
    main()
