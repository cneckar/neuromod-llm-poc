#!/usr/bin/env python3
"""
Human Reference Data Collection Workbook

This module provides a comprehensive workbook for collecting and managing
human reference data for neuromodulation research. It includes data entry
forms, validation procedures, and automated report generation.

Key Features:
- Interactive data entry forms
- Real-time validation and scoring
- Automated report generation
- Data quality control
- Export and backup procedures
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import json
from datetime import datetime, timedelta
import logging
from dataclasses import asdict

# Import our data collection modules
from .human_reference_worksheets import HumanReferenceDataCollector
from .signature_matching import SignatureMatcher, HumanSignature, ModelSignature

# Configure logging
logger = logging.getLogger(__name__)

class HumanReferenceWorkbook:
    """Main workbook for human reference data collection"""
    
    def __init__(self, output_dir: str = "outputs/analysis/human_reference"):
        """
        Initialize the human reference workbook
        
        Args:
            output_dir: Directory to save workbook data
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.data_collector = HumanReferenceDataCollector(str(self.output_dir))
        self.signature_matcher = SignatureMatcher(str(self.output_dir))
        
        # Study configuration
        self.study_config = self._load_study_configuration()
        
        # Data storage
        self.participants = {}
        self.sessions = {}
        self.assessments = {}
        self.signatures = {}
    
    def _load_study_configuration(self) -> Dict[str, Any]:
        """Load study configuration"""
        return {
            "study_id": "NEUROMOD_HUMAN_REF_001",
            "study_title": "Human Reference Data Collection for Neuromodulation Research",
            "protocol_version": "1.0",
            "start_date": "2024-01-01",
            "end_date": "2024-12-31",
            "target_participants": 120,
            "conditions": [
                {"name": "caffeine", "dosage": "200mg", "route": "oral"},
                {"name": "l_theanine", "dosage": "200mg", "route": "oral"},
                {"name": "modafinil", "dosage": "100mg", "route": "oral"},
                {"name": "lsd", "dosage": "100μg", "route": "oral"},
                {"name": "mdma", "dosage": "75mg", "route": "oral"},
                {"name": "thc", "dosage": "10mg", "route": "oral"},
                {"name": "placebo", "dosage": "0mg", "route": "oral"}
            ],
            "assessment_timeline": {
                "T-60": ["poms"],
                "T+0": [],
                "T+30": ["poms"],
                "T+60": ["ascs", "sdq", "poms", "caq", "digit_span", "stroop", "rat", "trail_making"],
                "T+120": ["ascs", "sdq", "poms", "caq", "digit_span", "stroop", "rat", "trail_making"],
                "T+240": ["ascs", "sdq", "poms"],
                "T+480": ["poms"]
            }
        }
    
    def register_participant(self, participant_id: str, demographics: Dict[str, Any]) -> Dict[str, Any]:
        """Register a new participant"""
        if participant_id in self.participants:
            raise ValueError(f"Participant {participant_id} already registered")
        
        # Create participant record
        participant = {
            "participant_id": participant_id,
            "registration_date": datetime.now().isoformat(),
            "demographics": demographics,
            "eligibility_status": "pending",
            "sessions_completed": 0,
            "total_sessions": len(self.study_config["conditions"]),
            "status": "active"
        }
        
        self.participants[participant_id] = participant
        
        # Save to file
        self._save_participant_data(participant_id)
        
        logger.info(f"Registered participant: {participant_id}")
        return participant
    
    def conduct_screening(self, participant_id: str, screening_data: Dict[str, Any]) -> Dict[str, Any]:
        """Conduct participant screening"""
        if participant_id not in self.participants:
            raise ValueError(f"Participant {participant_id} not found")
        
        # Update screening data
        self.participants[participant_id]["screening"] = screening_data
        self.participants[participant_id]["screening_date"] = datetime.now().isoformat()
        
        # Determine eligibility
        eligibility = self._assess_eligibility(screening_data)
        self.participants[participant_id]["eligibility_status"] = eligibility["status"]
        self.participants[participant_id]["eligibility_notes"] = eligibility["notes"]
        
        # Save updated data
        self._save_participant_data(participant_id)
        
        logger.info(f"Screening completed for {participant_id}: {eligibility['status']}")
        return eligibility
    
    def _assess_eligibility(self, screening_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess participant eligibility based on screening data"""
        eligibility_criteria = screening_data.get("eligibility_criteria", {})
        
        # Check inclusion criteria
        inclusion_checks = [
            eligibility_criteria.get("age_18_65", False),
            eligibility_criteria.get("no_psychiatric_history", False),
            eligibility_criteria.get("no_cardiovascular_conditions", False),
            eligibility_criteria.get("no_substance_dependence", False),
            eligibility_criteria.get("not_pregnant", False),
            eligibility_criteria.get("no_medication_interactions", False)
        ]
        
        # Check exclusion criteria
        exclusion_checks = [
            screening_data.get("medical_history", {}).get("psychiatric_history", []),
            screening_data.get("medical_history", {}).get("cardiovascular_conditions", []),
            screening_data.get("medical_history", {}).get("substance_dependence", [])
        ]
        
        # Determine eligibility
        if all(inclusion_checks) and not any(exclusion_checks):
            status = "eligible"
            notes = "Participant meets all eligibility criteria"
        else:
            status = "ineligible"
            notes = "Participant does not meet eligibility criteria"
        
        return {
            "status": status,
            "notes": notes,
            "inclusion_criteria_met": inclusion_checks,
            "exclusion_criteria_met": exclusion_checks
        }
    
    def schedule_session(self, participant_id: str, condition: str, 
                        session_date: str, session_time: str) -> str:
        """Schedule a study session"""
        if participant_id not in self.participants:
            raise ValueError(f"Participant {participant_id} not found")
        
        if self.participants[participant_id]["eligibility_status"] != "eligible":
            raise ValueError(f"Participant {participant_id} is not eligible")
        
        # Generate session ID
        session_id = f"{participant_id}_{condition}_{session_date.replace('-', '')}"
        
        # Create session record
        session = {
            "session_id": session_id,
            "participant_id": participant_id,
            "condition": condition,
            "session_date": session_date,
            "session_time": session_time,
            "status": "scheduled",
            "assessments_completed": 0,
            "total_assessments": 0,
            "created_date": datetime.now().isoformat()
        }
        
        self.sessions[session_id] = session
        
        # Save session data
        self._save_session_data(session_id)
        
        logger.info(f"Scheduled session: {session_id}")
        return session_id
    
    def conduct_session(self, session_id: str, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """Conduct a study session"""
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.sessions[session_id]
        
        # Update session data
        session.update(session_data)
        session["status"] = "completed"
        session["completion_date"] = datetime.now().isoformat()
        
        # Process assessments
        assessments = session_data.get("assessments", {})
        processed_assessments = {}
        
        for assessment_name, assessment_data in assessments.items():
            # Score and validate assessment
            scored_assessment = self.data_collector.score_assessment(assessment_data)
            validated_assessment = self.data_collector.validate_assessment(scored_assessment)
            
            processed_assessments[assessment_name] = validated_assessment
        
        session["assessments"] = processed_assessments
        session["assessments_completed"] = len(processed_assessments)
        
        # Create human signature
        human_signature = self._create_human_signature(session)
        self.signatures[f"{session_id}_human"] = human_signature
        
        # Save updated data
        self._save_session_data(session_id)
        self._save_signature_data(f"{session_id}_human", human_signature)
        
        logger.info(f"Completed session: {session_id}")
        return session
    
    def _create_human_signature(self, session: Dict[str, Any]) -> HumanSignature:
        """Create human signature from session data"""
        # Extract assessment scores
        assessment_scores = {}
        for assessment_name, assessment_data in session.get("assessments", {}).items():
            if "scores" in assessment_data and "subscale_scores" in assessment_data["scores"]:
                assessment_scores[assessment_name] = assessment_data["scores"]["subscale_scores"]
        
        # Create signature data
        signature_data = {
            "participant_id": session["participant_id"],
            "condition": session["condition"],
            "assessment_time": "T+60",  # Primary assessment time
            "assessments": assessment_scores
        }
        
        return self.signature_matcher.create_human_signature(signature_data)
    
    def generate_session_report(self, session_id: str) -> Dict[str, Any]:
        """Generate a session report"""
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.sessions[session_id]
        participant = self.participants[session["participant_id"]]
        
        # Calculate session statistics
        assessments = session.get("assessments", {})
        total_assessments = len(assessments)
        valid_assessments = sum(1 for a in assessments.values() 
                              if a.get("validity_checks", {}).get("overall_valid", False))
        
        # Calculate scores
        session_scores = {}
        for assessment_name, assessment_data in assessments.items():
            if "scores" in assessment_data:
                session_scores[assessment_name] = assessment_data["scores"]
        
        # Generate report
        report = {
            "session_id": session_id,
            "participant_id": session["participant_id"],
            "session_date": session["session_date"],
            "condition": session["condition"],
            "status": session["status"],
            "completion_rate": valid_assessments / total_assessments if total_assessments > 0 else 0,
            "total_assessments": total_assessments,
            "valid_assessments": valid_assessments,
            "session_scores": session_scores,
            "participant_demographics": participant["demographics"],
            "generated_date": datetime.now().isoformat()
        }
        
        # Save report
        report_path = self.output_dir / f"session_report_{session_id}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Generated session report: {report_path}")
        return report
    
    def generate_participant_summary(self, participant_id: str) -> Dict[str, Any]:
        """Generate a participant summary"""
        if participant_id not in self.participants:
            raise ValueError(f"Participant {participant_id} not found")
        
        participant = self.participants[participant_id]
        
        # Get all sessions for this participant
        participant_sessions = [s for s in self.sessions.values() 
                              if s["participant_id"] == participant_id]
        
        # Calculate summary statistics
        total_sessions = len(participant_sessions)
        completed_sessions = sum(1 for s in participant_sessions 
                               if s["status"] == "completed")
        
        # Get session conditions
        conditions_completed = [s["condition"] for s in participant_sessions 
                              if s["status"] == "completed"]
        
        # Generate summary
        summary = {
            "participant_id": participant_id,
            "registration_date": participant["registration_date"],
            "eligibility_status": participant["eligibility_status"],
            "total_sessions": total_sessions,
            "completed_sessions": completed_sessions,
            "completion_rate": completed_sessions / total_sessions if total_sessions > 0 else 0,
            "conditions_completed": conditions_completed,
            "demographics": participant["demographics"],
            "status": participant["status"],
            "generated_date": datetime.now().isoformat()
        }
        
        # Save summary
        summary_path = self.output_dir / f"participant_summary_{participant_id}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Generated participant summary: {summary_path}")
        return summary
    
    def generate_study_progress_report(self) -> Dict[str, Any]:
        """Generate a study progress report"""
        # Calculate study statistics
        total_participants = len(self.participants)
        eligible_participants = sum(1 for p in self.participants.values() 
                                  if p["eligibility_status"] == "eligible")
        
        total_sessions = len(self.sessions)
        completed_sessions = sum(1 for s in self.sessions.values() 
                               if s["status"] == "completed")
        
        # Calculate completion by condition
        condition_completion = {}
        for condition in self.study_config["conditions"]:
            condition_name = condition["name"]
            condition_sessions = [s for s in self.sessions.values() 
                                if s["condition"] == condition_name]
            condition_completion[condition_name] = {
                "total": len(condition_sessions),
                "completed": sum(1 for s in condition_sessions 
                               if s["status"] == "completed")
            }
        
        # Generate progress report
        progress_report = {
            "study_id": self.study_config["study_id"],
            "study_title": self.study_config["study_title"],
            "target_participants": self.study_config["target_participants"],
            "total_participants": total_participants,
            "eligible_participants": eligible_participants,
            "enrollment_rate": total_participants / self.study_config["target_participants"],
            "total_sessions": total_sessions,
            "completed_sessions": completed_sessions,
            "session_completion_rate": completed_sessions / total_sessions if total_sessions > 0 else 0,
            "condition_completion": condition_completion,
            "generated_date": datetime.now().isoformat()
        }
        
        # Save progress report
        progress_path = self.output_dir / "study_progress_report.json"
        with open(progress_path, 'w') as f:
            json.dump(progress_report, f, indent=2, default=str)
        
        logger.info(f"Generated study progress report: {progress_path}")
        return progress_report
    
    def _save_participant_data(self, participant_id: str):
        """Save participant data to file"""
        participant_path = self.output_dir / f"participant_{participant_id}.json"
        with open(participant_path, 'w') as f:
            json.dump(self.participants[participant_id], f, indent=2, default=str)
    
    def _save_session_data(self, session_id: str):
        """Save session data to file"""
        session_path = self.output_dir / f"session_{session_id}.json"
        with open(session_path, 'w') as f:
            json.dump(self.sessions[session_id], f, indent=2, default=str)
    
    def _save_signature_data(self, signature_id: str, signature: HumanSignature):
        """Save signature data to file"""
        signature_path = self.output_dir / f"signature_{signature_id}.json"
        with open(signature_path, 'w') as f:
            json.dump(asdict(signature), f, indent=2, default=str)
    
    def export_all_data(self, filename: str = None) -> Path:
        """Export all workbook data"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"human_reference_workbook_{timestamp}.json"
        
        export_path = self.output_dir / filename
        
        # Compile all data
        export_data = {
            "study_config": self.study_config,
            "participants": self.participants,
            "sessions": self.sessions,
            "signatures": {k: asdict(v) for k, v in self.signatures.items()},
            "export_date": datetime.now().isoformat()
        }
        
        with open(export_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"Exported all workbook data to: {export_path}")
        return export_path


def main():
    """Main function to demonstrate the human reference workbook"""
    print("📚 Human Reference Data Collection Workbook Demo")
    print("=" * 60)
    
    # Create workbook
    workbook = HumanReferenceWorkbook()
    
    # Register sample participants
    print("👥 Registering sample participants...")
    sample_participants = [
        {
            "participant_id": "P001",
            "demographics": {
                "age": 25,
                "gender": "female",
                "education": "bachelor",
                "occupation": "student"
            }
        },
        {
            "participant_id": "P002", 
            "demographics": {
                "age": 30,
                "gender": "male",
                "education": "master",
                "occupation": "researcher"
            }
        }
    ]
    
    for participant_data in sample_participants:
        workbook.register_participant(participant_data["participant_id"], 
                                    participant_data["demographics"])
    
    # Conduct screening
    print("\n🔍 Conducting screening...")
    screening_data = {
        "eligibility_criteria": {
            "age_18_65": True,
            "no_psychiatric_history": True,
            "no_cardiovascular_conditions": True,
            "no_substance_dependence": True,
            "not_pregnant": True,
            "no_medication_interactions": True
        },
        "medical_history": {
            "psychiatric_history": [],
            "cardiovascular_conditions": [],
            "substance_dependence": []
        }
    }
    
    for participant_data in sample_participants:
        eligibility = workbook.conduct_screening(participant_data["participant_id"], 
                                               screening_data)
        print(f"   {participant_data['participant_id']}: {eligibility['status']}")
    
    # Schedule sessions
    print("\n📅 Scheduling sessions...")
    conditions = ["caffeine", "lsd", "placebo"]
    session_ids = []
    
    for participant_data in sample_participants:
        for condition in conditions:
            session_id = workbook.schedule_session(
                participant_data["participant_id"],
                condition,
                "2024-01-15",
                "09:00"
            )
            session_ids.append(session_id)
    
    print(f"   Scheduled {len(session_ids)} sessions")
    
    # Conduct sample session
    print("\n🧪 Conducting sample session...")
    sample_session_data = {
        "assessments": {
            "ascs": {
                "assessment_name": "ascs",
                "items": [
                    {"item_number": 1, "item_text": "I feel as if I am floating", "response": "3", "subscale": "Oceanic Boundlessness"},
                    {"item_number": 2, "item_text": "I feel as if I am dissolving", "response": "2", "subscale": "Dread of Ego Dissolution"}
                ],
                "completion_time": 300
            },
            "poms": {
                "assessment_name": "poms",
                "items": [
                    {"item_number": 1, "item_text": "Tense", "response": "1", "subscale": "Tension"},
                    {"item_number": 2, "item_text": "Energetic", "response": "4", "subscale": "Vigor"}
                ],
                "completion_time": 180
            }
        }
    }
    
    # Conduct first session
    if session_ids:
        completed_session = workbook.conduct_session(session_ids[0], sample_session_data)
        print(f"   Completed session: {session_ids[0]}")
    
    # Generate reports
    print("\n📊 Generating reports...")
    
    # Session report
    if session_ids:
        session_report = workbook.generate_session_report(session_ids[0])
        print(f"   Session report: {session_report['completion_rate']:.1%} completion rate")
    
    # Participant summary
    participant_summary = workbook.generate_participant_summary("P001")
    print(f"   Participant summary: {participant_summary['completion_rate']:.1%} completion rate")
    
    # Study progress report
    progress_report = workbook.generate_study_progress_report()
    print(f"   Study progress: {progress_report['enrollment_rate']:.1%} enrollment rate")
    
    # Export all data
    print("\n💾 Exporting all data...")
    export_path = workbook.export_all_data()
    print(f"   Exported to: {export_path}")
    
    print("\n🎉 Human reference workbook system ready!")
    print(f"📂 All files saved to: {workbook.output_dir}")
    print("\n💡 Next steps:")
    print("   1. Use the workbook to manage participant registration")
    print("   2. Conduct screening and eligibility assessment")
    print("   3. Schedule and conduct study sessions")
    print("   4. Generate reports and export data")
    print("   5. Perform signature matching with model data")


if __name__ == "__main__":
    main()
