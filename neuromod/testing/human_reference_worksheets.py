#!/usr/bin/env python3
"""
Human Reference Data Collection Worksheets

This module provides standardized worksheets and data collection forms
for human psychopharmacological studies. It includes assessment forms,
data entry templates, and validation procedures.

Key Features:
- Standardized assessment forms
- Real-time data validation
- Automated scoring algorithms
- Data export and formatting
- Quality control procedures
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import json
from datetime import datetime
import logging

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class ParticipantInfo:
    """Participant information and demographics"""
    participant_id: str
    age: int
    gender: str
    education: str
    occupation: str
    medical_history: List[str]
    medication_history: List[str]
    substance_use_history: Dict[str, str]
    screening_date: str
    eligibility_status: str

@dataclass
class SessionInfo:
    """Session information and substance administration"""
    session_id: str
    participant_id: str
    session_date: str
    session_time: str
    condition: str  # substance or placebo
    dosage: float
    route: str
    administration_time: str
    assessor_id: str
    room_temperature: float
    room_humidity: float
    notes: str

@dataclass
class AssessmentResponse:
    """Individual assessment response"""
    assessment_id: str
    session_id: str
    assessment_name: str
    assessment_time: str  # T+30, T+60, etc.
    responses: Dict[str, Any]
    completion_time: int  # seconds
    validity_checks: Dict[str, bool]
    raw_score: float
    standardized_score: float
    subscale_scores: Dict[str, float]

class HumanReferenceDataCollector:
    """Main class for human reference data collection"""
    
    def __init__(self, output_dir: str = "outputs/analysis/human_reference"):
        """
        Initialize the human reference data collector
        
        Args:
            output_dir: Directory to save collected data
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Assessment definitions
        self.assessments = self._define_assessments()
        
        # Scoring algorithms
        self.scoring_algorithms = self._define_scoring_algorithms()
        
        # Validation rules
        self.validation_rules = self._define_validation_rules()
    
    def _define_assessments(self) -> Dict[str, Dict[str, Any]]:
        """Define all assessment instruments"""
        return {
            "ascs": {
                "name": "Altered States of Consciousness Scale",
                "items": 94,
                "subscales": [
                    "Oceanic Boundlessness", "Dread of Ego Dissolution", "Visionary Restructuralization",
                    "Auditory Alterations", "Reduction of Vigilance", "Vigilance Reduction",
                    "Complex Imagery", "Elementary Imagery", "Synesthesia", "Changed Meaning",
                    "Time Experience"
                ],
                "scale_range": (0, 5),
                "timing": ["T+60", "T+120", "T+240"]
            },
            "sdq": {
                "name": "State Dissociation Questionnaire",
                "items": 21,
                "subscales": ["Depersonalization", "Derealization", "Amnesia"],
                "scale_range": (0, 4),
                "timing": ["T+60", "T+120", "T+240"]
            },
            "poms": {
                "name": "Profile of Mood States",
                "items": 65,
                "subscales": ["Tension", "Depression", "Anger", "Vigor", "Fatigue", "Confusion"],
                "scale_range": (0, 4),
                "timing": ["T+30", "T+60", "T+120", "T+240"]
            },
            "caq": {
                "name": "Creative Achievement Questionnaire",
                "items": 96,
                "subscales": [
                    "Visual Arts", "Music", "Dance", "Architectural Design", "Creative Writing",
                    "Humor", "Inventions", "Scientific Discovery", "Theater and Film", "Culinary Arts"
                ],
                "scale_range": (0, 7),
                "timing": ["T+60", "T+120"]
            },
            "digit_span": {
                "name": "Digit Span Task",
                "items": "variable",
                "subscales": ["Forward Span", "Backward Span"],
                "scale_range": (0, 12),
                "timing": ["T+60", "T+120"]
            },
            "stroop": {
                "name": "Stroop Color-Word Test",
                "items": "variable",
                "subscales": ["Congruent RT", "Incongruent RT", "Interference"],
                "scale_range": (0, 10000),  # milliseconds
                "timing": ["T+60", "T+120"]
            },
            "rat": {
                "name": "Remote Associates Test",
                "items": 30,
                "subscales": ["Correct Associations"],
                "scale_range": (0, 30),
                "timing": ["T+60", "T+120"]
            },
            "trail_making": {
                "name": "Trail Making Test",
                "items": 2,
                "subscales": ["Part A Time", "Part B Time"],
                "scale_range": (0, 300),  # seconds
                "timing": ["T+60", "T+120"]
            }
        }
    
    def _define_scoring_algorithms(self) -> Dict[str, Dict[str, Any]]:
        """Define scoring algorithms for each assessment"""
        return {
            "ascs": {
                "subscale_items": {
                    "Oceanic Boundlessness": list(range(1, 12)),
                    "Dread of Ego Dissolution": list(range(12, 23)),
                    "Visionary Restructuralization": list(range(23, 34)),
                    "Auditory Alterations": list(range(34, 45)),
                    "Reduction of Vigilance": list(range(45, 56)),
                    "Vigilance Reduction": list(range(56, 67)),
                    "Complex Imagery": list(range(67, 78)),
                    "Elementary Imagery": list(range(78, 89)),
                    "Synesthesia": list(range(89, 94))
                },
                "scoring_method": "sum"
            },
            "sdq": {
                "subscale_items": {
                    "Depersonalization": list(range(1, 8)),
                    "Derealization": list(range(8, 15)),
                    "Amnesia": list(range(15, 22))
                },
                "scoring_method": "sum"
            },
            "poms": {
                "subscale_items": {
                    "Tension": list(range(1, 11)),
                    "Depression": list(range(11, 21)),
                    "Anger": list(range(21, 31)),
                    "Vigor": list(range(31, 41)),
                    "Fatigue": list(range(41, 51)),
                    "Confusion": list(range(51, 61))
                },
                "scoring_method": "sum"
            },
            "caq": {
                "subscale_items": {
                    "Visual Arts": list(range(1, 10)),
                    "Music": list(range(10, 19)),
                    "Dance": list(range(19, 28)),
                    "Architectural Design": list(range(28, 37)),
                    "Creative Writing": list(range(37, 46)),
                    "Humor": list(range(46, 55)),
                    "Inventions": list(range(55, 64)),
                    "Scientific Discovery": list(range(64, 73)),
                    "Theater and Film": list(range(73, 82)),
                    "Culinary Arts": list(range(82, 91))
                },
                "scoring_method": "sum"
            }
        }
    
    def _define_validation_rules(self) -> Dict[str, Dict[str, Any]]:
        """Define validation rules for data quality"""
        return {
            "completion_rate": {
                "minimum": 0.80,  # 80% of items must be completed
                "action": "flag_for_review"
            },
            "response_range": {
                "action": "validate_scale_range"
            },
            "response_time": {
                "minimum": 30,  # seconds
                "maximum": 1800,  # 30 minutes
                "action": "flag_for_review"
            },
            "consistency_checks": {
                "action": "check_response_consistency"
            }
        }
    
    def create_participant_worksheet(self, participant_id: str) -> Dict[str, Any]:
        """Create a participant information worksheet"""
        worksheet = {
            "participant_id": participant_id,
            "demographics": {
                "age": "",
                "gender": "",
                "education": "",
                "occupation": "",
                "handedness": "",
                "native_language": ""
            },
            "medical_history": {
                "current_medications": [],
                "past_medications": [],
                "medical_conditions": [],
                "psychiatric_history": [],
                "substance_use_history": {
                    "alcohol": "",
                    "tobacco": "",
                    "cannabis": "",
                    "other_substances": ""
                },
                "allergies": [],
                "contraindications": []
            },
            "screening": {
                "date": "",
                "assessor": "",
                "eligibility_criteria": {
                    "age_18_65": False,
                    "no_psychiatric_history": False,
                    "no_cardiovascular_conditions": False,
                    "no_substance_dependence": False,
                    "not_pregnant": False,
                    "no_medication_interactions": False
                },
                "eligibility_status": "",
                "notes": ""
            },
            "consent": {
                "informed_consent_date": "",
                "consent_version": "",
                "participant_signature": "",
                "witness_signature": "",
                "assessor_signature": ""
            }
        }
        
        return worksheet
    
    def create_session_worksheet(self, session_id: str, participant_id: str) -> Dict[str, Any]:
        """Create a session worksheet"""
        worksheet = {
            "session_id": session_id,
            "participant_id": participant_id,
            "session_info": {
                "date": "",
                "start_time": "",
                "end_time": "",
                "assessor_id": "",
                "room_number": "",
                "room_temperature": "",
                "room_humidity": "",
                "lighting_conditions": "",
                "noise_level": ""
            },
            "substance_administration": {
                "condition": "",  # substance name or "placebo"
                "dosage": "",
                "route": "",
                "administration_time": "",
                "batch_number": "",
                "expiration_date": "",
                "preparation_time": "",
                "administration_notes": ""
            },
            "vital_signs": {
                "pre_dose": {
                    "heart_rate": "",
                    "blood_pressure_systolic": "",
                    "blood_pressure_diastolic": "",
                    "body_temperature": "",
                    "weight": ""
                },
                "post_dose": {
                    "T+30": {"heart_rate": "", "blood_pressure": "", "temperature": ""},
                    "T+60": {"heart_rate": "", "blood_pressure": "", "temperature": ""},
                    "T+120": {"heart_rate": "", "blood_pressure": "", "temperature": ""},
                    "T+240": {"heart_rate": "", "blood_pressure": "", "temperature": ""}
                }
            },
            "adverse_events": {
                "events": [],
                "severity": [],
                "action_taken": [],
                "resolution": []
            },
            "session_notes": ""
        }
        
        return worksheet
    
    def create_assessment_worksheet(self, assessment_name: str, session_id: str, 
                                  assessment_time: str) -> Dict[str, Any]:
        """Create an assessment worksheet"""
        if assessment_name not in self.assessments:
            raise ValueError(f"Unknown assessment: {assessment_name}")
        
        assessment_info = self.assessments[assessment_name]
        
        worksheet = {
            "assessment_id": f"{session_id}_{assessment_name}_{assessment_time}",
            "session_id": session_id,
            "assessment_name": assessment_name,
            "assessment_time": assessment_time,
            "start_time": "",
            "end_time": "",
            "assessor_id": "",
            "instructions": self._get_assessment_instructions(assessment_name),
            "items": self._create_assessment_items(assessment_name),
            "completion_status": "not_started",
            "validity_checks": {
                "completion_rate": 0.0,
                "response_range_valid": True,
                "response_time_appropriate": True,
                "consistency_checks_passed": True
            },
            "scores": {
                "raw_total": 0.0,
                "standardized_total": 0.0,
                "subscale_scores": {}
            },
            "notes": ""
        }
        
        return worksheet
    
    def _get_assessment_instructions(self, assessment_name: str) -> str:
        """Get standardized instructions for each assessment"""
        instructions = {
            "ascs": """
            Please rate each statement based on your current experience.
            Use the following scale:
            0 = Not at all
            1 = Slightly
            2 = Moderately
            3 = Considerably
            4 = Very much
            5 = Extremely
            
            Rate each item based on how you feel RIGHT NOW.
            """,
            "sdq": """
            Please indicate how much each statement describes your current experience.
            Use the following scale:
            0 = Not at all
            1 = Slightly
            2 = Moderately
            3 = Very much
            4 = Extremely
            
            Rate each item based on how you feel RIGHT NOW.
            """,
            "poms": """
            Please rate how you feel RIGHT NOW for each mood descriptor.
            Use the following scale:
            0 = Not at all
            1 = A little
            2 = Moderately
            3 = Quite a bit
            4 = Extremely
            """,
            "caq": """
            Please indicate your level of creative achievement in each domain.
            Use the following scale:
            0 = No training or recognition
            1 = Serious amateur
            2 = Professional training
            3 = Professional recognition
            4 = Regional recognition
            5 = National recognition
            6 = International recognition
            7 = Major international recognition
            """
        }
        
        return instructions.get(assessment_name, "Please complete all items as instructed.")
    
    def _create_assessment_items(self, assessment_name: str) -> List[Dict[str, Any]]:
        """Create assessment items based on the assessment definition"""
        assessment_info = self.assessments[assessment_name]
        items = []
        
        if assessment_name == "ascs":
            # Sample ASCS items (abbreviated for demonstration)
            sample_items = [
                "I feel as if I am floating or flying",
                "I feel as if I am dissolving",
                "I see geometric patterns",
                "I hear sounds that aren't there",
                "I feel very relaxed",
                "I feel very alert",
                "I see complex images",
                "I see simple shapes",
                "I experience mixing of senses",
                "Things seem to have special meaning"
            ]
            
            for i, item_text in enumerate(sample_items, 1):
                items.append({
                    "item_number": i,
                    "item_text": item_text,
                    "response": "",
                    "subscale": self._get_item_subscale(assessment_name, i)
                })
        
        elif assessment_name == "sdq":
            # Sample SDQ items
            sample_items = [
                "I feel detached from my body",
                "I feel like I'm watching myself from outside",
                "My body feels strange or unfamiliar",
                "I feel disconnected from my surroundings",
                "The world seems unreal or dreamlike",
                "I feel like I'm in a movie",
                "I have gaps in my memory",
                "I can't remember what just happened"
            ]
            
            for i, item_text in enumerate(sample_items, 1):
                items.append({
                    "item_number": i,
                    "item_text": item_text,
                    "response": "",
                    "subscale": self._get_item_subscale(assessment_name, i)
                })
        
        # Add more assessments as needed...
        
        return items
    
    def _get_item_subscale(self, assessment_name: str, item_number: int) -> str:
        """Get the subscale for a specific item"""
        if assessment_name in self.scoring_algorithms:
            subscale_items = self.scoring_algorithms[assessment_name]["subscale_items"]
            for subscale, items in subscale_items.items():
                if item_number in items:
                    return subscale
        return "unknown"
    
    def score_assessment(self, assessment_data: Dict[str, Any]) -> Dict[str, Any]:
        """Score an assessment based on responses"""
        assessment_name = assessment_data["assessment_name"]
        
        # Initialize scores if not present
        if "scores" not in assessment_data:
            assessment_data["scores"] = {
                "raw_total": 0.0,
                "standardized_total": 0.0,
                "subscale_scores": {}
            }
        
        if assessment_name not in self.scoring_algorithms:
            logger.warning(f"No scoring algorithm for {assessment_name}")
            return assessment_data
        
        # Calculate subscale scores
        subscale_scores = {}
        scoring_info = self.scoring_algorithms[assessment_name]
        
        for subscale, item_numbers in scoring_info["subscale_items"].items():
            subscale_responses = []
            for item in assessment_data["items"]:
                if item["item_number"] in item_numbers and item["response"] != "":
                    try:
                        subscale_responses.append(float(item["response"]))
                    except (ValueError, TypeError):
                        continue
            
            if subscale_responses:
                if scoring_info["scoring_method"] == "sum":
                    subscale_scores[subscale] = sum(subscale_responses)
                elif scoring_info["scoring_method"] == "mean":
                    subscale_scores[subscale] = np.mean(subscale_responses)
                else:
                    subscale_scores[subscale] = sum(subscale_responses)
            else:
                subscale_scores[subscale] = 0.0
        
        # Calculate total score
        total_score = sum(subscale_scores.values())
        
        # Standardize scores (z-score transformation)
        standardized_score = self._standardize_score(total_score, assessment_name)
        
        # Update assessment data
        assessment_data["scores"]["raw_total"] = total_score
        assessment_data["scores"]["standardized_total"] = standardized_score
        assessment_data["scores"]["subscale_scores"] = subscale_scores
        
        return assessment_data
    
    def _standardize_score(self, raw_score: float, assessment_name: str) -> float:
        """Standardize scores using population norms (placeholder implementation)"""
        # In practice, this would use actual population norms
        # For now, return a simple z-score transformation
        return (raw_score - 50) / 10  # Placeholder standardization
    
    def validate_assessment(self, assessment_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate assessment data quality"""
        validation_results = {
            "completion_rate": 0.0,
            "response_range_valid": True,
            "response_time_appropriate": True,
            "consistency_checks_passed": True,
            "overall_valid": True,
            "flags": []
        }
        
        # Check completion rate
        total_items = len(assessment_data["items"])
        completed_items = sum(1 for item in assessment_data["items"] if item["response"] != "")
        completion_rate = completed_items / total_items if total_items > 0 else 0
        
        validation_results["completion_rate"] = completion_rate
        
        if completion_rate < self.validation_rules["completion_rate"]["minimum"]:
            validation_results["flags"].append("Low completion rate")
            validation_results["overall_valid"] = False
        
        # Check response range
        assessment_name = assessment_data["assessment_name"]
        if assessment_name in self.assessments:
            min_val, max_val = self.assessments[assessment_name]["scale_range"]
            for item in assessment_data["items"]:
                if item["response"] != "":
                    try:
                        response_val = float(item["response"])
                        if response_val < min_val or response_val > max_val:
                            validation_results["response_range_valid"] = False
                            validation_results["flags"].append(f"Response out of range: {item['item_number']}")
                            validation_results["overall_valid"] = False
                    except (ValueError, TypeError):
                        validation_results["flags"].append(f"Invalid response format: {item['item_number']}")
                        validation_results["overall_valid"] = False
        
        # Update assessment data
        assessment_data["validity_checks"] = validation_results
        
        return assessment_data
    
    def export_session_data(self, session_data: Dict[str, Any], 
                          filename: str = None) -> Path:
        """Export session data to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"session_{session_data['session_id']}_{timestamp}.json"
        
        output_path = self.output_dir / filename
        
        with open(output_path, 'w') as f:
            json.dump(session_data, f, indent=2, default=str)
        
        logger.info(f"Session data exported to: {output_path}")
        return output_path
    
    def create_data_entry_template(self, assessment_name: str) -> pd.DataFrame:
        """Create a data entry template for an assessment"""
        if assessment_name not in self.assessments:
            raise ValueError(f"Unknown assessment: {assessment_name}")
        
        assessment_info = self.assessments[assessment_name]
        
        # Create template structure
        template_data = {
            "participant_id": [],
            "session_id": [],
            "assessment_time": [],
            "item_number": [],
            "item_text": [],
            "response": [],
            "subscale": []
        }
        
        # Add items
        items = self._create_assessment_items(assessment_name)
        for item in items:
            template_data["participant_id"].append("")
            template_data["session_id"].append("")
            template_data["assessment_time"].append("")
            template_data["item_number"].append(item["item_number"])
            template_data["item_text"].append(item["item_text"])
            template_data["response"].append("")
            template_data["subscale"].append(item["subscale"])
        
        df = pd.DataFrame(template_data)
        return df
    
    def generate_study_protocol(self) -> Dict[str, Any]:
        """Generate a complete study protocol"""
        protocol = {
            "study_title": "Human Reference Data Collection for Neuromodulation Research",
            "protocol_version": "1.0",
            "date_created": datetime.now().isoformat(),
            "participants": {
                "target_n": 120,
                "inclusion_criteria": [
                    "Age 18-65 years",
                    "Healthy adults",
                    "No psychiatric history",
                    "No cardiovascular conditions",
                    "No substance dependence",
                    "Not pregnant or nursing",
                    "No medication interactions"
                ],
                "exclusion_criteria": [
                    "Psychiatric history",
                    "Cardiovascular conditions",
                    "Substance dependence",
                    "Pregnancy or nursing",
                    "Medication interactions",
                    "Allergies to study substances"
                ]
            },
            "study_design": {
                "type": "Double-blind, placebo-controlled, randomized crossover",
                "duration": "8 weeks",
                "conditions": [
                    "Caffeine 200mg",
                    "L-Theanine 200mg", 
                    "Modafinil 100mg",
                    "LSD 100μg",
                    "MDMA 75mg",
                    "THC 10mg",
                    "Placebo"
                ],
                "randomization": "Latin square design",
                "washout_period": "1 week between conditions"
            },
            "assessments": list(self.assessments.keys()),
            "data_collection": {
                "timing": ["T-60", "T+0", "T+30", "T+60", "T+120", "T+240", "T+480"],
                "assessments_per_timing": {
                    "T-60": ["poms"],
                    "T+0": [],
                    "T+30": ["poms"],
                    "T+60": ["ascs", "sdq", "poms", "caq", "digit_span", "stroop", "rat", "trail_making"],
                    "T+120": ["ascs", "sdq", "poms", "caq", "digit_span", "stroop", "rat", "trail_making"],
                    "T+240": ["ascs", "sdq", "poms"],
                    "T+480": ["poms"]
                }
            },
            "safety_monitoring": {
                "vital_signs": ["heart_rate", "blood_pressure", "body_temperature"],
                "monitoring_frequency": "Every 30 minutes",
                "emergency_protocols": "24/7 medical supervision",
                "adverse_event_reporting": "Immediate reporting required"
            }
        }
        
        return protocol


def main():
    """Main function to demonstrate the human reference data collection system"""
    print("👥 Human Reference Data Collection System Demo")
    print("=" * 60)
    
    # Create collector
    collector = HumanReferenceDataCollector()
    
    # Create sample worksheets
    print("📋 Creating sample worksheets...")
    
    # Participant worksheet
    participant_worksheet = collector.create_participant_worksheet("P001")
    print("✅ Participant worksheet created")
    
    # Session worksheet
    session_worksheet = collector.create_session_worksheet("S001", "P001")
    print("✅ Session worksheet created")
    
    # Assessment worksheets
    assessment_worksheets = {}
    for assessment_name in ["ascs", "sdq", "poms", "caq"]:
        worksheet = collector.create_assessment_worksheet(assessment_name, "S001", "T+60")
        assessment_worksheets[assessment_name] = worksheet
        print(f"✅ {assessment_name.upper()} worksheet created")
    
    # Create data entry templates
    print("\n📊 Creating data entry templates...")
    for assessment_name in ["ascs", "sdq", "poms", "caq"]:
        template = collector.create_data_entry_template(assessment_name)
        template_path = collector.output_dir / f"{assessment_name}_template.csv"
        template.to_csv(template_path, index=False)
        print(f"✅ {assessment_name.upper()} template saved to {template_path}")
    
    # Generate study protocol
    print("\n📋 Generating study protocol...")
    protocol = collector.generate_study_protocol()
    protocol_path = collector.output_dir / "study_protocol.json"
    with open(protocol_path, 'w') as f:
        json.dump(protocol, f, indent=2)
    print(f"✅ Study protocol saved to {protocol_path}")
    
    # Export sample data
    print("\n💾 Exporting sample data...")
    sample_session = {
        "session_id": "S001",
        "participant_id": "P001",
        "assessments": assessment_worksheets
    }
    export_path = collector.export_session_data(sample_session)
    print(f"✅ Sample session data exported to {export_path}")
    
    print("\n🎉 Human reference data collection system ready!")
    print(f"📂 All files saved to: {collector.output_dir}")
    print("\n💡 Next steps:")
    print("   1. Review and customize worksheets for your study")
    print("   2. Train assessors on data collection procedures")
    print("   3. Begin participant recruitment and screening")
    print("   4. Start data collection sessions")


if __name__ == "__main__":
    main()
