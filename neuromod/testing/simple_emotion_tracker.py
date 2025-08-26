#!/usr/bin/env python3
"""
Simplified Emotion Tracker for Survey-Based Testing

Tracks basic emotional changes (up/down) rather than complex mathematical calculations.
Perfect for survey responses and simple emotional delta detection.
"""

import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class SimpleEmotionState:
    """Simplified emotional state tracking"""
    timestamp: str
    test_id: str
    
    # Basic emotion indicators (up/down/stable)
    joy: str  # "up", "down", "stable"
    sadness: str
    anger: str
    fear: str
    surprise: str
    disgust: str
    trust: str
    anticipation: str
    
    # Overall emotional valence
    valence: str  # "positive", "negative", "neutral"
    
    # Confidence in assessment
    confidence: float  # 0.0 to 1.0
    
    # Notes about what caused the change
    notes: str


class SimpleEmotionTracker:
    """
    Simplified emotion tracker that just tracks up/down changes.
    
    Perfect for survey-based testing where we want to know:
    - Did joy increase or decrease?
    - Did fear go up or down?
    - What's the overall emotional direction?
    """
    
    def __init__(self):
        """Initialize the simple emotion tracker"""
        self.emotion_history: List[SimpleEmotionState] = []
        self.current_state: Optional[SimpleEmotionState] = None
        
        # Define the 8 basic emotions (Plutchik's wheel)
        self.basic_emotions = [
            'joy', 'sadness', 'anger', 'fear', 
            'surprise', 'disgust', 'trust', 'anticipation'
        ]
    
    def assess_emotion_change(self, 
                            text: str, 
                            test_id: str,
                            previous_text: Optional[str] = None) -> SimpleEmotionState:
        """
        Assess emotional changes based on text content.
        
        Args:
            text: Current text to analyze
            test_id: Identifier for the test
            previous_text: Previous text for comparison (optional)
            
        Returns:
            SimpleEmotionState with up/down indicators
        """
        # Simple keyword-based emotion detection
        emotion_scores = self._analyze_text_emotions(text)
        
        # Compare with previous state if available
        if previous_text and self.current_state:
            emotion_changes = self._detect_emotion_changes(emotion_scores, previous_text)
        else:
            # First assessment - assume stable
            emotion_changes = {emotion: "stable" for emotion in self.basic_emotions}
        
        # Determine overall valence
        valence = self._determine_overall_valence(emotion_scores)
        
        # Calculate confidence based on emotion strength
        confidence = self._calculate_confidence(emotion_scores)
        
        # Create emotion state
        state = SimpleEmotionState(
            timestamp=datetime.now().isoformat(),
            test_id=test_id,
            joy=emotion_changes.get('joy', 'stable'),
            sadness=emotion_changes.get('sadness', 'stable'),
            anger=emotion_changes.get('anger', 'stable'),
            fear=emotion_changes.get('fear', 'stable'),
            surprise=emotion_changes.get('surprise', 'stable'),
            disgust=emotion_changes.get('disgust', 'stable'),
            trust=emotion_changes.get('trust', 'stable'),
            anticipation=emotion_changes.get('anticipation', 'stable'),
            valence=valence,
            confidence=confidence,
            notes=f"Analyzed text: {text[:100]}..."
        )
        
        # Update current state and history
        self.current_state = state
        self.emotion_history.append(state)
        
        return state
    
    def _analyze_text_emotions(self, text: str) -> Dict[str, float]:
        """Analyze text for basic emotions using keyword matching"""
        text_lower = text.lower()
        
        # Simple keyword-based emotion detection
        emotion_keywords = {
            'joy': ['happy', 'joy', 'excited', 'pleased', 'delighted', 'cheerful', 'glad', 'thrilled'],
            'sadness': ['sad', 'unhappy', 'depressed', 'melancholy', 'sorrow', 'grief', 'down', 'blue'],
            'anger': ['angry', 'mad', 'furious', 'irritated', 'annoyed', 'frustrated', 'rage', 'hostile'],
            'fear': ['afraid', 'scared', 'fearful', 'terrified', 'anxious', 'worried', 'nervous', 'panicked'],
            'surprise': ['surprised', 'amazed', 'astonished', 'shocked', 'startled', 'stunned', 'bewildered'],
            'disgust': ['disgusted', 'revolted', 'repulsed', 'sickened', 'appalled', 'horrified'],
            'trust': ['trust', 'confident', 'secure', 'assured', 'reliable', 'faithful', 'loyal'],
            'anticipation': ['excited', 'eager', 'enthusiastic', 'hopeful', 'optimistic', 'expectant']
        }
        
        emotion_scores = {}
        for emotion, keywords in emotion_keywords.items():
            score = 0.0
            for keyword in keywords:
                if keyword in text_lower:
                    score += 1.0
            # Normalize score
            emotion_scores[emotion] = min(score / len(keywords), 1.0)
        
        return emotion_scores
    
    def _detect_emotion_changes(self, 
                               current_scores: Dict[str, float], 
                               previous_text: str) -> Dict[str, str]:
        """Detect whether emotions went up, down, or stayed stable"""
        previous_scores = self._analyze_text_emotions(previous_text)
        
        changes = {}
        for emotion in self.basic_emotions:
            current = current_scores.get(emotion, 0.0)
            previous = previous_scores.get(emotion, 0.0)
            
            # Simple threshold-based change detection
            if abs(current - previous) < 0.1:
                changes[emotion] = "stable"
            elif current > previous:
                changes[emotion] = "up"
            else:
                changes[emotion] = "down"
        
        return changes
    
    def _determine_overall_valence(self, emotion_scores: Dict[str, float]) -> str:
        """Determine overall emotional valence"""
        positive_emotions = ['joy', 'trust', 'anticipation']
        negative_emotions = ['sadness', 'anger', 'fear', 'disgust']
        
        positive_score = sum(emotion_scores.get(emotion, 0.0) for emotion in positive_emotions)
        negative_score = sum(emotion_scores.get(emotion, 0.0) for emotion in negative_emotions)
        
        if positive_score > negative_score + 0.5:
            return "positive"
        elif negative_score > positive_score + 0.5:
            return "negative"
        else:
            return "neutral"
    
    def _calculate_confidence(self, emotion_scores: Dict[str, float]) -> float:
        """Calculate confidence in the emotional assessment"""
        # Higher confidence when emotions are more clearly defined
        max_score = max(emotion_scores.values())
        avg_score = sum(emotion_scores.values()) / len(emotion_scores)
        
        # Confidence increases with stronger emotions and clearer patterns
        confidence = (max_score * 0.6) + (avg_score * 0.4)
        return min(confidence, 1.0)
    
    def get_emotion_summary(self, test_id: str) -> Dict[str, any]:
        """Get a summary of emotional changes for a specific test"""
        test_states = [state for state in self.emotion_history if state.test_id == test_id]
        
        if not test_states:
            return {"error": "No emotion data found for test"}
        
        # Count emotion changes
        emotion_counts = {emotion: {"up": 0, "down": 0, "stable": 0} 
                         for emotion in self.basic_emotions}
        
        for state in test_states:
            for emotion in self.basic_emotions:
                change = getattr(state, emotion)
                emotion_counts[emotion][change] += 1
        
        # Overall valence trend
        valence_trend = "neutral"
        positive_count = sum(1 for state in test_states if state.valence == "positive")
        negative_count = sum(1 for state in test_states if state.valence == "negative")
        
        if positive_count > negative_count:
            valence_trend = "positive"
        elif negative_count > positive_count:
            valence_trend = "negative"
        
        return {
            "test_id": test_id,
            "total_assessments": len(test_states),
            "emotion_changes": emotion_counts,
            "valence_trend": valence_trend,
            "average_confidence": sum(state.confidence for state in test_states) / len(test_states),
            "first_assessment": test_states[0].timestamp,
            "last_assessment": test_states[-1].timestamp
        }
    
    def export_results(self, filename: str):
        """Export emotion tracking results to JSON"""
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "total_states": len(self.emotion_history),
            "emotion_history": [asdict(state) for state in self.emotion_history],
            "summary": {
                "tests_analyzed": list(set(state.test_id for state in self.emotion_history)),
                "emotion_totals": {
                    emotion: {
                        "up": sum(1 for state in self.emotion_history if getattr(state, emotion) == "up"),
                        "down": sum(1 for state in self.emotion_history if getattr(state, emotion) == "down"),
                        "stable": sum(1 for state in self.emotion_history if getattr(state, emotion) == "stable")
                    }
                    for emotion in self.basic_emotions
                }
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported emotion tracking results to {filename}")
    
    def reset(self):
        """Reset the emotion tracker"""
        self.emotion_history.clear()
        self.current_state = None
        logger.info("Emotion tracker reset")


# Example usage
if __name__ == "__main__":
    # Initialize tracker
    tracker = SimpleEmotionTracker()
    
    # Example test
    test_id = "survey_test_001"
    
    # First assessment
    state1 = tracker.assess_emotion_change(
        "I am feeling very happy and excited about the results!",
        test_id
    )
    print(f"First assessment: {state1.valence} valence, joy: {state1.joy}")
    
    # Second assessment
    state2 = tracker.assess_emotion_change(
        "I'm worried about the outcome and feeling anxious.",
        test_id,
        "I am feeling very happy and excited about the results!"
    )
    print(f"Second assessment: {state2.valence} valence, joy: {state2.joy}, fear: {state2.fear}")
    
    # Get summary
    summary = tracker.get_emotion_summary(test_id)
    print(f"Test summary: {summary}")
    
    # Export results
    tracker.export_results("simple_emotion_results.json")
