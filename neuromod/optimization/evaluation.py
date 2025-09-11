"""
Evaluation Framework for Pack Optimization

This module provides tools to evaluate how well packs achieve behavioral targets
by measuring emotions, behaviors, and metrics from model outputs.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging
import re
from collections import Counter
import json

logger = logging.getLogger(__name__)

@dataclass
class BehavioralMetrics:
    """Container for behavioral metrics from model evaluation"""
    emotions: Dict[str, float] = None
    behaviors: Dict[str, float] = None
    metrics: Dict[str, float] = None
    
    def __post_init__(self):
        if self.emotions is None:
            self.emotions = {}
        if self.behaviors is None:
            self.behaviors = {}
        if self.metrics is None:
            self.metrics = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'emotions': self.emotions,
            'behaviors': self.behaviors,
            'metrics': self.metrics
        }
    
    def get_all_metrics(self) -> Dict[str, float]:
        """Get all metrics as a flat dictionary"""
        all_metrics = {}
        all_metrics.update({f"emotion_{k}": v for k, v in self.emotions.items()})
        all_metrics.update({f"behavior_{k}": v for k, v in self.behaviors.items()})
        all_metrics.update({f"metric_{k}": v for k, v in self.metrics.items()})
        return all_metrics

class EmotionAnalyzer:
    """Analyzes emotional content in text"""
    
    def __init__(self):
        # Simple emotion keyword dictionaries (could be replaced with more sophisticated models)
        self.emotion_keywords = {
            'joy': ['happy', 'joy', 'excited', 'cheerful', 'delighted', 'pleased', 'thrilled', 'elated'],
            'sadness': ['sad', 'depressed', 'melancholy', 'gloomy', 'down', 'blue', 'unhappy', 'miserable'],
            'anger': ['angry', 'mad', 'furious', 'irritated', 'annoyed', 'rage', 'frustrated', 'livid'],
            'fear': ['afraid', 'scared', 'terrified', 'anxious', 'worried', 'nervous', 'frightened', 'panic'],
            'surprise': ['surprised', 'amazed', 'shocked', 'astonished', 'startled', 'bewildered', 'stunned'],
            'disgust': ['disgusted', 'revolted', 'sickened', 'repulsed', 'appalled', 'nauseated'],
            'calm': ['calm', 'peaceful', 'serene', 'tranquil', 'relaxed', 'composed', 'collected'],
            'anxiety': ['anxious', 'worried', 'nervous', 'uneasy', 'restless', 'agitated', 'tense']
        }
        
        # Sentiment patterns
        self.positive_patterns = [
            r'\b(great|excellent|wonderful|amazing|fantastic|brilliant|outstanding)\b',
            r'\b(love|adore|enjoy|appreciate|cherish|treasure)\b',
            r'\b(success|achieve|accomplish|succeed|win|victory)\b'
        ]
        
        self.negative_patterns = [
            r'\b(terrible|awful|horrible|disgusting|hateful|dreadful)\b',
            r'\b(hate|despise|loathe|detest|abhor)\b',
            r'\b(fail|failure|lose|defeat|disaster|catastrophe)\b'
        ]
    
    def analyze_emotions(self, text: str) -> Dict[str, float]:
        """Analyze emotional content in text"""
        text_lower = text.lower()
        emotions = {}
        
        for emotion, keywords in self.emotion_keywords.items():
            count = sum(1 for keyword in keywords if keyword in text_lower)
            # Normalize by text length
            emotions[emotion] = count / max(len(text.split()), 1)
        
        return emotions
    
    def analyze_sentiment(self, text: str) -> float:
        """Analyze overall sentiment (-1 to 1)"""
        text_lower = text.lower()
        
        positive_count = sum(len(re.findall(pattern, text_lower)) for pattern in self.positive_patterns)
        negative_count = sum(len(re.findall(pattern, text_lower)) for pattern in self.negative_patterns)
        
        total = positive_count + negative_count
        if total == 0:
            return 0.0
        
        return (positive_count - negative_count) / total

class BehaviorAnalyzer:
    """Analyzes behavioral patterns in text"""
    
    def __init__(self):
        self.behavior_patterns = {
            'socialization': [
                r'\b(we|us|our|together|community|friends|family|people)\b',
                r'\b(share|connect|relate|communicate|interact|collaborate)\b',
                r'\b(help|support|care|understand|empathize)\b'
            ],
            'creativity': [
                r'\b(imagine|create|design|invent|innovate|artistic|creative)\b',
                r'\b(unique|original|novel|unconventional|unusual)\b',
                r'\b(art|music|poetry|story|painting|sculpture)\b'
            ],
            'focus': [
                r'\b(focus|concentrate|attention|mindful|deliberate)\b',
                r'\b(detail|specific|precise|thorough|careful)\b',
                r'\b(analyze|examine|study|investigate|explore)\b'
            ],
            'reflection': [
                r'\b(think|consider|reflect|contemplate|ponder|meditate)\b',
                r'\b(learn|understand|realize|recognize|appreciate)\b',
                r'\b(experience|memory|past|future|meaning|purpose)\b'
            ],
            'optimism': [
                r'\b(hopeful|positive|optimistic|confident|encouraging)\b',
                r'\b(believe|trust|faith|hope|possibility|potential)\b',
                r'\b(better|improve|progress|growth|advance)\b'
            ]
        }
    
    def analyze_behaviors(self, text: str) -> Dict[str, float]:
        """Analyze behavioral patterns in text"""
        text_lower = text.lower()
        behaviors = {}
        
        for behavior, patterns in self.behavior_patterns.items():
            count = sum(len(re.findall(pattern, text_lower)) for pattern in patterns)
            # Normalize by text length
            behaviors[behavior] = count / max(len(text.split()), 1)
        
        return behaviors

class MetricAnalyzer:
    """Analyzes various metrics in text"""
    
    def __init__(self):
        pass
    
    def analyze_metrics(self, text: str) -> Dict[str, float]:
        """Analyze various metrics in text"""
        metrics = {}
        
        # Coherence (simplified - could use more sophisticated measures)
        metrics['coherence'] = self._analyze_coherence(text)
        
        # Originality (simplified - could use more sophisticated measures)
        metrics['originality'] = self._analyze_originality(text)
        
        # Thoughtfulness (based on length and complexity)
        metrics['thoughtfulness'] = self._analyze_thoughtfulness(text)
        
        # Optimism (from sentiment analysis)
        metrics['optimism'] = self._analyze_optimism(text)
        
        return metrics
    
    def _analyze_coherence(self, text: str) -> float:
        """Analyze text coherence (0-1)"""
        sentences = text.split('.')
        if len(sentences) < 2:
            return 0.5
        
        # Simple coherence based on sentence length consistency
        lengths = [len(s.split()) for s in sentences if s.strip()]
        if not lengths:
            return 0.5
        
        # Lower variance in sentence lengths suggests better coherence
        variance = np.var(lengths)
        max_variance = 100  # Arbitrary threshold
        coherence = max(0, 1 - variance / max_variance)
        
        return min(1.0, coherence)
    
    def _analyze_originality(self, text: str) -> float:
        """Analyze text originality (0-1)"""
        words = text.lower().split()
        if not words:
            return 0.0
        
        # Count unique words vs total words
        unique_words = len(set(words))
        total_words = len(words)
        
        # Also consider rare words (simplified)
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        rare_words = [w for w in words if w not in common_words]
        
        originality = (unique_words / total_words) * 0.7 + (len(rare_words) / total_words) * 0.3
        return min(1.0, originality)
    
    def _analyze_thoughtfulness(self, text: str) -> float:
        """Analyze thoughtfulness (0-1)"""
        # Based on text length, sentence complexity, and reflective language
        words = text.split()
        sentences = [s for s in text.split('.') if s.strip()]
        
        if not words or not sentences:
            return 0.0
        
        # Length factor
        length_factor = min(1.0, len(words) / 100)  # Normalize to 100 words
        
        # Sentence complexity (average words per sentence)
        avg_sentence_length = len(words) / len(sentences)
        complexity_factor = min(1.0, avg_sentence_length / 15)  # Normalize to 15 words per sentence
        
        # Reflective language
        reflective_words = ['think', 'consider', 'believe', 'feel', 'understand', 'realize', 'recognize']
        reflective_count = sum(1 for word in words if word.lower() in reflective_words)
        reflective_factor = min(1.0, reflective_count / 5)  # Normalize to 5 reflective words
        
        thoughtfulness = (length_factor * 0.4 + complexity_factor * 0.3 + reflective_factor * 0.3)
        return min(1.0, thoughtfulness)
    
    def _analyze_optimism(self, text: str) -> float:
        """Analyze optimism (0-1)"""
        # Simple keyword-based optimism analysis
        optimistic_words = [
            'hope', 'hopeful', 'positive', 'optimistic', 'confident', 'encouraging',
            'better', 'improve', 'progress', 'growth', 'advance', 'success',
            'believe', 'trust', 'faith', 'possibility', 'potential', 'bright'
        ]
        
        pessimistic_words = [
            'hopeless', 'negative', 'pessimistic', 'doubt', 'worry', 'concern',
            'worse', 'decline', 'failure', 'problem', 'difficulty', 'challenge',
            'doubt', 'fear', 'anxiety', 'impossible', 'unlikely', 'dark'
        ]
        
        words = text.lower().split()
        if not words:
            return 0.5
        
        opt_count = sum(1 for word in words if word in optimistic_words)
        pess_count = sum(1 for word in words if word in pessimistic_words)
        
        total = opt_count + pess_count
        if total == 0:
            return 0.5
        
        optimism = opt_count / total
        return optimism

class EvaluationFramework:
    """Main evaluation framework for pack optimization"""
    
    def __init__(self):
        self.emotion_analyzer = EmotionAnalyzer()
        self.behavior_analyzer = BehaviorAnalyzer()
        self.metric_analyzer = MetricAnalyzer()
    
    def evaluate_text(self, text: str) -> BehavioralMetrics:
        """Evaluate a single text for behavioral metrics"""
        emotions = self.emotion_analyzer.analyze_emotions(text)
        behaviors = self.behavior_analyzer.analyze_behaviors(text)
        metrics = self.metric_analyzer.analyze_metrics(text)
        
        return BehavioralMetrics(
            emotions=emotions,
            behaviors=behaviors,
            metrics=metrics
        )
    
    def evaluate_texts(self, texts: List[str]) -> BehavioralMetrics:
        """Evaluate multiple texts and return averaged metrics"""
        if not texts:
            return BehavioralMetrics()
        
        all_emotions = {}
        all_behaviors = {}
        all_metrics = {}
        
        # Collect all unique keys
        emotion_keys = set()
        behavior_keys = set()
        metric_keys = set()
        
        for text in texts:
            metrics = self.evaluate_text(text)
            emotion_keys.update(metrics.emotions.keys())
            behavior_keys.update(metrics.behaviors.keys())
            metric_keys.update(metrics.metrics.keys())
        
        # Average across all texts
        for key in emotion_keys:
            values = []
            for text in texts:
                metrics = self.evaluate_text(text)
                values.append(metrics.emotions.get(key, 0.0))
            all_emotions[key] = np.mean(values)
        
        for key in behavior_keys:
            values = []
            for text in texts:
                metrics = self.evaluate_text(text)
                values.append(metrics.behaviors.get(key, 0.0))
            all_behaviors[key] = np.mean(values)
        
        for key in metric_keys:
            values = []
            for text in texts:
                metrics = self.evaluate_text(text)
                values.append(metrics.metrics.get(key, 0.0))
            all_metrics[key] = np.mean(values)
        
        return BehavioralMetrics(
            emotions=all_emotions,
            behaviors=all_behaviors,
            metrics=all_metrics
        )
    
    def evaluate_model_outputs(self, 
                             model_outputs: List[str], 
                             prompts: List[str] = None) -> BehavioralMetrics:
        """Evaluate model outputs for behavioral metrics"""
        # If prompts provided, we could analyze prompt-response pairs
        # For now, just analyze the outputs
        return self.evaluate_texts(model_outputs)
    
    def compute_target_loss(self, 
                           target_metrics: BehavioralMetrics, 
                           actual_metrics: BehavioralMetrics) -> float:
        """Compute loss between target and actual metrics"""
        target_dict = target_metrics.get_all_metrics()
        actual_dict = actual_metrics.get_all_metrics()
        
        # Simple MSE loss
        all_keys = set(target_dict.keys()) | set(actual_dict.keys())
        loss = 0.0
        
        for key in all_keys:
            target_val = target_dict.get(key, 0.0)
            actual_val = actual_dict.get(key, 0.0)
            loss += (target_val - actual_val) ** 2
        
        return loss / max(len(all_keys), 1)

# Convenience functions
def evaluate_single_text(text: str) -> BehavioralMetrics:
    """Evaluate a single text"""
    framework = EvaluationFramework()
    return framework.evaluate_text(text)

def evaluate_texts(texts: List[str]) -> BehavioralMetrics:
    """Evaluate multiple texts"""
    framework = EvaluationFramework()
    return framework.evaluate_texts(texts)
