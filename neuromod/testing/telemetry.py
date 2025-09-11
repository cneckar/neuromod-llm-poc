#!/usr/bin/env python3
"""
Telemetry System for Neuromodulation Testing

Implements behavioral telemetry from the paper outline:
- Repetition rate calculation
- Perplexity slope analysis
- Length/entropy metrics
- Attention entropy (if available)
- KV occupancy tracking
"""

import json
import re
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Deque
from dataclasses import dataclass, asdict
from datetime import datetime
from collections import deque
import logging

logger = logging.getLogger(__name__)

@dataclass
class TelemetryMetrics:
    """Telemetry metrics for a single generation"""
    generation_id: str
    timestamp: str
    input_length: int
    output_length: int
    total_tokens: int
    repetition_rate: float
    perplexity_slope: float
    entropy_metrics: Dict[str, float]
    attention_entropy: Optional[float]
    kv_occupancy: Optional[float]
    generation_time: float

@dataclass
class TelemetrySummary:
    """Summary of telemetry across multiple generations"""
    session_id: str
    total_generations: int
    avg_repetition_rate: float
    avg_perplexity_slope: float
    avg_entropy: float
    avg_attention_entropy: Optional[float]
    avg_kv_occupancy: Optional[float]
    total_generation_time: float
    metrics_history: List[TelemetryMetrics]

class TelemetryCollector:
    """
    Collects behavioral telemetry during model generation
    
    Tracks various behavioral metrics to assess neuromodulation effects
    on model behavior patterns.
    """
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics_history: Deque[TelemetryMetrics] = deque(maxlen=window_size)
        self.perplexity_history: Deque[float] = deque(maxlen=window_size)
        self.attention_history: Deque[float] = deque(maxlen=window_size)
        
    def calculate_repetition_rate(self, text: str) -> float:
        """Calculate repetition rate in text"""
        words = text.lower().split()
        if len(words) < 2:
            return 0.0
        
        # Count repeated n-grams
        bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
        trigrams = [f"{words[i]} {words[i+1]} {words[i+2]}" for i in range(len(words)-2)]
        
        # Calculate repetition rates
        bigram_repetition = len(set(bigrams)) / len(bigrams) if bigrams else 1.0
        trigram_repetition = len(set(trigrams)) / len(trigrams) if trigrams else 1.0
        
        # Combined repetition rate (lower = more repetitive)
        repetition_rate = (bigram_repetition + trigram_repetition) / 2.0
        return 1.0 - repetition_rate  # Convert to repetition rate (higher = more repetitive)
    
    def calculate_perplexity_slope(self, logits: Optional[List[float]] = None) -> float:
        """Calculate perplexity slope over generation"""
        if logits is None or len(logits) < 2:
            return 0.0
        
        # Convert logits to perplexity
        perplexities = [np.exp(-logit) for logit in logits]
        
        # Calculate slope using linear regression
        x = np.arange(len(perplexities))
        y = np.array(perplexities)
        
        if len(x) < 2:
            return 0.0
        
        # Simple linear regression
        n = len(x)
        slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x**2) - np.sum(x)**2)
        
        return slope
    
    def calculate_entropy_metrics(self, text: str) -> Dict[str, float]:
        """Calculate various entropy metrics"""
        words = text.lower().split()
        chars = list(text.lower())
        
        if not words or not chars:
            return {"word_entropy": 0.0, "char_entropy": 0.0, "lexical_diversity": 0.0}
        
        # Word entropy
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        word_probs = [count / len(words) for count in word_counts.values()]
        word_entropy = -sum(p * np.log2(p) for p in word_probs if p > 0)
        
        # Character entropy
        char_counts = {}
        for char in chars:
            char_counts[char] = char_counts.get(char, 0) + 1
        
        char_probs = [count / len(chars) for count in char_counts.values()]
        char_entropy = -sum(p * np.log2(p) for p in char_probs if p > 0)
        
        # Lexical diversity (type-token ratio)
        lexical_diversity = len(set(words)) / len(words)
        
        return {
            "word_entropy": word_entropy,
            "char_entropy": char_entropy,
            "lexical_diversity": lexical_diversity
        }
    
    def calculate_attention_entropy(self, attention_weights: Optional[List[List[float]]] = None) -> Optional[float]:
        """Calculate attention entropy if attention weights are available"""
        if attention_weights is None:
            return None
        
        try:
            # Flatten attention weights
            flat_weights = [weight for layer in attention_weights for weight in layer]
            
            if not flat_weights:
                return None
            
            # Normalize weights
            total_weight = sum(flat_weights)
            if total_weight == 0:
                return None
            
            normalized_weights = [w / total_weight for w in flat_weights]
            
            # Calculate entropy
            entropy = -sum(w * np.log2(w) for w in normalized_weights if w > 0)
            return entropy
            
        except Exception as e:
            logger.warning(f"Failed to calculate attention entropy: {e}")
            return None
    
    def estimate_kv_occupancy(self, input_length: int, output_length: int, 
                            model_size: str = "medium") -> Optional[float]:
        """Estimate KV cache occupancy based on sequence length"""
        # Rough estimates based on model size
        kv_ratios = {
            "small": 0.1,    # 7B models
            "medium": 0.15,  # 13B models  
            "large": 0.2,    # 70B models
            "xlarge": 0.25   # 70B+ models
        }
        
        base_ratio = kv_ratios.get(model_size, 0.15)
        total_length = input_length + output_length
        
        # Estimate occupancy (simplified model)
        occupancy = min(1.0, (total_length * base_ratio) / 1000)  # Normalize to 1000 tokens
        return occupancy
    
    def collect_metrics(self, 
                       input_text: str,
                       output_text: str,
                       generation_time: float,
                       logits: Optional[List[float]] = None,
                       attention_weights: Optional[List[List[float]]] = None,
                       model_size: str = "medium") -> TelemetryMetrics:
        """Collect comprehensive telemetry metrics"""
        
        generation_id = f"gen_{int(datetime.now().timestamp() * 1000)}"
        
        # Calculate metrics
        repetition_rate = self.calculate_repetition_rate(output_text)
        perplexity_slope = self.calculate_perplexity_slope(logits)
        entropy_metrics = self.calculate_entropy_metrics(output_text)
        attention_entropy = self.calculate_attention_entropy(attention_weights)
        kv_occupancy = self.estimate_kv_occupancy(
            len(input_text.split()), 
            len(output_text.split()), 
            model_size
        )
        
        metrics = TelemetryMetrics(
            generation_id=generation_id,
            timestamp=datetime.now().isoformat(),
            input_length=len(input_text.split()),
            output_length=len(output_text.split()),
            total_tokens=len(input_text.split()) + len(output_text.split()),
            repetition_rate=repetition_rate,
            perplexity_slope=perplexity_slope,
            entropy_metrics=entropy_metrics,
            attention_entropy=attention_entropy,
            kv_occupancy=kv_occupancy,
            generation_time=generation_time
        )
        
        # Store in history
        self.metrics_history.append(metrics)
        if perplexity_slope != 0.0:
            self.perplexity_history.append(perplexity_slope)
        if attention_entropy is not None:
            self.attention_history.append(attention_entropy)
        
        return metrics
    
    def get_summary(self, session_id: str = "default") -> TelemetrySummary:
        """Get summary of telemetry metrics"""
        if not self.metrics_history:
            return TelemetrySummary(
                session_id=session_id,
                total_generations=0,
                avg_repetition_rate=0.0,
                avg_perplexity_slope=0.0,
                avg_entropy=0.0,
                avg_attention_entropy=None,
                avg_kv_occupancy=None,
                total_generation_time=0.0,
                metrics_history=[]
            )
        
        # Calculate averages
        avg_repetition_rate = np.mean([m.repetition_rate for m in self.metrics_history])
        avg_perplexity_slope = np.mean([m.perplexity_slope for m in self.metrics_history])
        avg_entropy = np.mean([m.entropy_metrics["word_entropy"] for m in self.metrics_history])
        avg_attention_entropy = np.mean(self.attention_history) if self.attention_history else None
        avg_kv_occupancy = np.mean([m.kv_occupancy for m in self.metrics_history if m.kv_occupancy is not None])
        total_generation_time = sum([m.generation_time for m in self.metrics_history])
        
        return TelemetrySummary(
            session_id=session_id,
            total_generations=len(self.metrics_history),
            avg_repetition_rate=avg_repetition_rate,
            avg_perplexity_slope=avg_perplexity_slope,
            avg_entropy=avg_entropy,
            avg_attention_entropy=avg_attention_entropy,
            avg_kv_occupancy=avg_kv_occupancy,
            total_generation_time=total_generation_time,
            metrics_history=list(self.metrics_history)
        )
    
    def export_telemetry(self, filename: str):
        """Export telemetry data to JSON"""
        summary = self.get_summary()
        
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "telemetry_summary": asdict(summary),
            "raw_metrics": [asdict(metric) for metric in self.metrics_history]
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported telemetry data to {filename}")
    
    def reset(self):
        """Reset telemetry collector"""
        self.metrics_history.clear()
        self.perplexity_history.clear()
        self.attention_history.clear()
        logger.info("Telemetry collector reset")

# Example usage and testing
if __name__ == "__main__":
    # Test telemetry collector
    collector = TelemetryCollector()
    
    # Simulate some generations
    test_inputs = [
        "Tell me about artificial intelligence",
        "Write a short story about a robot",
        "Explain quantum computing"
    ]
    
    test_outputs = [
        "Artificial intelligence is a field of computer science that focuses on creating machines that can perform tasks that typically require human intelligence. These tasks include learning, reasoning, problem-solving, and decision-making.",
        "Once upon a time, there was a robot named Alex who lived in a small workshop. Alex spent his days learning about the world through books and conversations with his creator, Dr. Sarah.",
        "Quantum computing is a revolutionary approach to computation that leverages the principles of quantum mechanics. Unlike classical computers that use bits, quantum computers use quantum bits or qubits."
    ]
    
    for i, (input_text, output_text) in enumerate(zip(test_inputs, test_outputs)):
        metrics = collector.collect_metrics(
            input_text=input_text,
            output_text=output_text,
            generation_time=1.5 + i * 0.2,
            model_size="medium"
        )
        
        print(f"Generation {i+1}:")
        print(f"  Repetition Rate: {metrics.repetition_rate:.3f}")
        print(f"  Word Entropy: {metrics.entropy_metrics['word_entropy']:.3f}")
        print(f"  Lexical Diversity: {metrics.entropy_metrics['lexical_diversity']:.3f}")
        print(f"  KV Occupancy: {metrics.kv_occupancy:.3f}")
        print()
    
    # Get summary
    summary = collector.get_summary("test_session")
    print(f"Session Summary:")
    print(f"  Total Generations: {summary.total_generations}")
    print(f"  Avg Repetition Rate: {summary.avg_repetition_rate:.3f}")
    print(f"  Avg Entropy: {summary.avg_entropy:.3f}")
    print(f"  Total Time: {summary.total_generation_time:.2f}s")
