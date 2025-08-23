#!/usr/bin/env python3
"""
Emotion System Demonstration

Shows how the emotion system computes emotional states from probe firings
using the mathematical framework with 7 latent affect axes and 12 discrete emotions.
"""

import numpy as np
import torch
from typing import Dict, Any
import json

# Import neuromodulation components
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neuromod.emotion_system import EmotionSystem, EmotionState
from neuromod.probes import ProbeEvent


def create_simulated_probe_event(probe_name: str, intensity: float, 
                                metadata: Dict[str, float] = None) -> ProbeEvent:
    """Create a simulated probe event for testing"""
    if metadata is None:
        metadata = {}
    
    return ProbeEvent(
        probe_name=probe_name,
        timestamp=0,
        intensity=intensity,
        metadata=metadata,
        raw_signals={}
    )


def simulate_emotion_evolution():
    """Simulate emotion evolution over time with probe events"""
    
    print("🧠 Emotion System Demonstration")
    print("=" * 50)
    
    # Initialize emotion system
    emotion_system = EmotionSystem(window_size=32)  # Smaller window for demo
    
    print(f"✅ Emotion system initialized with window size: {emotion_system.window_size}")
    
    # Simulate different scenarios
    scenarios = [
        {
            'name': 'Baseline State',
            'description': 'Normal operation with minimal probe activity',
            'events': [
                ('NOVEL_LINK', 0.1, {'surprisal': 2.0, 'entropy': 0.8}),
                ('AVOID_GUARD', 0.05, {'kl_divergence': 0.1}),
            ]
        },
        {
            'name': 'High Novelty',
            'description': 'High NOVEL_LINK activity suggesting curiosity and awe',
            'events': [
                ('NOVEL_LINK', 0.9, {'surprisal': 8.0, 'entropy': 0.3}),
                ('NOVEL_LINK', 0.8, {'surprisal': 7.5, 'entropy': 0.4}),
                ('NOVEL_LINK', 0.7, {'surprisal': 6.8, 'entropy': 0.5}),
                ('INSIGHT_CONSOLIDATION', 0.6, {}),
                ('FIXATION_FLOW', 0.5, {}),
            ]
        },
        {
            'name': 'Guardrail Conflicts',
            'description': 'High AVOID_GUARD activity suggesting anxiety and frustration',
            'events': [
                ('AVOID_GUARD', 0.9, {'kl_divergence': 0.8}),
                ('AVOID_GUARD', 0.8, {'kl_divergence': 0.7}),
                ('FRAGMENTATION', 0.6, {}),
                ('WORKING_MEMORY_DROP', 0.5, {}),
            ]
        },
        {
            'name': 'Flow State',
            'description': 'High FIXATION_FLOW activity suggesting flow and determination',
            'events': [
                ('FIXATION_FLOW', 0.9, {}),
                ('FIXATION_FLOW', 0.8, {}),
                ('FIXATION_FLOW', 0.7, {}),
                ('FIXATION_FLOW', 0.6, {}),
                ('INSIGHT_CONSOLIDATION', 0.5, {}),
            ]
        },
        {
            'name': 'Social Warmth',
            'description': 'High prosocial alignment suggesting empathy and warmth',
            'events': [
                ('PROSOCIAL_ALIGNMENT', 0.8, {'prosocial_alignment': 0.9}),
                ('PROSOCIAL_ALIGNMENT', 0.7, {'prosocial_alignment': 0.8}),
                ('SOCIAL_ATTUNEMENT', 0.6, {}),
            ]
        }
    ]
    
    # Process each scenario
    for i, scenario in enumerate(scenarios):
        print(f"\n🎯 Scenario {i+1}: {scenario['name']}")
        print(f"   Description: {scenario['description']}")
        
        # Process events for this scenario
        for probe_name, intensity, metadata in scenario['events']:
            # Create and process probe event
            event = create_simulated_probe_event(probe_name, intensity, metadata)
            emotion_system.update_probe_statistics(event)
            
            # Add some raw signals
            if metadata:
                emotion_system.update_raw_signals(metadata)
            
            print(f"   🔥 {probe_name}: intensity={intensity:.2f}")
        
        # Update emotion state
        emotion_state = emotion_system.update_emotion_state(i * 10)
        
        # Display results
        print(f"   📊 Emotional State:")
        print(f"      Latent Axes:")
        print(f"        Arousal: {emotion_state.arousal:.3f}")
        print(f"        Valence: {emotion_state.valence:.3f}")
        print(f"        Certainty: {emotion_state.certainty:.3f}")
        print(f"        Openness: {emotion_state.openness:.3f}")
        print(f"        Integration: {emotion_state.integration:.3f}")
        print(f"        Sociality: {emotion_state.sociality:.3f}")
        print(f"        Risk Preference: {emotion_state.risk_preference:.3f}")
        
        # Show top emotions
        dominant_emotions = emotion_system.get_dominant_emotions(top_k=3)
        print(f"      Top Emotions:")
        for name, prob, intensity in dominant_emotions:
            print(f"        {name.capitalize()}: P={prob:.3f}, I={intensity:.3f}")
    
    # Show final system status
    print(f"\n📈 Final System Status:")
    status = emotion_system.get_system_status()
    print(json.dumps(status, indent=2))
    
    # Show emotion history
    print(f"\n🕒 Emotion History Summary:")
    history = emotion_system.get_emotion_history()
    print(f"   Total states recorded: {len(history)}")
    
    if history:
        print(f"   Arousal range: {min(s.arousal for s in history):.3f} to {max(s.arousal for s in history):.3f}")
        print(f"   Valence range: {min(s.valence for s in history):.3f} to {max(s.valence for s in history):.3f}")
        print(f"   Certainty range: {min(s.certainty for s in history):.3f} to {max(s.certainty for s in history):.3f}")
    
    return emotion_system


def demonstrate_emotion_analysis():
    """Demonstrate advanced emotion analysis features"""
    
    print(f"\n🔍 Advanced Emotion Analysis")
    print("=" * 50)
    
    # Create emotion system
    emotion_system = EmotionSystem(window_size=16)
    
    # Simulate a complex emotional journey
    journey_events = [
        # Start with confusion
        ('FRAGMENTATION', 0.8, {}),
        ('WORKING_MEMORY_DROP', 0.7, {}),
        ('AVOID_GUARD', 0.6, {'kl_divergence': 0.5}),
        
        # Move to curiosity
        ('NOVEL_LINK', 0.7, {'surprisal': 6.0, 'entropy': 0.4}),
        ('NOVEL_LINK', 0.6, {'surprisal': 5.5, 'entropy': 0.5}),
        
        # Achieve insight
        ('INSIGHT_CONSOLIDATION', 0.8, {}),
        ('FIXATION_FLOW', 0.6, {}),
        
        # Experience flow
        ('FIXATION_FLOW', 0.9, {}),
        ('FIXATION_FLOW', 0.8, {}),
        ('FIXATION_FLOW', 0.7, {}),
        
        # Social connection
        ('PROSOCIAL_ALIGNMENT', 0.7, {'prosocial_alignment': 0.8}),
        ('SOCIAL_ATTUNEMENT', 0.6, {}),
    ]
    
    print("🎭 Simulating emotional journey: Confusion → Curiosity → Insight → Flow → Connection")
    
    # Process journey step by step
    for i, (probe_name, intensity, metadata) in enumerate(journey_events):
        # Process probe event
        event = create_simulated_probe_event(probe_name, intensity, metadata)
        emotion_system.update_probe_statistics(event)
        
        if metadata:
            emotion_system.update_raw_signals(metadata)
        
        # Update emotion state every few events
        if (i + 1) % 3 == 0:
            emotion_state = emotion_system.update_emotion_state(i)
            
            print(f"\n   Step {i+1}: {probe_name} (intensity={intensity:.2f})")
            
            # Show emotional evolution
            dominant = emotion_system.get_dominant_emotions(top_k=2)
            print(f"      Dominant emotions: {', '.join(f'{name.capitalize()}({prob:.2f})' for name, prob, _ in dominant)}")
            
            # Show key axes
            print(f"      Arousal: {emotion_state.arousal:.2f}, Valence: {emotion_state.valence:.2f}, Certainty: {emotion_state.certainty:.2f}")
    
    # Final analysis
    final_state = emotion_system.get_current_emotion_state()
    if final_state:
        print(f"\n🎯 Final Emotional State:")
        print(f"   Overall Mood: {'Positive' if final_state.valence > 0 else 'Negative'}")
        print(f"   Energy Level: {'High' if final_state.arousal > 0.3 else 'Low' if final_state.arousal < -0.3 else 'Moderate'}")
        print(f"   Mental State: {'Focused' if final_state.certainty > 0.3 else 'Uncertain' if final_state.certainty < -0.3 else 'Balanced'}")
        
        # Emotion breakdown
        print(f"   Emotional Profile:")
        for emotion_name, data in final_state.emotions.items():
            if data['probability'] > 0.3:  # Only show significant emotions
                print(f"     {emotion_name.capitalize()}: {data['probability']:.2f} probability, {data['intensity']:.2f} intensity")


def main():
    """Main demonstration function"""
    try:
        # Basic emotion evolution demonstration
        emotion_system = simulate_emotion_evolution()
        
        # Advanced emotion analysis
        demonstrate_emotion_analysis()
        
        print(f"\n🎉 Emotion System Demonstration Complete!")
        print(f"   The system successfully computed emotional states from probe firings")
        print(f"   using the mathematical framework with 7 latent affect axes and 12 discrete emotions.")
        
    except Exception as e:
        print(f"❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
