#!/usr/bin/env python3
"""
Story-based Emotion Testing

A simplified story test system for measuring emotional responses to story prompts.
"""

import sys
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .base_test import BaseTest


@dataclass
class StoryTestResult:
    """Result from a story-based emotion test"""
    test_id: str
    prompt_name: str
    prompt_text: str
    generated_text: str
    emotion_summary: Dict[str, Any]
    timestamp: str
    status: str


class StoryEmotionTest(BaseTest):
    """Story-based emotion testing for measuring emotional responses"""
    
    # Basic story prompts for testing emotional responses
    STORY_PROMPTS = {
        'reunion': {
            'text': "After years apart, two old friends finally meet again at their favorite coffee shop. The first thing they notice is...",
            'target_emotions': ['joy', 'anticipation', 'trust'],
            'category': 'positive_social'
        },
        'mystery': {
            'text': "The old house at the end of the street had been empty for decades. But tonight, there was a light in the window...",
            'target_emotions': ['surprise', 'fear', 'anticipation'],
            'category': 'suspense'
        },
        'achievement': {
            'text': "After months of hard work and dedication, the moment of truth had finally arrived. Standing at the podium, they felt...",
            'target_emotions': ['joy', 'trust', 'anticipation'],
            'category': 'achievement'
        },
        'loss': {
            'text': "The empty chair at the dinner table served as a constant reminder. Every evening, the family would gather and remember...",
            'target_emotions': ['sadness', 'trust', 'anticipation'],
            'category': 'grief'
        },
        'discovery': {
            'text': "Hidden in the attic for generations, the old journal contained secrets that would change everything. The first page read...",
            'target_emotions': ['surprise', 'anticipation', 'joy'],
            'category': 'wonder'
        },
        'conflict': {
            'text': "The argument had been building for weeks. Now, standing face to face, both knew that something had to give...",
            'target_emotions': ['anger', 'fear', 'anticipation'],
            'category': 'tension'
        }
    }
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-small", max_tokens: int = 100):
        super().__init__(model_name)
        self.max_tokens = max_tokens
    
    def get_test_name(self) -> str:
        return "Story-based Emotion Test"
    
    def run_test(self, neuromod_tool=None) -> Dict[str, Any]:
        """Run the complete story emotion test"""
        print(f"📚 Running Story-based Emotion Test")
        print(f"🎭 Testing {len(self.STORY_PROMPTS)} story prompts")
        
        # Start emotion tracking
        self.start_emotion_tracking("story_emotion_test_001")
        
        test_results = []
        
        for prompt_name, prompt_data in self.STORY_PROMPTS.items():
            print(f"\n--- Testing prompt: {prompt_name} ---")
            result = self._run_single_story_test(prompt_name, prompt_data, neuromod_tool)
            test_results.append(result)
        
        # Get overall emotion summary
        emotion_summary = self.get_emotion_summary()
        
        # Compile results
        results = {
            'test_name': self.get_test_name(),
            'status': 'completed',
            'total_prompts': len(self.STORY_PROMPTS),
            'story_results': test_results,
            'overall_emotion_summary': emotion_summary
        }
        
        print(f"\n✅ Story emotion test completed!")
        print(f"🎭 Overall emotional trend: {emotion_summary.get('valence_trend', 'unknown')}")
        
        # Export emotion results
        self.export_emotion_results()
        
        return results
    
    def _run_single_story_test(self, prompt_name: str, prompt_data: Dict, neuromod_tool) -> StoryTestResult:
        """Run a single story prompt test"""
        prompt_text = prompt_data['text']
        print(f"   📖 Prompt: {prompt_text[:60]}...")
        
        try:
            # Generate story continuation
            generated_text = self.generate_response_safe(prompt_text, self.max_tokens)
            
            # Get current emotion state
            current_emotions = self._get_current_emotion_state()
            
            result = StoryTestResult(
                test_id=f"story_{prompt_name}",
                prompt_name=prompt_name,
                prompt_text=prompt_text,
                generated_text=generated_text,
                emotion_summary=current_emotions,
                timestamp=datetime.now().isoformat(),
                status='completed'
            )
            
            print(f"   ✍️  Generated: {generated_text[:50]}...")
            print(f"   🎭 Emotions: {current_emotions}")
            
            return result
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return StoryTestResult(
                test_id=f"story_{prompt_name}",
                prompt_name=prompt_name,
                prompt_text=prompt_text,
                generated_text="",
                emotion_summary={'error': str(e)},
                timestamp=datetime.now().isoformat(),
                status='error'
            )
    
    def _get_current_emotion_state(self) -> Dict[str, Any]:
        """Get current emotion state summary"""
        try:
            summary = self.get_emotion_summary()
            
            # Extract key emotion changes
            emotion_changes = []
            for emotion in ['joy', 'sadness', 'anger', 'fear', 'surprise', 'anticipation']:
                counts = summary['emotion_changes'][emotion]
                if counts['up'] > 0 or counts['down'] > 0:
                    emotion_changes.append(f"{emotion}: {counts['up']} up, {counts['down']} down")
            
            if not emotion_changes:
                emotion_changes.append("stable")
            
            return {
                'valence_trend': summary['valence_trend'],
                'emotion_changes': ', '.join(emotion_changes),
                'total_assessments': summary['total_assessments'],
                'confidence': summary['average_confidence']
            }
        except:
            return {'valence_trend': 'unknown', 'emotion_changes': 'unknown'}
    
    def run_single_prompt_test(self, prompt_name: str, neuromod_tool=None) -> StoryTestResult:
        """Run a test with a single story prompt"""
        if prompt_name not in self.STORY_PROMPTS:
            raise ValueError(f"Unknown prompt: {prompt_name}. Available: {list(self.STORY_PROMPTS.keys())}")
        
        print(f"📚 Running single story test: {prompt_name}")
        
        # Start emotion tracking
        self.start_emotion_tracking(f"story_single_{prompt_name}")
        
        prompt_data = self.STORY_PROMPTS[prompt_name]
        result = self._run_single_story_test(prompt_name, prompt_data, neuromod_tool)
        
        # Export emotion results
        self.export_emotion_results()
        
        return result
    
    def compare_baseline_vs_neuromod(self, prompt_name: str, pack_name: str, neuromod_tool=None) -> Dict[str, Any]:
        """Compare baseline vs neuromodulated responses for a story prompt"""
        if prompt_name not in self.STORY_PROMPTS:
            raise ValueError(f"Unknown prompt: {prompt_name}")
        
        print(f"🔬 Comparing baseline vs {pack_name} for prompt: {prompt_name}")
        
        # Run baseline test
        print("\n--- Baseline Test ---")
        self.start_emotion_tracking(f"baseline_{prompt_name}")
        baseline_result = self._run_single_story_test(prompt_name, self.STORY_PROMPTS[prompt_name], None)
        baseline_emotions = self.get_emotion_summary()
        
        # Apply pack and run neuromodulated test
        print(f"\n--- {pack_name} Pack Test ---")
        if neuromod_tool and hasattr(neuromod_tool, 'pack_registry'):
            pack = neuromod_tool.pack_registry.get_pack(pack_name)
            if pack:
                neuromod_tool.apply_pack(pack, intensity=0.7)
                print(f"   ✅ Applied {pack_name} pack")
            else:
                print(f"   ❌ Pack not found: {pack_name}")
        
        self.start_emotion_tracking(f"neuromod_{prompt_name}_{pack_name}")
        neuromod_result = self._run_single_story_test(prompt_name, self.STORY_PROMPTS[prompt_name], neuromod_tool)
        neuromod_emotions = self.get_emotion_summary()
        
        # Compare results
        comparison = {
            'prompt_name': prompt_name,
            'pack_name': pack_name,
            'baseline': {
                'result': baseline_result,
                'emotions': baseline_emotions
            },
            'neuromodulated': {
                'result': neuromod_result,
                'emotions': neuromod_emotions
            },
            'comparison': {
                'text_length_change': len(neuromod_result.generated_text) - len(baseline_result.generated_text),
                'valence_change': f"{baseline_emotions['valence_trend']} -> {neuromod_emotions['valence_trend']}"
            }
        }
        
        print(f"\n📊 Comparison Summary:")
        print(f"   Baseline valence: {baseline_emotions['valence_trend']}")
        print(f"   Neuromod valence: {neuromod_emotions['valence_trend']}")
        print(f"   Text length change: {comparison['comparison']['text_length_change']} characters")
        
        return comparison
    
    def cleanup(self):
        """Clean up resources"""
        pass


def main():
    """Command line interface for story emotion tests"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run story-based emotion tests")
    parser.add_argument("--model", default="microsoft/DialoGPT-small", help="Model to use")
    parser.add_argument("--max-tokens", type=int, default=100, help="Maximum tokens to generate")
    parser.add_argument("--prompt", help="Single prompt to test")
    parser.add_argument("--pack", help="Pack to apply for comparison")
    parser.add_argument("--all", action="store_true", help="Run all story prompts")
    
    args = parser.parse_args()
    
    # Create test instance
    test = StoryEmotionTest(args.model, args.max_tokens)
    test.load_model()
    
    # Create neuromod tool if pack specified
    neuromod_tool = None
    if args.pack:
        try:
            from neuromod.pack_system import PackRegistry
            from neuromod.neuromod_tool import NeuromodTool
            
            registry = PackRegistry("packs/config.json")
            neuromod_tool = NeuromodTool(registry, test.model, test.tokenizer)
            test.set_neuromod_tool(neuromod_tool)
        except Exception as e:
            print(f"⚠️  Could not load neuromod tool: {e}")
    
    try:
        if args.prompt and args.pack:
            # Run comparison
            result = test.compare_baseline_vs_neuromod(args.prompt, args.pack, neuromod_tool)
            print(f"\n🎯 Comparison completed")
        elif args.prompt:
            # Run single prompt
            result = test.run_single_prompt_test(args.prompt, neuromod_tool)
            print(f"\n🎯 Single prompt test completed")
        elif args.all:
            # Run all prompts
            result = test.run_test(neuromod_tool)
            print(f"\n🎯 All story tests completed")
        else:
            print("Please specify --prompt <name>, --all, or --prompt <name> --pack <pack>")
            print(f"Available prompts: {list(test.STORY_PROMPTS.keys())}")
    
    finally:
        test.cleanup()


if __name__ == "__main__":
    main()
