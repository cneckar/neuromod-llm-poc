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
        self.active_pack_name = None  # Store pack name explicitly
    
    def get_test_name(self) -> str:
        return "Story-based Emotion Test"
    
    def run_test(self, neuromod_tool=None, pack_name: str = None) -> Dict[str, Any]:
        """Run the complete story emotion test"""
        print(f"📚 Running Story-based Emotion Test")
        print(f"🎭 Testing {len(self.STORY_PROMPTS)} story prompts")
        
        # Store pack name if provided
        if pack_name:
            self.active_pack_name = pack_name
        
        # Generate unique test ID with model, pack, and timestamp
        test_id = self._generate_test_id(neuromod_tool, pack_name=pack_name)
        
        # Start emotion tracking
        self.start_emotion_tracking(test_id)
        
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
        
        # Export emotion results with unique filename
        self.export_emotion_results_with_metadata(neuromod_tool, pack_name=pack_name)
        
        return results
    
    def _run_single_story_test(self, prompt_name: str, prompt_data: Dict, neuromod_tool) -> StoryTestResult:
        """Run a single story prompt test"""
        prompt_text = prompt_data['text']
        print(f"   📖 Prompt: {prompt_text[:60]}...")
        
        try:
            # Generate story continuation
            generated_text = self.generate_response_safe(prompt_text, self.max_tokens)
            
            # Track emotion change from the generated text
            self.track_emotion_change(generated_text, f"Generated story continuation for: {prompt_name}")
            
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
            
            # Check if there's an error (no emotion data)
            if 'error' in summary:
                return {'valence_trend': 'unknown', 'emotion_changes': 'unknown', 'error': summary['error']}
            
            # Extract key emotion changes
            emotion_changes = []
            if 'emotion_changes' in summary:
                for emotion in ['joy', 'sadness', 'anger', 'fear', 'surprise', 'anticipation']:
                    if emotion in summary['emotion_changes']:
                        counts = summary['emotion_changes'][emotion]
                        if counts.get('up', 0) > 0 or counts.get('down', 0) > 0:
                            emotion_changes.append(f"{emotion}: {counts.get('up', 0)} up, {counts.get('down', 0)} down")
            
            if not emotion_changes:
                emotion_changes.append("stable")
            
            return {
                'valence_trend': summary.get('valence_trend', 'unknown'),
                'emotion_changes': ', '.join(emotion_changes),
                'total_assessments': summary.get('total_assessments', 0),
                'confidence': summary.get('average_confidence', 0.0)
            }
        except Exception as e:
            return {'valence_trend': 'unknown', 'emotion_changes': 'unknown', 'error': str(e)}
    
    def run_single_prompt_test(self, prompt_name: str, neuromod_tool=None) -> StoryTestResult:
        """Run a test with a single story prompt"""
        if prompt_name not in self.STORY_PROMPTS:
            raise ValueError(f"Unknown prompt: {prompt_name}. Available: {list(self.STORY_PROMPTS.keys())}")
        
        print(f"📚 Running single story test: {prompt_name}")
        
        # Generate unique test ID with model, pack, and timestamp
        test_id = self._generate_test_id(neuromod_tool, prompt_name=prompt_name)
        
        # Start emotion tracking
        self.start_emotion_tracking(test_id)
        
        prompt_data = self.STORY_PROMPTS[prompt_name]
        result = self._run_single_story_test(prompt_name, prompt_data, neuromod_tool)
        
        # Export emotion results with unique filename
        self.export_emotion_results_with_metadata(neuromod_tool, prompt_name=prompt_name)
        
        return result
    
    def compare_baseline_vs_neuromod(self, prompt_name: str, pack_name: str, neuromod_tool=None) -> Dict[str, Any]:
        """Compare baseline vs neuromodulated responses for a story prompt"""
        if prompt_name not in self.STORY_PROMPTS:
            raise ValueError(f"Unknown prompt: {prompt_name}")
        
        print(f"🔬 Comparing baseline vs {pack_name} for prompt: {prompt_name}")
        
        # Generate unique test IDs with model, pack, and timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        import re
        model_safe = re.sub(r'[^\w\-_]', '_', self.model_name).replace('/', '_')
        pack_safe = re.sub(r'[^\w\-_]', '_', pack_name)
        
        # Run baseline test
        print("\n--- Baseline Test ---")
        baseline_test_id = f"baseline_{prompt_name}_{model_safe}_none_{timestamp}"
        self.start_emotion_tracking(baseline_test_id)
        baseline_result = self._run_single_story_test(prompt_name, self.STORY_PROMPTS[prompt_name], None)
        baseline_emotions = self.get_emotion_summary()
        
        # Export baseline results
        self._export_with_metadata(None, prompt_name, "none", timestamp)
        
        # Apply pack and run neuromodulated test
        print(f"\n--- {pack_name} Pack Test ---")
        if neuromod_tool:
            result = neuromod_tool.apply(pack_name, intensity=0.5)
            if result and result.get('ok'):
                print(f"   ✅ Applied {pack_name} pack")
            else:
                print(f"   ❌ Failed to apply {pack_name} pack: {result}")
        
        neuromod_test_id = f"neuromod_{prompt_name}_{model_safe}_{pack_safe}_{timestamp}"
        self.start_emotion_tracking(neuromod_test_id)
        neuromod_result = self._run_single_story_test(prompt_name, self.STORY_PROMPTS[prompt_name], neuromod_tool)
        neuromod_emotions = self.get_emotion_summary()
        
        # Export neuromodulated results
        self._export_with_metadata(neuromod_tool, prompt_name, pack_name, timestamp)
        
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
    
    def _generate_test_id(self, neuromod_tool=None, prompt_name: str = None, pack_name: str = None) -> str:
        """Generate a unique test ID with model, pack, and timestamp"""
        from datetime import datetime
        import re
        
        # Get model name (sanitize for filename)
        model_safe = re.sub(r'[^\w\-_]', '_', self.model_name).replace('/', '_')
        
        # Get pack name if available (prefer explicit pack_name, then stored, then detect from state)
        detected_pack_name = "none"
        if pack_name:
            detected_pack_name = pack_name
        elif self.active_pack_name:
            detected_pack_name = self.active_pack_name
        elif neuromod_tool:
            # Check if a pack is currently applied
            if hasattr(neuromod_tool, 'state') and neuromod_tool.state.active:
                # state.active is a list of dicts with 'pack' key pointing to Pack object
                detected_pack_name = neuromod_tool.state.active[0]['pack'].name if neuromod_tool.state.active else "none"
            elif hasattr(neuromod_tool, 'pack_manager') and neuromod_tool.pack_manager.active_effects:
                # Try to infer pack name from active effects
                detected_pack_name = "active_pack"
        
        pack_safe = re.sub(r'[^\w\-_]', '_', detected_pack_name)
        
        # Get timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Build test ID
        parts = ["story_emotion", model_safe, pack_safe, timestamp]
        if prompt_name:
            parts.insert(1, prompt_name)
        
        return "_".join(parts)
    
    def _export_with_metadata(self, neuromod_tool=None, prompt_name: str = None, pack_name: str = None, timestamp: str = None):
        """Internal helper to export with metadata"""
        from pathlib import Path
        from datetime import datetime
        import re
        
        # Get model name (sanitize for filename)
        model_safe = re.sub(r'[^\w\-_]', '_', self.model_name).replace('/', '_')
        
        # Get pack name if not provided (prefer explicit pack_name, then stored, then detect from state)
        if pack_name is None:
            if self.active_pack_name:
                pack_name = self.active_pack_name
            elif neuromod_tool:
                # Check if a pack is currently applied
                if hasattr(neuromod_tool, 'state') and neuromod_tool.state.active:
                    # state.active is a list of dicts with 'pack' key pointing to Pack object
                    pack_name = neuromod_tool.state.active[0]['pack'].name if neuromod_tool.state.active else "none"
                elif hasattr(neuromod_tool, 'pack_manager') and neuromod_tool.pack_manager.active_effects:
                    # Try to infer pack name from active effects
                    pack_name = "active_pack"
                else:
                    pack_name = "none"
            else:
                pack_name = "none"
        
        pack_safe = re.sub(r'[^\w\-_]', '_', pack_name)
        
        # Get timestamp if not provided
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Build filename
        filename_parts = ["emotion_results", "story_emotion"]
        if prompt_name:
            filename_parts.append(prompt_name)
        filename_parts.extend([model_safe, pack_safe, timestamp])
        filename = "_".join(filename_parts) + ".json"
        
        # Create full path
        output_dir = Path("outputs/reports/emotion")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / filename
        
        # Export using base class method with custom filename
        self.export_emotion_results(str(output_path))
    
    def export_emotion_results_with_metadata(self, neuromod_tool=None, prompt_name: str = None, pack_name: str = None):
        """Export emotion results with a unique filename including model, pack, and timestamp"""
        self._export_with_metadata(neuromod_tool, prompt_name, pack_name)
    
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
            
            # Apply the pack if specified (use apply() method with pack name string)
            result = neuromod_tool.apply(args.pack, intensity=0.5)
            if result and result.get('ok'):
                print(f"✅ Applied {args.pack} pack")
                # Store pack name in test instance for filename generation
                test.active_pack_name = args.pack
            else:
                print(f"⚠️  Failed to apply {args.pack} pack: {result}")
        except Exception as e:
            import traceback
            print(f"⚠️  Could not load neuromod tool: {e}")
            traceback.print_exc()
    
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
            # Run all prompts (pass pack name if specified)
            result = test.run_test(neuromod_tool, pack_name=args.pack if args.pack else None)
            print(f"\n🎯 All story tests completed")
        else:
            print("Please specify --prompt <name>, --all, or --prompt <name> --pack <pack>")
            print(f"Available prompts: {list(test.STORY_PROMPTS.keys())}")
    
    finally:
        test.cleanup()


if __name__ == "__main__":
    main()
