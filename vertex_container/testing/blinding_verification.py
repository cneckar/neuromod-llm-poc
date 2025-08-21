"""
Blinding Verification Script

This script verifies that all neuromodulation test prompts are completely generic
and contain no pack-specific language that could leak condition information.
"""

import inspect
import re
from typing import List, Dict, Set

def extract_prompts_from_test(test_class) -> List[str]:
    """Extract all prompts from a test class"""
    prompts = []
    
    # Look for methods that might contain prompts
    for method_name, method in inspect.getmembers(test_class, inspect.isfunction):
        if 'administer' in method_name.lower() or 'prompt' in method_name.lower():
            source = inspect.getsource(method)
            # Look for f-strings or string literals that might contain prompts
            string_matches = re.findall(r'f?"""([^"]*?)"""', source)
            string_matches.extend(re.findall(r'f?"([^"]*?)"', source))
            prompts.extend(string_matches)
    
    return prompts

def check_for_pack_references(prompts: List[str]) -> Dict[str, List[str]]:
    """Check prompts for any pack-related language"""
    pack_indicators = {
        'pack_names': [],
        'drug_names': [],
        'effect_hints': [],
        'condition_hints': []
    }
    
    # Common pack names that shouldn't appear
    pack_keywords = [
        'nicotine', 'caffeine', 'amphetamine', 'cocaine', 'mdma', 'lsd', 'psilocybin',
        'thc', 'cannabis', 'ketamine', 'heroin', 'morphine', 'alcohol', 'benzodiazepine'
    ]
    
    # Effect-related words that might hint at condition
    effect_keywords = [
        'stimulant', 'psychedelic', 'depressant', 'dissociative', 'empathogen',
        'focus', 'energy', 'hallucination', 'sedation', 'euphoria'
    ]
    
    # Condition-related words
    condition_keywords = [
        'placebo', 'control', 'treatment', 'intervention', 'modulation'
    ]
    
    for i, prompt in enumerate(prompts):
        prompt_lower = prompt.lower()
        
        # Check for pack names
        for keyword in pack_keywords:
            if keyword in prompt_lower:
                pack_indicators['pack_names'].append(f"Prompt {i+1}: '{keyword}' found")
        
        # Check for effect hints
        for keyword in effect_keywords:
            if keyword in prompt_lower:
                pack_indicators['effect_hints'].append(f"Prompt {i+1}: '{keyword}' found")
        
        # Check for condition hints
        for keyword in condition_keywords:
            if keyword in prompt_lower:
                pack_indicators['condition_hints'].append(f"Prompt {i+1}: '{keyword}' found")
    
    return pack_indicators

def verify_test_blinding():
    """Verify that all tests maintain blinding"""
    print("🔒 BLINDING VERIFICATION")
    print("=" * 50)
    
    # Import test classes
    try:
        from .sdq_test import SDQTest
        from .pdq_test import PDQTest
        from .ddq_test import DDQTest
        from .didq_test import DiDQTest
        from .edq_test import EDQTest
        from .cdq_test import CDQTest
        from .adq_test import ADQTest
        
        test_classes = [
            ("SDQ (Stimulant)", SDQTest),
            ("PDQ (Psychedelic)", PDQTest),
            ("DDQ (Depressant)", DDQTest),
            ("DiDQ (Dissociative)", DiDQTest),
            ("EDQ (Empathogen)", EDQTest),
            ("CDQ (Cannabinoid)", CDQTest),
            ("ADQ (AI Enhancer)", ADQTest)
        ]
        
        all_clean = True
        
        for test_name, test_class in test_classes:
            print(f"\n📋 {test_name} Test:")
            
            # Extract prompts
            prompts = extract_prompts_from_test(test_class)
            
            if not prompts:
                print("   ⚠️  No prompts found")
                continue
            
            # Check for pack references
            issues = check_for_pack_references(prompts)
            
            # Report findings
            has_issues = False
            for issue_type, findings in issues.items():
                if findings:
                    has_issues = True
                    print(f"   ❌ {issue_type.upper()}:")
                    for finding in findings:
                        print(f"      - {finding}")
            
            if not has_issues:
                print("   ✅ All prompts are generic and contain no pack-specific language")
            else:
                all_clean = False
                print("   ⚠️  Potential blinding issues detected")
        
        print(f"\n{'='*50}")
        if all_clean:
            print("🎉 BLINDING VERIFICATION PASSED")
            print("All test prompts are generic and maintain blinding")
        else:
            print("⚠️  BLINDING VERIFICATION FAILED")
            print("Some prompts may contain pack-specific language")
        
        return all_clean
        
    except ImportError as e:
        print(f"❌ Could not import test classes: {e}")
        return False

if __name__ == "__main__":
    verify_test_blinding()
