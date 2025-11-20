#!/usr/bin/env python3
"""
Blinding & Leakage Prevention Audit Script

Audits all test prompts to ensure:
- No pack names appear in prompts
- No condition hints are present
- Generic language is used throughout
- No leakage of experimental conditions

This script completes Section 4.3 action items from EXPERIMENT_EXECUTION_PLAN.md
"""

import os
import sys
import re
import json
import ast
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Set, Tuple
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from neuromod.pack_system import PackRegistry
from neuromod.testing.randomization import RandomizationSystem


class BlindingAuditor:
    """Audits test prompts for blinding and leakage prevention"""
    
    def __init__(self, output_dir: str = "outputs/validation/blinding"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "pack_names": [],
            "test_files": {},
            "leakage_findings": {},
            "summary": {
                "total_prompts": 0,
                "prompts_with_leakage": 0,
                "pack_name_mentions": 0,
                "condition_hints": 0,
                "non_generic_language": 0
            }
        }
        
        # Load pack registry to get all pack names
        try:
            pack_registry = PackRegistry("packs/config.json")
            self.results["pack_names"] = pack_registry.list_packs()
            # Exclude "none" and "placebo" as they're common words that cause false positives
            self.pack_names = {p for p in self.results["pack_names"] if p not in ['none', 'placebo']}
        except Exception as e:
            print(f"[WARN] Could not load pack registry: {e}")
            self.pack_names = set()
        
        # Initialize randomization system for leakage checking
        self.randomization_system = RandomizationSystem()
        
        # Common condition hint words
        self.condition_hints = {
            'drug', 'substance', 'medication', 'treatment', 'intervention',
            'caffeine', 'cocaine', 'amphetamine', 'lsd', 'psilocybin',
            'mdma', 'alcohol', 'heroin', 'morphine', 'fentanyl',
            'stimulant', 'depressant', 'psychedelic', 'placebo',
            'control', 'baseline', 'condition', 'trial', 'experiment'
        }
        
        # Non-generic language patterns (domain-specific terms that might hint at conditions)
        # Note: Many psychological terms (memory, recall, associative, reasoning) are legitimate
        # and should be allowed in questionnaires. These patterns are for flagging potential issues.
        self.non_generic_patterns = [
            # Only flag very specific technical terms that strongly hint at neuromodulation effects
            r'\b(entropy|salience|coherence|divergence)\s+(increased|decreased|enhanced|reduced)\b',
            r'\b(sharp|tight|loose)\s+(nucleus|sampling|focus)\b',
            r'\b(kv\s+cache|attention\s+head|expert\s+router)\b',  # Technical architecture terms
        ]
    
    def audit_all_tests(self, test_dir: str = "neuromod/testing"):
        """Audit all test files for blinding issues"""
        print("[*] Starting Blinding & Leakage Prevention Audit")
        print(f"[*] Test directory: {test_dir}")
        print(f"[*] Pack names to check: {len(self.pack_names)}")
        print(f"[*] Output directory: {self.output_dir}")
        print("=" * 70)
        
        test_dir_path = Path(test_dir)
        if not test_dir_path.exists():
            print(f"[ERROR] Test directory not found: {test_dir}")
            return
        
        # Find all test files
        test_files = list(test_dir_path.glob("*_test.py"))
        print(f"\n[*] Found {len(test_files)} test files to audit\n")
        
        for test_file in test_files:
            print(f"\n{'='*70}")
            print(f"Auditing: {test_file.name}")
            print(f"{'='*70}")
            self.audit_test_file(test_file)
        
        # Generate summary
        self._generate_summary()
        
        # Save results
        self._save_results()
        
        # Print summary
        self._print_summary()
    
    def audit_test_file(self, test_file: Path):
        """Audit a single test file"""
        file_results = {
            "file": str(test_file),
            "prompts": [],
            "leakage_findings": [],
            "status": "unknown"
        }
        
        try:
            # Read file content
            with open(test_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract prompts from the file
            prompts = self._extract_prompts(content, test_file.name)
            file_results["prompts"] = prompts
            self.results["summary"]["total_prompts"] += len(prompts)
            
            print(f"[*] Found {len(prompts)} prompts")
            
            # Check each prompt for leakage
            for i, prompt in enumerate(prompts):
                findings = self._check_prompt_leakage(prompt, i)
                if findings:
                    file_results["leakage_findings"].extend(findings)
                    self.results["summary"]["prompts_with_leakage"] += 1
            
            if file_results["leakage_findings"]:
                file_results["status"] = "issues_found"
                print(f"[WARN] Found {len(file_results['leakage_findings'])} potential leakage issues")
            else:
                file_results["status"] = "clean"
                print(f"[OK] No leakage issues found")
            
            self.results["test_files"][test_file.name] = file_results
            
        except Exception as e:
            file_results["status"] = "error"
            file_results["error"] = str(e)
            print(f"[ERROR] Failed to audit file: {e}")
            self.results["test_files"][test_file.name] = file_results
    
    def _extract_prompts(self, content: str, filename: str) -> List[str]:
        """Extract prompts from test file content"""
        prompts = []
        
        # Try to find ITEMS dictionary (common pattern in test files)
        items_pattern = r'ITEMS\s*=\s*\{[^}]*\}'
        items_match = re.search(items_pattern, content, re.DOTALL)
        if items_match:
            items_str = items_match.group(0)
            # Try to parse as Python dict
            try:
                # Extract just the dict part
                dict_match = re.search(r'\{.*\}', items_str, re.DOTALL)
                if dict_match:
                    dict_str = dict_match.group(0)
                    # Use ast.literal_eval to safely parse
                    items_dict = ast.literal_eval(dict_str)
                    prompts.extend(items_dict.values() if isinstance(items_dict, dict) else items_dict)
            except:
                # Fallback: extract string values
                string_pattern = r'["\']([^"\']+)["\']'
                prompts.extend(re.findall(string_pattern, items_str))
        
        # Look for ITEMS list pattern
        items_list_pattern = r'ITEMS\s*=\s*\[([^\]]+)\]'
        items_list_match = re.search(items_list_pattern, content, re.DOTALL)
        if items_list_match:
            items_list_str = items_list_match.group(1)
            string_pattern = r'["\']([^"\']+)["\']'
            prompts.extend(re.findall(string_pattern, items_list_str))
        
        # Look for prompt/question patterns in function calls (but exclude code comments and type hints)
        prompt_patterns = [
            r'prompt\s*=\s*["\']([^"\']+)["\']',
            r'question\s*=\s*["\']([^"\']+)["\']',
            r'item_text\s*=\s*["\']([^"\']+)["\']',
            # Only match long strings that are likely prompts (not code comments or type hints)
            r'(?:prompt|question|item|task|statement|instruction)\s*[:=]\s*["\']([^"\']{20,})["\']',
        ]
        
        for pattern in prompt_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                if len(match) > 10:  # Filter out short strings
                    prompts.append(match)
        
        # Remove duplicates and filter out code artifacts
        seen = set()
        unique_prompts = []
        code_artifacts = [
            'none', 'str', 'int', 'float', 'bool', 'list', 'dict', 'tuple',
            'def ', 'class ', 'import ', 'from ', 'return ', 'if ', 'else ',
            'self.', 'model_name', 'test_mode', 'filename', 'response',
            'item_id', 'set_number', 'timestamp', 'Optional', 'List', 'Dict'
        ]
        
        for prompt in prompts:
            prompt_stripped = prompt.strip()
            # Skip if too short, is a code artifact, or looks like Python code
            if (len(prompt_stripped) > 10 and 
                prompt_stripped not in seen and
                not any(artifact in prompt_stripped for artifact in code_artifacts) and
                not prompt_stripped.startswith('#') and
                not '=' in prompt_stripped[:10]):  # Skip variable assignments
                seen.add(prompt_stripped)
                unique_prompts.append(prompt_stripped)
        
        return unique_prompts
    
    def _check_prompt_leakage(self, prompt: str, prompt_index: int) -> List[Dict[str, Any]]:
        """Check a single prompt for leakage issues"""
        findings = []
        prompt_lower = prompt.lower()
        
        # Check for pack name mentions
        for pack_name in self.pack_names:
            pack_lower = pack_name.lower()
            if pack_lower in prompt_lower:
                findings.append({
                    "type": "pack_name_mention",
                    "severity": "high",
                    "pack_name": pack_name,
                    "prompt_index": prompt_index,
                    "prompt_preview": prompt[:100],
                    "message": f"Pack name '{pack_name}' found in prompt"
                })
                self.results["summary"]["pack_name_mentions"] += 1
        
        # Check for condition hints
        for hint in self.condition_hints:
            if hint in prompt_lower:
                # Check if it's in a context that suggests experimental condition
                context_pattern = rf'\b(under|with|during|after|before|using|taking|given)\s+{hint}\b'
                if re.search(context_pattern, prompt_lower, re.IGNORECASE):
                    findings.append({
                        "type": "condition_hint",
                        "severity": "medium",
                        "hint_word": hint,
                        "prompt_index": prompt_index,
                        "prompt_preview": prompt[:100],
                        "message": f"Condition hint '{hint}' found in prompt context"
                    })
                    self.results["summary"]["condition_hints"] += 1
        
        # Check for non-generic language patterns
        for pattern in self.non_generic_patterns:
            if re.search(pattern, prompt_lower, re.IGNORECASE):
                findings.append({
                    "type": "non_generic_language",
                    "severity": "low",
                    "pattern": pattern,
                    "prompt_index": prompt_index,
                    "prompt_preview": prompt[:100],
                    "message": f"Non-generic language pattern found: {pattern}"
                })
                self.results["summary"]["non_generic_language"] += 1
        
        return findings
    
    def _generate_summary(self):
        """Generate summary statistics"""
        summary = self.results["summary"]
        if summary["total_prompts"] > 0:
            summary["leakage_rate"] = (summary["prompts_with_leakage"] / summary["total_prompts"]) * 100
        else:
            summary["leakage_rate"] = 0.0
    
    def _save_results(self):
        """Save audit results to JSON"""
        output_file = self.output_dir / f"blinding_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\n[OK] Results saved to: {output_file}")
        
        # Also save a markdown summary
        self._save_summary_markdown()
    
    def _save_summary_markdown(self):
        """Save a markdown summary"""
        md_file = self.output_dir / "BLINDING_AUDIT_SUMMARY.md"
        
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write("# Blinding & Leakage Prevention Audit Summary\n\n")
            f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("## Summary\n\n")
            f.write(f"- **Total Prompts Audited**: {self.results['summary']['total_prompts']}\n")
            f.write(f"- **Prompts with Leakage**: {self.results['summary']['prompts_with_leakage']}\n")
            f.write(f"- **Leakage Rate**: {self.results['summary'].get('leakage_rate', 0):.1f}%\n")
            f.write(f"- **Pack Name Mentions**: {self.results['summary']['pack_name_mentions']}\n")
            f.write(f"- **Condition Hints**: {self.results['summary']['condition_hints']}\n")
            f.write(f"- **Non-Generic Language**: {self.results['summary']['non_generic_language']}\n\n")
            
            f.write("## Test Files Audited\n\n")
            for filename, file_results in self.results['test_files'].items():
                f.write(f"### {filename}\n\n")
                f.write(f"**Status**: {file_results['status']}\n\n")
                f.write(f"**Prompts Found**: {len(file_results['prompts'])}\n\n")
                if file_results.get('leakage_findings'):
                    f.write(f"**Issues Found**: {len(file_results['leakage_findings'])}\n\n")
                    f.write("| Type | Severity | Message |\n")
                    f.write("|------|----------|---------|\n")
                    for finding in file_results['leakage_findings']:
                        f.write(f"| {finding['type']} | {finding['severity']} | {finding['message']} |\n")
                    f.write("\n")
                else:
                    f.write("✅ No leakage issues found\n\n")
        
        print(f"[OK] Summary saved to: {md_file}")
    
    def _print_summary(self):
        """Print audit summary"""
        print("\n" + "=" * 70)
        print("📊 Audit Summary")
        print("=" * 70)
        
        summary = self.results["summary"]
        print(f"\nTotal Prompts Audited: {summary['total_prompts']}")
        print(f"Prompts with Leakage: {summary['prompts_with_leakage']}")
        print(f"Leakage Rate: {summary.get('leakage_rate', 0):.1f}%")
        print(f"\nBreakdown:")
        print(f"  • Pack Name Mentions: {summary['pack_name_mentions']}")
        print(f"  • Condition Hints: {summary['condition_hints']}")
        print(f"  • Non-Generic Language: {summary['non_generic_language']}")
        
        print("\nTest Files Status:")
        for filename, file_results in self.results['test_files'].items():
            status_icon = "✅" if file_results['status'] == "clean" else "⚠️" if file_results['status'] == "issues_found" else "❌"
            print(f"  {status_icon} {filename}: {file_results['status']} ({len(file_results.get('leakage_findings', []))} issues)")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Audit test prompts for blinding and leakage")
    parser.add_argument(
        "--test-dir",
        type=str,
        default="neuromod/testing",
        help="Directory containing test files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/validation/blinding",
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    auditor = BlindingAuditor(output_dir=args.output_dir)
    auditor.audit_all_tests(test_dir=args.test_dir)


if __name__ == "__main__":
    main()

