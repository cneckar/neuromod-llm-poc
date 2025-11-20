#!/usr/bin/env python3
"""
Pack Validation Script

Tests loading and application of all neuromodulation packs and documents:
- Pack loading success/failure
- Pack effect compositions
- Application verification (syntax/configuration checks)
- Effect type coverage

This script completes Section 4.2 action items from EXPERIMENT_EXECUTION_PLAN.md
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from neuromod.pack_system import PackRegistry, Pack, EffectConfig
from neuromod.effects import EffectRegistry
from neuromod.neuromod_factory import create_test_neuromod_tool, cleanup_neuromod_tool


class PackValidator:
    """Validates all neuromodulation packs"""
    
    def __init__(self, output_dir: str = "outputs/validation/packs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "packs": {},
            "summary": {
                "total_packs": 0,
                "loaded_successfully": 0,
                "failed_to_load": 0,
                "effect_types_used": set(),
                "total_effects": 0
            }
        }
        self.effect_registry = EffectRegistry()
    
    def validate_all_packs(self, config_path: str = "packs/config.json"):
        """Validate all packs from config file"""
        print("[*] Starting Pack Validation")
        print(f"[*] Config file: {config_path}")
        print(f"[*] Output directory: {self.output_dir}")
        print("=" * 70)
        
        # Load pack registry
        try:
            registry = PackRegistry(config_path)
            pack_names = registry.list_packs()
            self.results["summary"]["total_packs"] = len(pack_names)
            print(f"[*] Found {len(pack_names)} packs to validate\n")
        except Exception as e:
            print(f"[ERROR] Failed to load pack registry: {e}")
            return
        
        # Validate each pack
        for pack_name in pack_names:
            result = self.validate_pack(registry, pack_name)
            self.results["packs"][pack_name] = result
        
        # Generate summary
        self._generate_summary()
        
        # Save results
        self._save_results()
        
        # Print summary
        self._print_summary()
    
    def validate_pack(self, registry: PackRegistry, pack_name: str) -> Dict[str, Any]:
        """Validate a single pack"""
        print(f"\n{'='*70}")
        print(f"Validating pack: {pack_name}")
        print(f"{'='*70}")
        
        result = {
            "pack_name": pack_name,
            "status": "unknown",
            "error": None,
            "pack_info": None,
            "effects": [],
            "effect_validation": {},
            "application_test": None
        }
        
        try:
            # Test 1: Load pack
            print(f"[*] Loading pack...")
            pack = registry.get_pack(pack_name)
            pack_info = registry.get_pack_info(pack_name)
            result["pack_info"] = pack_info
            result["status"] = "loaded"
            self.results["summary"]["loaded_successfully"] += 1
            print(f"[OK] Pack loaded successfully")
            print(f"    Description: {pack_info['description']}")
            print(f"    Effects: {len(pack_info['effects'])}")
            
            # Test 2: Validate each effect
            print(f"[*] Validating effects...")
            for i, effect_data in enumerate(pack_info['effects']):
                effect_name = effect_data['effect']
                effect_result = self._validate_effect(effect_name, effect_data)
                result["effects"].append(effect_data)
                result["effect_validation"][effect_name] = effect_result
                self.results["summary"]["effect_types_used"].add(effect_name)
                self.results["summary"]["total_effects"] += 1
                
                if effect_result["valid"]:
                    print(f"    [{i+1}] ✅ {effect_name} (weight={effect_data.get('weight', 0.5)}, direction={effect_data.get('direction', 'up')})")
                else:
                    print(f"    [{i+1}] ❌ {effect_name}: {effect_result['error']}")
            
            # Test 3: Verify pack structure
            print(f"[*] Verifying pack structure...")
            structure_errors = self._verify_pack_structure(pack)
            if structure_errors:
                print(f"[WARN] Pack structure issues: {structure_errors}")
                result["structure_warnings"] = structure_errors
            else:
                print(f"[OK] Pack structure valid")
            
            # Test 4: Application test (syntax check only, no actual model needed)
            print(f"[*] Testing pack application (syntax check)...")
            application_result = self._test_pack_application(pack)
            result["application_test"] = application_result
            if application_result["can_apply"]:
                print(f"[OK] Pack can be applied (syntax valid)")
            else:
                print(f"[WARN] Pack application issues: {application_result.get('error', 'Unknown')}")
            
            result["status"] = "success"
            
        except Exception as e:
            result["status"] = "failed"
            result["error"] = str(e)
            self.results["summary"]["failed_to_load"] += 1
            print(f"[ERROR] Failed to validate pack: {e}")
        
        return result
    
    def _validate_effect(self, effect_name: str, effect_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that an effect exists and has valid configuration"""
        result = {
            "valid": False,
            "error": None,
            "effect_exists": False
        }
        
        try:
            # Check if effect exists in registry
            if effect_name in self.effect_registry.effects:
                result["effect_exists"] = True
                result["valid"] = True
            else:
                result["error"] = f"Effect '{effect_name}' not found in EffectRegistry"
                result["valid"] = False
            
            # Validate weight
            weight = effect_data.get('weight', 0.5)
            if not (0.0 <= weight <= 1.0):
                result["valid"] = False
                result["error"] = f"Invalid weight: {weight} (must be 0.0-1.0)"
            
            # Validate direction
            direction = effect_data.get('direction', 'up')
            if direction not in ['up', 'down', 'both']:
                result["valid"] = False
                if result["error"]:
                    result["error"] += f"; Invalid direction: {direction}"
                else:
                    result["error"] = f"Invalid direction: {direction}"
            
        except Exception as e:
            result["valid"] = False
            result["error"] = str(e)
        
        return result
    
    def _verify_pack_structure(self, pack: Pack) -> List[str]:
        """Verify pack structure is valid"""
        errors = []
        
        if not pack.name:
            errors.append("Pack name is empty")
        
        if not pack.description:
            errors.append("Pack description is empty")
        
        if not pack.effects:
            errors.append("Pack has no effects")
        
        for i, effect in enumerate(pack.effects):
            if not isinstance(effect, EffectConfig):
                errors.append(f"Effect {i} is not EffectConfig instance")
            if not effect.effect:
                errors.append(f"Effect {i} has no effect name")
            if not (0.0 <= effect.weight <= 1.0):
                errors.append(f"Effect {i} has invalid weight: {effect.weight}")
        
        return errors
    
    def _test_pack_application(self, pack: Pack) -> Dict[str, Any]:
        """Test if pack can be applied (syntax check, no actual model)"""
        result = {
            "can_apply": False,
            "error": None
        }
        
        try:
            # Just verify the pack structure is correct for application
            # We don't need an actual model for this test
            if not pack.effects:
                result["error"] = "Pack has no effects to apply"
                return result
            
            # Check that all effects can be instantiated
            for effect_config in pack.effects:
                if effect_config.effect not in self.effect_registry.effects:
                    result["error"] = f"Effect '{effect_config.effect}' not available"
                    return result
            
            result["can_apply"] = True
            
        except Exception as e:
            result["error"] = str(e)
        
        return result
    
    def _generate_summary(self):
        """Generate summary statistics"""
        summary = self.results["summary"]
        summary["effect_types_used"] = list(summary["effect_types_used"])
        summary["success_rate"] = (
            summary["loaded_successfully"] / summary["total_packs"] * 100
            if summary["total_packs"] > 0 else 0
        )
    
    def _save_results(self):
        """Save validation results to JSON"""
        output_file = self.output_dir / f"pack_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Convert set to list for JSON serialization
        results_copy = json.loads(json.dumps(self.results, default=str))
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results_copy, f, indent=2, default=str)
        
        print(f"\n[OK] Results saved to: {output_file}")
        
        # Also save a markdown summary
        self._save_summary_markdown()
    
    def _save_summary_markdown(self):
        """Save a markdown summary"""
        md_file = self.output_dir / "PACK_VALIDATION_SUMMARY.md"
        
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write("# Pack Validation Summary\n\n")
            f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("## Summary\n\n")
            f.write(f"- **Total Packs**: {self.results['summary']['total_packs']}\n")
            f.write(f"- **Loaded Successfully**: {self.results['summary']['loaded_successfully']}\n")
            f.write(f"- **Failed to Load**: {self.results['summary']['failed_to_load']}\n")
            f.write(f"- **Success Rate**: {self.results['summary'].get('success_rate', 0):.1f}%\n")
            f.write(f"- **Total Effects**: {self.results['summary']['total_effects']}\n")
            f.write(f"- **Unique Effect Types**: {len(self.results['summary']['effect_types_used'])}\n\n")
            
            f.write("## Effect Types Used\n\n")
            for effect_type in sorted(self.results['summary']['effect_types_used']):
                f.write(f"- {effect_type}\n")
            
            f.write("\n## Pack Details\n\n")
            for pack_name, pack_result in self.results['packs'].items():
                f.write(f"### {pack_name}\n\n")
                f.write(f"**Status**: {pack_result['status']}\n\n")
                if pack_result.get('pack_info'):
                    f.write(f"**Description**: {pack_result['pack_info']['description']}\n\n")
                    f.write(f"**Effects**: {len(pack_result['pack_info']['effects'])}\n\n")
                    f.write("| Effect | Weight | Direction | Valid |\n")
                    f.write("|--------|--------|-----------|-------|\n")
                    for effect_data in pack_result['pack_info']['effects']:
                        effect_name = effect_data['effect']
                        validation = pack_result['effect_validation'].get(effect_name, {})
                        valid = "✅" if validation.get('valid', False) else "❌"
                        f.write(f"| {effect_name} | {effect_data.get('weight', 0.5)} | {effect_data.get('direction', 'up')} | {valid} |\n")
                    f.write("\n")
                if pack_result.get('error'):
                    f.write(f"**Error**: {pack_result['error']}\n\n")
        
        print(f"[OK] Summary saved to: {md_file}")
    
    def _print_summary(self):
        """Print validation summary"""
        print("\n" + "=" * 70)
        print("📊 Validation Summary")
        print("=" * 70)
        
        summary = self.results["summary"]
        print(f"\nTotal Packs: {summary['total_packs']}")
        print(f"✅ Loaded Successfully: {summary['loaded_successfully']}")
        print(f"❌ Failed to Load: {summary['failed_to_load']}")
        print(f"📈 Success Rate: {summary.get('success_rate', 0):.1f}%")
        print(f"🔧 Total Effects: {summary['total_effects']}")
        print(f"📦 Unique Effect Types: {len(summary['effect_types_used'])}")
        
        print("\nEffect Types Used:")
        for effect_type in sorted(summary['effect_types_used']):
            print(f"  • {effect_type}")
        
        print("\nPack Status:")
        for pack_name, pack_result in self.results['packs'].items():
            status_icon = "✅" if pack_result['status'] == "success" else "❌"
            print(f"  {status_icon} {pack_name}: {pack_result['status']}")
            if pack_result.get('pack_info'):
                print(f"     Effects: {len(pack_result['pack_info']['effects'])}")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate neuromodulation packs")
    parser.add_argument(
        "--config",
        type=str,
        default="packs/config.json",
        help="Path to pack config file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/validation/packs",
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    validator = PackValidator(output_dir=args.output_dir)
    validator.validate_all_packs(config_path=args.config)


if __name__ == "__main__":
    main()

