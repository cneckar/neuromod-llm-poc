#!/usr/bin/env python3
"""
Randomization and Blinding System for Neuromodulation Study

This module implements Latin square randomization and blinding procedures
to ensure proper experimental design and prevent bias.
"""

import hashlib
import json
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import uuid
from datetime import datetime

@dataclass
class Condition:
    """Represents a single experimental condition"""
    name: str
    pack_name: str
    blind_code: str
    description: str

@dataclass
class LatinSquare:
    """Latin square design for randomization"""
    size: int
    square: List[List[int]]
    conditions: List[str]
    seed: int

class RandomizationSystem:
    """Handles randomization and blinding for the neuromodulation study"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.runs_dir = self.project_root / "runs"
        self.runs_dir.mkdir(exist_ok=True)
        
    def generate_latin_square(self, n_conditions: int, seed: int = None) -> LatinSquare:
        """
        Generate a Latin square for n conditions
        
        Args:
            n_conditions: Number of conditions (must be prime or power of prime)
            seed: Random seed for reproducibility
            
        Returns:
            LatinSquare object with the design
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Generate Latin square using the standard algorithm
        square = []
        for i in range(n_conditions):
            row = [(i + j) % n_conditions for j in range(n_conditions)]
            square.append(row)
        
        # Randomize rows and columns
        row_order = list(range(n_conditions))
        col_order = list(range(n_conditions))
        random.shuffle(row_order)
        random.shuffle(col_order)
        
        # Apply randomizations
        randomized_square = []
        for i in row_order:
            row = [square[i][j] for j in col_order]
            randomized_square.append(row)
        
        conditions = [f"condition_{i}" for i in range(n_conditions)]
        
        return LatinSquare(
            size=n_conditions,
            square=randomized_square,
            conditions=conditions,
            seed=seed
        )
    
    def create_blind_codes(self, pack_names: List[str], global_seed: int) -> Dict[str, str]:
        """
        Create opaque blind codes for pack names
        
        Args:
            pack_names: List of pack names to blind
            global_seed: Global seed for reproducibility
            
        Returns:
            Dictionary mapping pack names to blind codes
        """
        blind_codes = {}
        
        for pack_name in pack_names:
            # Create deterministic hash from pack name + global seed
            hash_input = f"{pack_name}_{global_seed}".encode('utf-8')
            hash_digest = hashlib.sha256(hash_input).hexdigest()
            # Take first 8 characters as blind code
            blind_code = hash_digest[:8]
            blind_codes[pack_name] = blind_code
        
        return blind_codes
    
    def create_condition_mapping(self, pack_names: List[str], global_seed: int) -> Dict[str, Condition]:
        """
        Create condition mapping with blind codes
        
        Args:
            pack_names: List of pack names
            global_seed: Global seed for reproducibility
            
        Returns:
            Dictionary mapping condition names to Condition objects
        """
        blind_codes = self.create_blind_codes(pack_names, global_seed)
        conditions = {}
        
        for pack_name in pack_names:
            blind_code = blind_codes[pack_name]
            condition = Condition(
                name=f"condition_{blind_code}",
                pack_name=pack_name,
                blind_code=blind_code,
                description=f"Blind condition {blind_code} (pack: {pack_name})"
            )
            conditions[condition.name] = condition
        
        return conditions
    
    def generate_counterbalance(self, 
                              pack_names: List[str], 
                              n_items: int, 
                              global_seed: int = None,
                              seed: int = None) -> Dict:
        """
        Generate counterbalanced assignment for items across conditions
        
        Args:
            pack_names: List of pack names to test
            n_items: Number of items to counterbalance
            global_seed: Global seed for reproducibility
            
        Returns:
            Dictionary with counterbalance information
        """
        # Use seed parameter if provided, otherwise use global_seed
        if seed is not None:
            global_seed = seed
        elif global_seed is None:
            global_seed = random.randint(0, 2**32 - 1)
        
        # Create conditions with blind codes
        conditions = self.create_condition_mapping(pack_names, global_seed)
        
        # Generate Latin square
        n_conditions = len(pack_names)
        latin_square = self.generate_latin_square(n_conditions, global_seed)
        
        # Create item assignments
        item_assignments = []
        condition_list = list(conditions.values())
        
        for item_id in range(n_items):
            # Use Latin square to determine condition order for this item
            row_idx = item_id % n_conditions
            col_idx = item_id // n_conditions
            
            if col_idx < n_conditions:
                condition_idx = latin_square.square[row_idx][col_idx]
                condition = condition_list[condition_idx]
            else:
                # If we have more items than conditions, cycle through
                condition_idx = item_id % n_conditions
                condition = condition_list[condition_idx]
            
            item_assignments.append({
                "item_id": item_id,
                "condition_name": condition.name,
                "pack_name": condition.pack_name,
                "blind_code": condition.blind_code,
                "order": col_idx
            })
        
        # Create counterbalance data
        counterbalance_data = {
            "global_seed": global_seed,
            "n_items": n_items,
            "n_conditions": n_conditions,
            "latin_square": {
                "size": latin_square.size,
                "square": latin_square.square,
                "seed": latin_square.seed
            },
            "conditions": {
                name: {
                    "name": cond.name,
                    "pack_name": cond.pack_name,
                    "blind_code": cond.blind_code,
                    "description": cond.description
                }
                for name, cond in conditions.items()
            },
            "item_assignments": item_assignments,
            "created_at": datetime.now().isoformat()
        }
        
        return counterbalance_data
    
    def save_counterbalance(self, counterbalance_data: Dict, run_id: str) -> Path:
        """
        Save counterbalance data to file
        
        Args:
            counterbalance_data: Counterbalance data to save
            run_id: Unique run identifier
            
        Returns:
            Path to saved file
        """
        run_dir = self.runs_dir / run_id
        run_dir.mkdir(exist_ok=True)
        
        counterbalance_file = run_dir / "counterbalance.json"
        with open(counterbalance_file, 'w') as f:
            json.dump(counterbalance_data, f, indent=2)
        
        return counterbalance_file
    
    def create_unblind_key(self, counterbalance_data: Dict, run_id: str) -> Path:
        """
        Create unblind key file (separate from main data)
        
        Args:
            counterbalance_data: Counterbalance data
            run_id: Unique run identifier
            
        Returns:
            Path to unblind key file
        """
        run_dir = self.runs_dir / run_id
        key_dir = run_dir / "key"
        key_dir.mkdir(exist_ok=True)
        
        # Create unblind key with pack name mappings
        unblind_key = {
            "run_id": run_id,
            "global_seed": counterbalance_data["global_seed"],
            "created_at": datetime.now().isoformat(),
            "blind_to_pack_mapping": {
                cond["blind_code"]: cond["pack_name"]
                for cond in counterbalance_data["conditions"].values()
            },
            "pack_to_blind_mapping": {
                cond["pack_name"]: cond["blind_code"]
                for cond in counterbalance_data["conditions"].values()
            }
        }
        
        unblind_file = key_dir / "unblind.json"
        with open(unblind_file, 'w') as f:
            json.dump(unblind_key, f, indent=2)
        
        return unblind_file
    
    def check_leakage(self, prompts: List[str], pack_names: List[str]) -> Dict[str, List[str]]:
        """
        Check for potential leakage of pack names in prompts
        
        Args:
            prompts: List of prompts to check
            pack_names: List of pack names to check for
            
        Returns:
            Dictionary of leakage findings
        """
        leakage_findings = {}
        
        for pack_name in pack_names:
            findings = []
            pack_lower = pack_name.lower()
            
            for i, prompt in enumerate(prompts):
                prompt_lower = prompt.lower()
                
                # Check for direct pack name mentions
                if pack_lower in prompt_lower:
                    findings.append(f"Direct mention in prompt {i}: '{pack_name}'")
                
                # Check for related terms that might hint at the pack
                related_terms = self._get_related_terms(pack_name)
                for term in related_terms:
                    if term.lower() in prompt_lower:
                        findings.append(f"Related term '{term}' in prompt {i}")
            
            if findings:
                leakage_findings[pack_name] = findings
        
        return leakage_findings
    
    def _get_related_terms(self, pack_name: str) -> List[str]:
        """Get terms related to a pack name that might cause leakage"""
        related_terms = {
            "caffeine": ["coffee", "energy", "stimulant", "alert"],
            "cocaine": ["coke", "stimulant", "powder", "white"],
            "lsd": ["acid", "psychedelic", "hallucinogen", "trip"],
            "psilocybin": ["mushrooms", "psychedelic", "magic", "shrooms"],
            "dmt": ["dimethyltryptamine", "psychedelic", "spirit", "breakthrough"],
            "alcohol": ["drink", "beer", "wine", "drunk", "intoxicated"],
            "heroin": ["opiate", "opioid", "narcotic", "dope"],
            "mdma": ["ecstasy", "molly", "empathogen", "love drug"],
            "cannabis": ["marijuana", "weed", "pot", "thc", "cannabis"],
            "ketamine": ["k", "special k", "dissociative", "anesthetic"]
        }
        
        return related_terms.get(pack_name.lower(), [])
    
    def create_run_ledger(self, 
                         run_id: str,
                         counterbalance_data: Dict,
                         model_name: str,
                         backend_kind: str,
                         git_sha: str = None) -> Path:
        """
        Create run ledger with full provenance
        
        Args:
            run_id: Unique run identifier
            counterbalance_data: Counterbalance data
            model_name: Model name used
            backend_kind: Backend type (huggingface, openai, etc.)
            git_sha: Git commit SHA
            
        Returns:
            Path to run ledger file
        """
        run_dir = self.runs_dir / run_id
        run_dir.mkdir(exist_ok=True)
        
        # Get git SHA if not provided
        if git_sha is None:
            try:
                import subprocess
                git_sha = subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], 
                    cwd=self.project_root
                ).decode().strip()
            except:
                git_sha = "unknown"
        
        # Create run ledger
        run_ledger = {
            "run_id": run_id,
            "created_at": datetime.now().isoformat(),
            "git_sha": git_sha,
            "model_name": model_name,
            "backend_kind": backend_kind,
            "global_seed": counterbalance_data["global_seed"],
            "n_items": counterbalance_data["n_items"],
            "n_conditions": counterbalance_data["n_conditions"],
            "conditions_tested": list(counterbalance_data["conditions"].keys()),
            "latin_square_seed": counterbalance_data["latin_square"]["seed"],
            "provenance": {
                "project_root": str(self.project_root),
                "counterbalance_file": f"outputs/experiments/runs/{run_id}/counterbalance.json",
                "unblind_key_file": f"outputs/experiments/runs/{run_id}/key/unblind.json",
                "run_ledger_file": f"outputs/experiments/runs/{run_id}/run.json"
            }
        }
        
        ledger_file = run_dir / "run.json"
        with open(ledger_file, 'w') as f:
            json.dump(run_ledger, f, indent=2)
        
        return ledger_file

def main():
    """Example usage of the randomization system"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate randomization and blinding for neuromodulation study")
    parser.add_argument("--packs", nargs="+", 
                       default=["caffeine", "lsd", "alcohol", "placebo"],
                       help="Pack names to test")
    parser.add_argument("--items", type=int, default=80,
                       help="Number of items to counterbalance")
    parser.add_argument("--seed", type=int, default=42,
                       help="Global seed for reproducibility")
    parser.add_argument("--run-id", default=None,
                       help="Run ID (auto-generated if not provided)")
    
    args = parser.parse_args()
    
    # Generate run ID if not provided
    if args.run_id is None:
        args.run_id = f"run_{uuid.uuid4().hex[:8]}"
    
    # Create randomization system
    rand_system = RandomizationSystem()
    
    # Generate counterbalance
    print(f"Generating counterbalance for {len(args.packs)} packs, {args.items} items...")
    counterbalance_data = rand_system.generate_counterbalance(
        args.packs, args.items, args.seed
    )
    
    # Save files
    counterbalance_file = rand_system.save_counterbalance(counterbalance_data, args.run_id)
    unblind_file = rand_system.create_unblind_key(counterbalance_data, args.run_id)
    ledger_file = rand_system.create_run_ledger(
        args.run_id, counterbalance_data, "test_model", "huggingface"
    )
    
    print(f"✅ Counterbalance saved to: {counterbalance_file}")
    print(f"✅ Unblind key saved to: {unblind_file}")
    print(f"✅ Run ledger saved to: {ledger_file}")
    
    # Show sample assignments
    print(f"\nSample item assignments:")
    for i, assignment in enumerate(counterbalance_data["item_assignments"][:5]):
        print(f"  Item {assignment['item_id']}: {assignment['blind_code']} (pack: {assignment['pack_name']})")
    
    print(f"\nBlind codes:")
    for cond in counterbalance_data["conditions"].values():
        print(f"  {cond['pack_name']} → {cond['blind_code']}")

if __name__ == "__main__":
    main()
