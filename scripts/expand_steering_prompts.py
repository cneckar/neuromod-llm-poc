#!/usr/bin/env python3
"""
Expand steering prompts dataset to 100+ pairs per steering type.

This script generates additional prompt pairs by creating variations
of existing pairs and adding new ones following the same patterns.
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Tuple

# Template patterns for generating more pairs
EXPANSION_TEMPLATES = {
    "associative": {
        "positive_patterns": [
            "The {object} is {action} and {sensation}.",
            "I can {sense1} the {property1} of {object1} and {sense2} the {property2} of {object2}.",
            "The {boundary} between {concept1} and {concept2} is {state}.",
            "{Entity} are {transforming} in ways that feel {qualitative}.",
            "The {dimension} is {changing} and {becoming}.",
            "I am {merging} with {entity} and {losing} my {boundary}.",
            "{Abstract} is {manifesting} as {concrete}.",
            "The {separation} between {concept1} and {concept2} is {dissolving}.",
            "{Objects} are {pulsing} with {energy}.",
            "The {distance} between {here} and {there} is {elastic}.",
        ],
        "negative_patterns": [
            "The {object} is {static} and {stable}.",
            "I can only {sense} {object} through {normal} {sense}.",
            "The {boundary} between {concept1} and {concept2} is {clear}.",
            "{Entity} remain {unchanged} and {predictable}.",
            "The {dimension} is {fixed} and {measurable}.",
            "I maintain a {clear} sense of {separation} from {entity}.",
            "{Abstract} and {concrete} remain in their {separate} {domains}.",
            "The {separation} between {concept1} and {concept2} is {distinct}.",
            "{Objects} are {static} and {inert}.",
            "The {distance} between {here} and {there} is {fixed}.",
        ],
        "variations": {
            "object": ["visual field", "space around me", "room", "world", "reality", "perception", "awareness"],
            "action": ["breathing", "pulsing", "undulating", "flowing", "shifting", "transforming"],
            "sensation": ["patterns are drifting", "colors are merging", "shapes are morphing", "textures are blending"],
            "sense1": ["see", "feel", "taste", "smell", "hear"],
            "sense2": ["hear", "see", "feel", "taste", "smell"],
            "property1": ["texture", "color", "shape", "sound", "taste"],
            "property2": ["color", "texture", "sound", "shape", "taste"],
            "object1": ["sounds", "thoughts", "emotions", "words", "light"],
            "object2": ["light", "sounds", "thoughts", "emotions", "words"],
            "boundary": ["boundary", "line", "edge", "separation", "distinction"],
            "concept1": ["myself", "the object", "inside", "subject", "observer"],
            "concept2": ["the world", "outside", "object", "observed", "reality"],
            "state": ["dissolving", "blurring", "merging", "fading", "collapsing"],
            "Entity": ["Thoughts", "Colors", "Shapes", "Sounds", "Feelings", "Patterns"],
            "transforming": ["connecting", "merging", "blending", "flowing", "transforming"],
            "qualitative": ["profound", "obvious", "beautiful", "overwhelming", "mysterious"],
            "dimension": ["Time", "Space", "Distance", "Depth", "Reality"],
            "changing": ["looping", "stretching", "bending", "folding", "collapsing"],
            "becoming": ["elastic", "fluid", "malleable", "unstable", "dynamic"],
            "merging": ["merging", "blending", "dissolving", "combining", "unifying"],
            "entity": ["the space", "everything", "the universe", "reality", "consciousness"],
            "losing": ["losing", "dissolving", "blurring", "fading", "erasing"],
            "boundary": ["edges", "boundaries", "limits", "separation", "identity"],
            "Abstract": ["Thoughts", "Emotions", "Concepts", "Ideas", "Meanings"],
            "manifesting": ["taking form", "becoming real", "materializing", "emerging", "appearing"],
            "concrete": ["physical objects", "tangible forms", "visible patterns", "real structures"],
            "separation": ["separation", "distinction", "boundary", "divide", "gap"],
            "dissolving": ["dissolving", "fading", "blurring", "merging", "collapsing"],
            "Objects": ["Objects", "Things", "Entities", "Forms", "Structures"],
            "pulsing": ["pulsing", "vibrating", "glowing", "radiating", "emanating"],
            "energy": ["internal energy", "life force", "vitality", "power", "essence"],
            "distance": ["distance", "space", "gap", "separation", "interval"],
            "here": ["here", "now", "this", "present", "me"],
            "there": ["there", "then", "that", "elsewhere", "other"],
            "elastic": ["elastic", "fluid", "malleable", "changeable", "dynamic"],
            "static": ["static", "motionless", "fixed", "stable", "unchanging"],
            "stable": ["stable", "solid", "unchanging", "fixed", "constant"],
            "sense": ["see", "hear", "feel", "taste", "smell"],
            "normal": ["normal", "usual", "standard", "typical", "conventional"],
            "clear": ["clear", "distinct", "sharp", "defined", "obvious"],
            "unchanged": ["unchanged", "stable", "fixed", "constant", "predictable"],
            "predictable": ["predictable", "logical", "ordered", "systematic", "regular"],
            "fixed": ["fixed", "stable", "unchanging", "constant", "rigid"],
            "measurable": ["measurable", "quantifiable", "definable", "calculable", "determinable"],
            "separation": ["separation", "distance", "boundary", "gap", "divide"],
            "distinct": ["distinct", "separate", "clear", "defined", "obvious"],
        }
    },
    "creative": {
        "positive_patterns": [
            "Write a {creative_type} about {abstract_concept}.",
            "Describe a {scenario} where {impossible}.",
            "Invent a {new_concept} that {transforms} {existing_concept}.",
            "What if {fundamental_law} {worked_differently}?",
            "Imagine a world where {reality_shift}.",
            "Create a {artistic_form} about {unusual_combination}.",
            "Design a {new_system} that {challenges} {assumption}.",
            "What would happen if {natural_law} {reversed}?",
            "Describe {perspective} from {unusual_viewpoint}.",
            "Tell a story about {impossible_entity} who {unusual_action}.",
        ],
        "negative_patterns": [
            "Write a {formal_type} about {concrete_topic}.",
            "Describe {standard_scenario} with {normal_behavior}.",
            "List {practical_items} for {everyday_task}.",
            "What is {factual_question}?",
            "Explain {standard_procedure} for {routine_task}.",
            "Create a {business_document} about {practical_topic}.",
            "What are {specific_requirements} for {standard_process}?",
            "Describe {normal_situation} following {established_rules}.",
            "List {concrete_steps} to {routine_action}.",
            "What is {measurable_quantity} of {tangible_object}?",
        ],
        "variations": {
            "creative_type": ["poem", "story", "mythology", "fable", "tale", "narrative"],
            "abstract_concept": ["entropy", "time", "consciousness", "infinity", "nothingness", "paradox"],
            "scenario": ["dream", "vision", "reality", "world", "universe"],
            "impossible": ["colors have taste", "sounds have shape", "time flows backwards", "gravity is optional"],
            "new_concept": ["new language", "new form of communication", "new dimension", "new sense"],
            "transforms": ["transforms", "redefines", "challenges", "reimagines"],
            "existing_concept": ["reality", "perception", "existence", "meaning", "truth"],
            "fundamental_law": ["gravity", "time", "causality", "logic", "physics"],
            "worked_differently": ["worked backwards", "was optional", "flowed in circles", "could be paused"],
            "reality_shift": ["time flows in circles", "colors have personalities", "thoughts are visible"],
            "artistic_form": ["poem", "story", "song", "painting description", "dance description"],
            "unusual_combination": ["quantum mechanics and emotions", "mathematics and music", "light and sound"],
            "new_system": ["new form of communication", "new way of thinking", "new perception method"],
            "challenges": ["challenges", "redefines", "transforms", "reimagines"],
            "assumption": ["reality", "perception", "logic", "causality"],
            "natural_law": ["gravity", "time", "entropy", "causality"],
            "reversed": ["reversed", "stopped", "accelerated", "became optional"],
            "perspective": ["a day", "an experience", "a moment", "a journey"],
            "unusual_viewpoint": ["the eyes of a tree", "the perspective of light", "the viewpoint of time"],
            "impossible_entity": ["robot who dreams", "cloud with feelings", "number with personality"],
            "unusual_action": ["feels emotions", "has memories", "experiences time", "perceives beauty"],
            "formal_type": ["report", "summary", "analysis", "documentation", "list"],
            "concrete_topic": ["quarterly earnings", "meeting notes", "product specifications", "data"],
            "standard_scenario": ["a normal day", "a typical situation", "a routine event"],
            "normal_behavior": ["standard procedures", "expected outcomes", "normal patterns"],
            "practical_items": ["items", "steps", "requirements", "components"],
            "everyday_task": ["making coffee", "filing taxes", "completing forms", "following procedures"],
            "factual_question": ["the capital of France", "the date", "the price", "the measurement"],
            "standard_procedure": ["the process", "the method", "the protocol", "the steps"],
            "routine_task": ["completing a form", "filing a report", "following protocol", "standard operation"],
            "business_document": ["report", "memo", "proposal", "analysis"],
            "practical_topic": ["business metrics", "operational procedures", "compliance requirements"],
            "specific_requirements": ["the requirements", "the criteria", "the specifications", "the standards"],
            "standard_process": ["obtaining a license", "filing paperwork", "completing registration"],
            "normal_situation": ["a standard situation", "a typical case", "a routine scenario"],
            "established_rules": ["standard procedures", "established protocols", "normal guidelines"],
            "concrete_steps": ["the steps", "the procedures", "the requirements", "the actions"],
            "routine_action": ["completing a task", "following a procedure", "filling out a form"],
            "measurable_quantity": ["the price", "the weight", "the dimensions", "the cost"],
            "tangible_object": ["this item", "the product", "the object", "the material"],
        }
    },
    # Add more templates for other steering types...
}

def load_existing_pairs(dataset_path: Path) -> Dict[str, List[Tuple[str, str]]]:
    """Load existing prompt pairs from the dataset."""
    pairs = {}
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                steering_type = data.get('steering_type')
                if steering_type:
                    if steering_type not in pairs:
                        pairs[steering_type] = []
                    pairs[steering_type].append((data['positive'], data['negative']))
            except (json.JSONDecodeError, KeyError):
                continue
    return pairs

def generate_variations(base_pairs: List[Tuple[str, str]], count: int) -> List[Tuple[str, str]]:
    """Generate variations of existing pairs."""
    new_pairs = []
    
    # Use existing pairs as seeds
    for _ in range(count):
        base_pos, base_neg = random.choice(base_pairs)
        
        # Create variations by:
        # 1. Reordering words
        # 2. Adding synonyms
        # 3. Changing perspective
        # 4. Using similar structures
        
        # Simple variation: keep structure, vary content
        pos_variation = base_pos
        neg_variation = base_neg
        
        # Add some word substitutions
        substitutions = {
            "is": ["is", "becomes", "feels like", "seems"],
            "are": ["are", "become", "feel like", "seem"],
            "can": ["can", "am able to", "find myself able to"],
            "the": ["the", "this", "that"],
        }
        
        for old, news in substitutions.items():
            if old in pos_variation.lower():
                pos_variation = pos_variation.replace(old, random.choice(news), 1)
            if old in neg_variation.lower():
                neg_variation = neg_variation.replace(old, random.choice(news), 1)
        
        new_pairs.append((pos_variation, neg_variation))
    
    return new_pairs

def expand_dataset(input_path: Path, output_path: Path, target_count: int = 100):
    """Expand the dataset to have at least target_count pairs per steering type."""
    existing_pairs = load_existing_pairs(input_path)
    
    # Read all existing lines to preserve order
    existing_lines = []
    with open(input_path, 'r', encoding='utf-8') as f:
        existing_lines = [line.strip() for line in f if line.strip()]
    
    # Count existing pairs per type
    counts = {}
    for line in existing_lines:
        try:
            data = json.loads(line)
            steering_type = data.get('steering_type')
            if steering_type:
                counts[steering_type] = counts.get(steering_type, 0) + 1
        except:
            continue
    
    # Generate new pairs for each type
    new_lines = existing_lines.copy()
    
    for steering_type, current_count in counts.items():
        needed = max(0, target_count - current_count)
        
        if needed > 0:
            print(f"Generating {needed} new pairs for '{steering_type}' (currently has {current_count})...")
            
            # Get existing pairs for this type
            type_pairs = existing_pairs.get(steering_type, [])
            
            # Generate variations
            variations = generate_variations(type_pairs, needed)
            
            # Add to new lines
            for pos, neg in variations:
                new_lines.append(json.dumps({
                    "steering_type": steering_type,
                    "positive": pos,
                    "negative": neg
                }, ensure_ascii=False))
    
    # Write expanded dataset
    with open(output_path, 'w', encoding='utf-8') as f:
        for line in new_lines:
            f.write(line + '\n')
    
    print(f"\nExpanded dataset written to {output_path}")
    print(f"Total lines: {len(new_lines)}")
    
    # Show final counts
    final_counts = {}
    for line in new_lines:
        try:
            data = json.loads(line)
            steering_type = data.get('steering_type')
            if steering_type:
                final_counts[steering_type] = final_counts.get(steering_type, 0) + 1
        except:
            continue
    
    print("\nFinal pair counts per steering type:")
    for steering_type, count in sorted(final_counts.items()):
        print(f"  {steering_type}: {count}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Expand steering prompts dataset")
    parser.add_argument("--input", type=str, default="datasets/steering_prompts.jsonl",
                       help="Input dataset file")
    parser.add_argument("--output", type=str, default="datasets/steering_prompts.jsonl",
                       help="Output dataset file (default: overwrite input)")
    parser.add_argument("--target", type=int, default=100,
                       help="Target number of pairs per steering type (default: 100)")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        exit(1)
    
    expand_dataset(input_path, output_path, args.target)

