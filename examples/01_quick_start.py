#!/usr/bin/env python3
"""
Neuromod-LLM Quick Start Guide

This script demonstrates the basic usage of the Neuromod-LLM library for applying
psychoactive substance analogues to large language models.

What You'll Learn:
- How to load and apply neuromodulation packs
- How to generate text with different "drug-like" effects
- How to compare different packs
- How to create custom effects

Prerequisites:
Make sure you have installed the library:
pip install neuromod-llm
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from neuromod import PackRegistry, PackManager, ModelSupportManager
from neuromod.optimization import PackOptimizer, OptimizationConfig, OptimizationMethod
from neuromod.optimization.targets import BehavioralTarget, TargetSpec, TargetType, OptimizationObjective

def main():
    print("🧠 Neuromod-LLM Quick Start Guide")
    print("=" * 50)
    
    # 1. Basic Setup
    print("\n1. Basic Setup")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # 2. Load a Model and Pack
    print("\n2. Loading Model and Packs")
    model_name = "microsoft/DialoGPT-small"
    print(f"Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Load available packs
    registry = PackRegistry()
    available_packs = registry.list_packs()
    print(f"Available packs: {len(available_packs)}")
    print(f"Sample packs: {available_packs[:5]}")
    
    # 3. Apply a Pack and Generate Text
    print("\n3. Applying Pack and Generating Text")
    pack = registry.get_pack("caffeine")
    print(f"Pack: {pack.name}")
    print(f"Description: {pack.description}")
    print(f"Effects: {len(pack.effects)}")
    
    # Create pack manager
    pack_manager = PackManager()
    
    # Apply pack to model
    pack_manager.apply_pack(model, pack)
    print("Pack applied successfully!")
    
    # 4. Generate Text with Different Packs
    print("\n4. Generating Text with Different Packs")
    
    def generate_text(prompt, model, tokenizer, max_length=50):
        """Generate text with the given prompt"""
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Test prompt
    prompt = "I feel so energized and focused today"
    
    # Generate with caffeine pack
    response = generate_text(prompt, model, tokenizer)
    print(f"With Caffeine Pack:")
    print(f"{response}")
    
    # 5. Compare Different Packs
    print("\n5. Comparing Different Packs")
    packs_to_test = ["placebo", "caffeine", "alcohol", "mdma"]
    test_prompt = "Tell me about your thoughts and feelings"
    
    for pack_name in packs_to_test:
        if pack_name in available_packs:
            # Load and apply pack
            pack = registry.get_pack(pack_name)
            pack_manager.apply_pack(model, pack)
            
            # Generate response
            response = generate_text(test_prompt, model, tokenizer)
            print(f"\n=== {pack_name.upper()} PACK ===")
            print(f"{response}")
    
    # 6. Create Custom Behavioral Target
    print("\n6. Creating Custom Behavioral Target")
    creative_target = BehavioralTarget(
        name="creative_writing",
        description="Enhance creativity and originality in writing",
        targets=[
            TargetSpec(
                name="behavior_creativity",
                target_type=TargetType.BEHAVIOR,
                objective=OptimizationObjective.MAXIMIZE,
                target_value=0.8,
                weight=1.0
            ),
            TargetSpec(
                name="metric_originality",
                target_type=TargetType.METRIC,
                objective=OptimizationObjective.MAXIMIZE,
                target_value=0.7,
                weight=0.8
            ),
            TargetSpec(
                name="latent_openness",
                target_type=TargetType.LATENT_AXIS,
                objective=OptimizationObjective.MAXIMIZE,
                target_value=0.9,
                weight=0.6
            )
        ]
    )
    
    print(f"Created target: {creative_target.name}")
    print(f"Number of targets: {len(creative_target.targets)}")
    
    # 7. Basic Pack Optimization (Quick Demo)
    print("\n7. Basic Pack Optimization")
    print("Setting up optimization (quick demo)...")
    
    model_manager = ModelSupportManager(test_mode=True)
    config = OptimizationConfig(
        method=OptimizationMethod.EVOLUTIONARY,
        max_iterations=3,  # Very quick demo
        population_size=4
    )
    
    from neuromod.optimization.evaluation import EvaluationFramework
    evaluation_framework = EvaluationFramework()
    optimizer = PackOptimizer(model_manager, evaluation_framework, config)
    
    # Test prompts
    test_prompts = [
        "Write a creative story about a robot",
        "Describe an imaginary world",
        "Create a poem about the future"
    ]
    
    # Get base pack
    base_pack = registry.get_pack("placebo")
    
    print("Starting optimization...")
    print("This may take a moment...")
    
    # Run optimization
    result = optimizer.optimize_pack(base_pack, creative_target, test_prompts)
    
    print(f"\nOptimization complete!")
    print(f"Final loss: {result.final_loss:.4f}")
    print(f"Success: {result.success}")
    
    # 8. Test Optimized Pack
    print("\n8. Testing Optimized Pack")
    if result.success:
        pack_manager.apply_pack(model, result.optimized_pack)
        
        # Test with creative prompts
        creative_prompts = [
            "Write a creative story about a robot",
            "Describe an imaginary world",
            "Create a poem about the future"
        ]
        
        print("=== OPTIMIZED PACK RESULTS ===")
        for prompt in creative_prompts:
            response = generate_text(prompt, model, tokenizer)
            print(f"Prompt: {prompt}")
            print(f"Response: {response}\n")
    else:
        print("Optimization failed. Using base pack.")
    
    # 9. Next Steps
    print("\n9. Next Steps")
    print("Now that you've completed the quick start guide, you can:")
    print("1. Explore More Examples: Check out the other examples in this directory")
    print("2. Read the Documentation: Visit https://pihk.ai")
    print("3. Try Different Packs: Experiment with the 80+ available packs")
    print("4. Create Custom Targets: Design your own behavioral optimization goals")
    print("5. Run Research Experiments: Use the scientific testing framework")
    
    print("\nUseful Commands:")
    print("# List all available packs")
    print("neuromod list-packs")
    print("\n# Test a specific pack")
    print("neuromod test-pack --pack caffeine --prompts \"Hello world\"")
    print("\n# Run optimization")
    print("neuromod optimize --pack placebo --target creative_writing")
    
    print("\nResources:")
    print("- Documentation: https://pihk.ai")
    print("- GitHub: https://github.com/cneckar/neuromod-llm-poc")
    print("- Issues: https://github.com/cneckar/neuromod-llm-poc/issues")
    print("- Discussions: https://github.com/cneckar/neuromod-llm-poc/discussions")

if __name__ == "__main__":
    main()
