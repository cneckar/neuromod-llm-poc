#!/usr/bin/env python3
"""
Neuromod-LLM Command Line Interface

This module provides the main CLI entry point for the neuromod-llm library.
"""

import click
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from neuromod.pack_system import PackRegistry
from neuromod.optimization.cli import optimization_cli
from neuromod.testing.test_runner import main as test_runner_main

@click.group()
@click.version_option(version="0.2.0")
def cli():
    """Neuromod-LLM: Psychoactive substance analogues for Large Language Models
    
    A research framework for applying neuromodulatory effects to large language models,
    enabling systematic investigation of how different "drug-like" interventions affect
    AI behavior and cognition.
    """
    pass

@cli.command()
def list_packs():
    """List all available neuromodulation packs"""
    try:
        registry = PackRegistry()
        packs = registry.list_packs()
        
        click.echo(f"Available packs ({len(packs)}):")
        click.echo("=" * 50)
        
        for pack_name in sorted(packs):
            pack = registry.get_pack(pack_name)
            click.echo(f"• {pack_name}")
            click.echo(f"  Description: {pack.description}")
            click.echo(f"  Effects: {len(pack.effects)}")
            click.echo()
            
    except Exception as e:
        click.echo(f"Error loading packs: {e}", err=True)
        sys.exit(1)

@cli.command()
@click.argument('pack_name')
def describe_pack(pack_name):
    """Describe a specific neuromodulation pack"""
    try:
        registry = PackRegistry()
        pack = registry.get_pack(pack_name)
        
        click.echo(f"Pack: {pack.name}")
        click.echo(f"Description: {pack.description}")
        click.echo(f"Effects: {len(pack.effects)}")
        click.echo()
        
        click.echo("Effects:")
        for i, effect in enumerate(pack.effects, 1):
            click.echo(f"  {i}. {effect.effect}")
            click.echo(f"     Weight: {effect.weight}, Direction: {effect.direction}")
            if effect.parameters:
                click.echo(f"     Parameters: {effect.parameters}")
            click.echo()
            
    except Exception as e:
        click.echo(f"Error loading pack '{pack_name}': {e}", err=True)
        sys.exit(1)

@cli.command()
@click.argument('pack_name')
@click.option('--prompts', '-p', multiple=True, help='Test prompts')
@click.option('--model', '-m', default='microsoft/DialoGPT-small', help='Model to use')
@click.option('--max-length', default=50, help='Maximum generation length')
def test_pack(pack_name, prompts, model, max_length):
    """Test a neuromodulation pack with given prompts"""
    if not prompts:
        prompts = ["Hello, how are you?", "Tell me about your thoughts"]
    
    try:
        from neuromod.model_support import ModelSupportManager
        from neuromod.pack_system import PackManager
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        
        # Load model
        click.echo(f"Loading model: {model}")
        tokenizer = AutoTokenizer.from_pretrained(model)
        model_obj = AutoModelForCausalLM.from_pretrained(model)
        
        # Load pack
        registry = PackRegistry()
        pack = registry.get_pack(pack_name)
        
        # Apply pack
        pack_manager = PackManager()
        pack_manager.apply_pack(model_obj, pack)
        
        click.echo(f"Testing pack: {pack.name}")
        click.echo("=" * 50)
        
        # Generate responses
        for i, prompt in enumerate(prompts, 1):
            click.echo(f"Prompt {i}: {prompt}")
            
            inputs = tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                outputs = model_obj.generate(
                    inputs.input_ids,
                    max_length=max_length,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            click.echo(f"Response: {response}")
            click.echo()
            
    except Exception as e:
        click.echo(f"Error testing pack: {e}", err=True)
        sys.exit(1)

@cli.command()
def test():
    """Run the test suite"""
    test_runner_main()

# Add optimization subcommands
cli.add_command(optimization_cli, name='optimize')

def main():
    """Main entry point"""
    cli()

if __name__ == '__main__':
    main()
