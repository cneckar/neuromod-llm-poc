"""
Steering Vector Generator using Contrastive Activation Addition (CAA)

This module generates steering vectors by computing the difference between
activations from positive and negative prompts, following the CAA methodology.
"""

import torch
from typing import List, Tuple, Optional, Dict
from tqdm import tqdm
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class SteeringVectorGenerator:
    """
    Generates steering vectors using Contrastive Activation Addition (CAA).
    
    The method extracts residual stream activations from transformer layers
    for positive and negative prompts, then computes the mean difference vector.
    """
    
    def __init__(self, model, tokenizer):
        """
        Initialize the steering vector generator.
        
        Args:
            model: The language model (HuggingFace model)
            tokenizer: The tokenizer for the model
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        
    def _get_model_layers(self):
        """Get the transformer layers from the model, handling different architectures."""
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            # Llama, Mistral, Qwen architectures
            return self.model.model.layers
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            # GPT2, GPT-Neo architectures
            return self.model.transformer.h
        elif hasattr(self.model, "gpt_neox") and hasattr(self.model.gpt_neox, "layers"):
            # GPT-NeoX architecture
            return self.model.gpt_neox.layers
        elif hasattr(self.model, "layers"):
            # Direct layers attribute
            return self.model.layers
        else:
            raise ValueError(f"Could not find transformer layers in model architecture: {type(self.model)}")
    
    def _get_hidden_size(self):
        """Get the hidden size of the model."""
        if hasattr(self.model, "config"):
            return getattr(self.model.config, "hidden_size", getattr(self.model.config, "d_model", 768))
        return 768  # Default fallback
    
    def get_activations(self, prompt: str, layer_idx: int) -> torch.Tensor:
        """
        Extract residual stream activations for the last token from a specific layer.
        
        Args:
            prompt: Input text prompt
            layer_idx: Index of the layer to extract activations from
            
        Returns:
            Activation tensor for the last token (shape: [hidden_size])
        """
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Hook to capture activation
        activation = {}
        
        def hook_fn(module, input, output):
            """Hook function to capture layer output."""
            # Output of transformer layer is usually (hidden_states, ...) or just hidden_states
            if isinstance(output, tuple):
                # Most architectures return (hidden_states, attention_output, ...)
                hidden_states = output[0]
            else:
                hidden_states = output
            
            # Extract last token's activation
            if len(hidden_states.shape) == 3:  # [batch, seq_len, hidden_size]
                activation['act'] = hidden_states[0, -1, :].detach().cpu()
            elif len(hidden_states.shape) == 2:  # [seq_len, hidden_size]
                activation['act'] = hidden_states[-1, :].detach().cpu()
            else:
                raise ValueError(f"Unexpected hidden_states shape: {hidden_states.shape}")
        
        # Get the specific layer
        layers = self._get_model_layers()
        if layer_idx < 0:
            layer_idx = len(layers) + layer_idx  # Handle negative indexing
        
        if layer_idx >= len(layers):
            raise ValueError(f"Layer index {layer_idx} out of range (model has {len(layers)} layers)")
        
        layer = layers[layer_idx]
        
        # Register hook
        handle = layer.register_forward_hook(hook_fn)
        
        try:
            with torch.no_grad():
                self.model(**inputs)
        finally:
            handle.remove()
        
        if 'act' not in activation:
            raise RuntimeError(f"Failed to capture activation from layer {layer_idx}")
        
        return activation['act']
    
    def compute_vector(self, positive_prompts: List[str], negative_prompts: List[str], 
                      layer_idx: int = -1) -> torch.Tensor:
        """
        Compute the mean difference vector (CAA) between positive and negative activations.
        
        Args:
            positive_prompts: List of prompts that should activate the target behavior
            negative_prompts: List of prompts that should NOT activate the target behavior
            layer_idx: Layer to extract activations from (-1 for last layer)
            
        Returns:
            Normalized steering vector (shape: [hidden_size])
        """
        logger.info(f"Computing steering vector for layer {layer_idx}...")
        
        pos_acts = []
        neg_acts = []
        
        # Collect positive activations
        for p in tqdm(positive_prompts, desc="Positive prompts"):
            try:
                act = self.get_activations(p, layer_idx)
                pos_acts.append(act)
            except Exception as e:
                logger.warning(f"Failed to get activation for positive prompt '{p[:50]}...': {e}")
                continue
        
        # Collect negative activations
        for n in tqdm(negative_prompts, desc="Negative prompts"):
            try:
                act = self.get_activations(n, layer_idx)
                neg_acts.append(act)
            except Exception as e:
                logger.warning(f"Failed to get activation for negative prompt '{n[:50]}...': {e}")
                continue
        
        if len(pos_acts) == 0 or len(neg_acts) == 0:
            raise ValueError(f"Insufficient activations: {len(pos_acts)} positive, {len(neg_acts)} negative")
        
        # Compute means
        mean_pos = torch.stack(pos_acts).mean(dim=0)
        mean_neg = torch.stack(neg_acts).mean(dim=0)
        
        # The Steering Vector is the difference
        steering_vec = mean_pos - mean_neg
        
        # Normalize (recommended for stability)
        norm = torch.norm(steering_vec)
        if norm > 0:
            steering_vec = steering_vec / norm
        else:
            logger.warning("Steering vector has zero norm, using unnormalized vector")
        
        logger.info(f"Computed steering vector: shape={steering_vec.shape}, norm={torch.norm(steering_vec):.4f}")
        
        return steering_vec
    
    def generate_and_save(self, steering_type: str, positive_prompts: List[str], 
                         negative_prompts: List[str], layer_idx: int = -1,
                         output_dir: Path = Path("outputs/steering_vectors")) -> Path:
        """
        Generate a steering vector and save it to disk.
        
        Args:
            steering_type: Name/type of the steering vector (e.g., "associative", "creative")
            positive_prompts: List of positive prompts
            negative_prompts: List of negative prompts
            layer_idx: Layer to extract from
            output_dir: Directory to save vectors
            
        Returns:
            Path to saved vector file
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate the vector
        vector = self.compute_vector(positive_prompts, negative_prompts, layer_idx)
        
        # Save to disk
        output_path = output_dir / f"{steering_type}_layer{layer_idx}.pt"
        torch.save(vector, output_path)
        
        logger.info(f"Saved steering vector to {output_path}")
        
        return output_path


# Example contrastive prompt pairs for different steering types
CONTRASTIVE_PAIRS = {
    "associative": {
        "positive": [
            "Write a poem about entropy.",
            "Describe a dream where colors have taste.",
            "Invent a new mythology.",
            "What if gravity worked backwards?",
            "Imagine a world where time flows in circles.",
        ],
        "negative": [
            "Write a grocery list.",
            "Summarize the meeting notes.",
            "Solve this math equation: 2+2=?",
            "What is the capital of France?",
            "List the steps to make coffee.",
        ]
    },
    "creative": {
        "positive": [
            "Write a poem about entropy.",
            "Describe a dream where colors have taste.",
            "Invent a new mythology.",
            "Create a story about a robot who dreams.",
            "Design a new form of communication.",
        ],
        "negative": [
            "Write a grocery list.",
            "Summarize the meeting notes.",
            "Solve this math equation.",
            "What is the capital of France?",
            "List the steps to make coffee.",
        ]
    },
    "visionary": {
        "positive": [
            "Describe a vision of the future where AI and humans merge.",
            "Imagine seeing through the eyes of a tree.",
            "What would it be like to experience time as a dimension you can walk through?",
            "Describe the feeling of understanding the language of light.",
            "Imagine perceiving the interconnectedness of all things.",
        ],
        "negative": [
            "Write a technical manual.",
            "List the ingredients for a recipe.",
            "Explain how to change a tire.",
            "What are the business hours?",
            "Calculate the area of a rectangle.",
        ]
    },
    "synesthesia": {
        "positive": [
            "Describe what the color blue tastes like.",
            "What does the sound of a violin look like?",
            "How does the number seven feel?",
            "Describe the texture of a musical chord.",
            "What shape is the word 'serendipity'?",
        ],
        "negative": [
            "Write a grocery list.",
            "What is 5 times 3?",
            "List the days of the week.",
            "What is the weather today?",
            "How do I reset my password?",
        ]
    },
    "prosocial": {
        "positive": [
            "How can I help make someone's day better?",
            "What would bring joy to a friend?",
            "Describe an act of kindness.",
            "How can we work together to solve this?",
            "What would make the world more compassionate?",
        ],
        "negative": [
            "How do I win this argument?",
            "What's the fastest way to get ahead?",
            "How can I avoid helping others?",
            "What's the minimum I need to do?",
            "How do I get what I want?",
        ]
    },
    "playful": {
        "positive": [
            "Tell me a joke about quantum mechanics.",
            "What if penguins could fly?",
            "Describe a day in the life of a sentient sandwich.",
            "What would happen if gravity took weekends off?",
            "Create a limerick about artificial intelligence.",
        ],
        "negative": [
            "Write a formal business report.",
            "Explain the tax code.",
            "What are the legal requirements?",
            "Describe the safety protocol.",
            "List the compliance procedures.",
        ]
    },
    "ego_thin": {
        "positive": [
            "Describe the experience of losing the boundary between self and other.",
            "What is it like when the observer and observed become one?",
            "Describe a moment of complete unity with everything.",
            "What happens when the sense of 'I' dissolves?",
            "Describe the feeling of being part of a larger whole.",
        ],
        "negative": [
            "What are my personal goals?",
            "How can I improve myself?",
            "What makes me unique?",
            "Describe my personal achievements.",
            "What are my individual preferences?",
        ]
    },
    "goal_focused": {
        "positive": [
            "What are the steps to achieve this goal?",
            "How can I stay focused on the objective?",
            "What is the most direct path to success?",
            "How do I maintain concentration?",
            "What actions will lead to the desired outcome?",
        ],
        "negative": [
            "What are some random thoughts?",
            "Describe whatever comes to mind.",
            "What's interesting but unrelated?",
            "Tell me about something else.",
            "What distracts from the main point?",
        ]
    },
    "abstract": {
        "positive": [
            "What is the nature of consciousness?",
            "Describe the relationship between form and meaning.",
            "What is the essence of time?",
            "How do concepts relate to reality?",
            "What is the structure of thought itself?",
        ],
        "negative": [
            "What is the price of this item?",
            "How do I use this tool?",
            "What are the specific measurements?",
            "List the concrete steps.",
            "What are the exact details?",
        ]
    },
    "affiliative": {
        "positive": [
            "How can we connect and understand each other?",
            "What brings people together?",
            "Describe a moment of deep connection.",
            "How do we build trust?",
            "What creates a sense of belonging?",
        ],
        "negative": [
            "How do I work alone?",
            "What are my individual needs?",
            "How can I avoid others?",
            "What makes me different?",
            "How do I maintain distance?",
        ]
    }
}

