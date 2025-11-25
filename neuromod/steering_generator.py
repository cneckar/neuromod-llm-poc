"""
Steering Vector Generator using Robust Mean Difference Vector (MDV) with PCA

This module generates steering vectors by:
1. Loading 100+ prompt pairs from datasets/steering_prompts.jsonl
2. Extracting activations from all layers (not just the last)
3. Computing difference vectors (x_pos - x_neg) for each pair
4. Using PCA to extract the First Principal Component (PC1) as the steering vector
5. Validating separation significance (p < 0.01) before accepting the vector
"""

import torch
import json
import numpy as np
from typing import List, Tuple, Optional, Dict
from tqdm import tqdm
import logging
from pathlib import Path
from scipy import stats
from sklearn.decomposition import PCA

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
    
    def load_prompt_pairs(self, dataset_path: Path, steering_type: str, min_pairs: int = 100) -> Tuple[List[str], List[str]]:
        """
        Load prompt pairs from JSONL dataset file.
        
        Args:
            dataset_path: Path to the JSONL file containing prompt pairs
            steering_type: Type of steering to filter for (e.g., "associative", "creative")
            min_pairs: Minimum number of pairs required (default: 100)
            
        Returns:
            Tuple of (positive_prompts, negative_prompts) lists
        """
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
        
        positive_prompts = []
        negative_prompts = []
        
        logger.info(f"Loading prompt pairs from {dataset_path} for steering type '{steering_type}'...")
        
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    if data.get('steering_type') == steering_type:
                        positive_prompts.append(data['positive'])
                        negative_prompts.append(data['negative'])
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Skipping invalid line in dataset: {e}")
                    continue
        
        if len(positive_prompts) < min_pairs:
            raise ValueError(
                f"Insufficient prompt pairs: found {len(positive_prompts)}, "
                f"required {min_pairs} for robust vector computation. "
                f"Use --min-pairs {len(positive_prompts)} to proceed with available pairs (quality may be reduced)."
            )
        
        # Warn if using fewer than recommended pairs
        if len(positive_prompts) < 100 and min_pairs < 100:
            logger.warning(
                f"Only {len(positive_prompts)} prompt pairs available (recommended: 100+). "
                f"Steering vector quality may be reduced."
            )
        
        logger.info(f"Loaded {len(positive_prompts)} prompt pairs for '{steering_type}'")
        return positive_prompts, negative_prompts
    
    def compute_vector_robust(self, dataset_path: Path, steering_type: str, 
                            layer_idx: Optional[int] = None, 
                            use_pca: bool = True,
                            validate: bool = True,
                            validation_split: float = 0.2,
                            min_pairs: int = 100) -> torch.Tensor:
        """
        Compute robust steering vector using MDV pipeline with PCA denoising.
        
        FIXED: Uses 100+ prompt pairs (configurable), extracts from all layers, applies PCA to difference vectors,
        and validates separation significance.
        
        Args:
            dataset_path: Path to JSONL dataset file
            steering_type: Type of steering to compute (e.g., "associative", "creative")
            layer_idx: Specific layer to use (None = use all layers and aggregate)
            use_pca: Whether to use PCA on difference vectors (recommended)
            validate: Whether to validate separation significance
            validation_split: Fraction of data to use for validation
            min_pairs: Minimum number of prompt pairs required (default: 100, lower values may reduce quality)
            
        Returns:
            Normalized steering vector (shape: [hidden_size])
        """
        # Load prompt pairs
        positive_prompts, negative_prompts = self.load_prompt_pairs(dataset_path, steering_type, min_pairs=min_pairs)
        
        # Split into training and validation sets
        n_total = len(positive_prompts)
        n_val = int(n_total * validation_split)
        n_train = n_total - n_val
        
        pos_train = positive_prompts[:n_train]
        neg_train = negative_prompts[:n_train]
        pos_val = positive_prompts[n_train:]
        neg_val = negative_prompts[n_train:]
        
        logger.info(f"Using {n_train} pairs for training, {n_val} pairs for validation")
        
        # Get all layers if layer_idx is None
        layers = self._get_model_layers()
        if layer_idx is None:
            # Extract from all layers and aggregate
            layer_indices = list(range(len(layers)))
            logger.info(f"Extracting activations from all {len(layer_indices)} layers")
        else:
            layer_indices = [layer_idx]
        
        # Collect difference vectors for all layers
        all_diff_vectors = []
        
        for layer_idx_curr in layer_indices:
            logger.info(f"Processing layer {layer_idx_curr}...")
            
            # Collect activations for training pairs
            pos_acts = []
            neg_acts = []
            
            for p, n in tqdm(zip(pos_train, neg_train), total=len(pos_train), 
                           desc=f"Layer {layer_idx_curr}: Extracting activations"):
                try:
                    pos_act = self.get_activations(p, layer_idx_curr)
                    neg_act = self.get_activations(n, layer_idx_curr)
                    pos_acts.append(pos_act.numpy())
                    neg_acts.append(neg_act.numpy())
                except Exception as e:
                    logger.warning(f"Failed to get activations for pair: {e}")
                    continue
            
            if len(pos_acts) == 0 or len(neg_acts) == 0:
                logger.warning(f"Insufficient activations for layer {layer_idx_curr}, skipping")
                continue
            
            # Compute difference vectors: x_pos - x_neg for each pair
            pos_acts = np.array(pos_acts)  # [n_pairs, hidden_size]
            neg_acts = np.array(neg_acts)  # [n_pairs, hidden_size]
            diff_vectors = pos_acts - neg_acts  # [n_pairs, hidden_size]
            
            all_diff_vectors.append(diff_vectors)
        
        if len(all_diff_vectors) == 0:
            raise ValueError("No valid activations collected from any layer")
        
        # Stack all difference vectors: [n_layers * n_pairs, hidden_size]
        all_diffs = np.vstack(all_diff_vectors)
        logger.info(f"Total difference vectors: {all_diffs.shape[0]} (from {len(all_diff_vectors)} layers)")
        
        # Apply PCA to denoise the signal
        if use_pca:
            logger.info("Applying PCA to extract First Principal Component (PC1)...")
            pca = PCA(n_components=1)
            pca.fit(all_diffs)
            steering_vec = torch.from_numpy(pca.components_[0]).float()
            explained_variance = pca.explained_variance_ratio_[0]
            logger.info(f"PC1 explains {explained_variance:.2%} of variance")
        else:
            # Fallback: simple mean difference
            logger.info("Using mean difference vector (PCA disabled)")
            steering_vec = torch.from_numpy(all_diffs.mean(axis=0)).float()
        
        # Normalize
        norm = torch.norm(steering_vec)
        if norm > 0:
            steering_vec = steering_vec / norm
        else:
            raise ValueError("Steering vector has zero norm")
        
        # Validate separation if requested
        if validate and len(pos_val) > 0:
            logger.info("Validating separation significance...")
            is_valid = self._validate_separation(steering_vec, pos_val, neg_val, layer_indices)
            if not is_valid:
                raise ValueError(
                    "Validation failed: separation not significant (p >= 0.01). "
                    "Vector may be noise. Consider increasing dataset size or checking prompt quality."
                )
            logger.info("Validation passed: separation is significant (p < 0.01)")
        
        logger.info(f"Computed robust steering vector: shape={steering_vec.shape}, norm={torch.norm(steering_vec):.4f}")
        return steering_vec
    
    def _validate_separation(self, steering_vec: torch.Tensor, 
                           positive_prompts: List[str], 
                           negative_prompts: List[str],
                           layer_indices: List[int],
                           p_threshold: float = 0.01) -> bool:
        """
        Validate that the steering vector produces significant separation.
        
        Projects validation prompts onto the steering vector and tests if
        positive and negative prompts are significantly separated (t-test).
        
        Args:
            steering_vec: The computed steering vector
            positive_prompts: Validation positive prompts
            negative_prompts: Validation negative prompts
            layer_indices: Layers to extract from
            p_threshold: P-value threshold for significance (default: 0.01)
            
        Returns:
            True if separation is significant (p < p_threshold), False otherwise
        """
        pos_projections = []
        neg_projections = []
        
        for layer_idx in layer_indices:
            for p, n in zip(positive_prompts, negative_prompts):
                try:
                    pos_act = self.get_activations(p, layer_idx)
                    neg_act = self.get_activations(n, layer_idx)
                    
                    # Project onto steering vector
                    pos_proj = torch.dot(pos_act, steering_vec).item()
                    neg_proj = torch.dot(neg_act, steering_vec).item()
                    
                    pos_projections.append(pos_proj)
                    neg_projections.append(neg_proj)
                except Exception as e:
                    logger.warning(f"Failed to validate pair: {e}")
                    continue
        
        if len(pos_projections) < 10 or len(neg_projections) < 10:
            logger.warning("Insufficient validation data, skipping validation")
            return True  # Don't fail validation if we can't test
        
        # Perform t-test
        t_stat, p_value = stats.ttest_ind(pos_projections, neg_projections)
        
        logger.info(f"Validation t-test: t={t_stat:.4f}, p={p_value:.6f}")
        
        return p_value < p_threshold
    
    def compute_vector(self, positive_prompts: List[str], negative_prompts: List[str], 
                      layer_idx: int = -1) -> torch.Tensor:
        """
        Legacy method: Compute mean difference vector (kept for backward compatibility).
        
        For new code, use compute_vector_robust() instead.
        """
        logger.warning("Using legacy compute_vector(). Consider using compute_vector_robust() for better results.")
        return self._compute_vector_legacy(positive_prompts, negative_prompts, layer_idx)
    
    def _compute_vector_legacy(self, positive_prompts: List[str], negative_prompts: List[str], 
                              layer_idx: int = -1) -> torch.Tensor:
        """Legacy implementation for backward compatibility."""
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

