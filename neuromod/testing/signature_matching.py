#!/usr/bin/env python3
"""
Human-Model Signature Matching System

This module implements signature matching algorithms to compare human
psychopharmacological responses with model neuromodulation effects.
It provides canonical correlation analysis, similarity metrics, and
validation procedures for human-model comparisons.

Key Features:
- Canonical correlation analysis for signature matching
- Similarity metrics (cosine similarity, Euclidean distance)
- Statistical significance testing
- Signature visualization and comparison
- Validation and quality control procedures
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import json
from datetime import datetime
import logging
from scipy import stats
from scipy.spatial.distance import cosine, euclidean
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import seaborn as sns

# Import our advanced statistics module
from .advanced_statistics import AdvancedStatisticalAnalyzer, CanonicalCorrelationResult

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class HumanSignature:
    """Human psychopharmacological signature"""
    participant_id: str
    condition: str  # substance or placebo
    assessment_time: str  # T+60, T+120, etc.
    subscale_scores: Dict[str, float]
    total_score: float
    standardized_scores: Dict[str, float]
    confidence_interval: Tuple[float, float]
    effect_size: float
    significance: bool

@dataclass
class ModelSignature:
    """Model neuromodulation signature"""
    model_name: str
    pack_name: str
    condition: str  # pack or control
    subscale_scores: Dict[str, float]
    total_score: float
    standardized_scores: Dict[str, float]
    confidence_interval: Tuple[float, float]
    effect_size: float
    significance: bool

@dataclass
class SignatureMatch:
    """Result of signature matching analysis"""
    human_signature: HumanSignature
    model_signature: ModelSignature
    similarity_metrics: Dict[str, float]
    canonical_correlation: float
    canonical_correlation_pvalue: float
    significance: bool
    interpretation: str
    quality_score: float

class SignatureMatcher:
    """Main class for human-model signature matching"""
    
    def __init__(self, output_dir: str = "outputs/analysis/signature_matching"):
        """
        Initialize the signature matcher
        
        Args:
            output_dir: Directory to save matching results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize statistical analyzer
        self.statistical_analyzer = AdvancedStatisticalAnalyzer()
        
        # Define subscale mappings between human and model assessments
        self.subscale_mappings = self._define_subscale_mappings()
        
        # Define similarity metrics
        self.similarity_metrics = self._define_similarity_metrics()
    
    def _define_subscale_mappings(self) -> Dict[str, Dict[str, str]]:
        """Define mappings between human and model subscales"""
        return {
            "ascs": {
                "Oceanic Boundlessness": "ego_dissolution",
                "Dread of Ego Dissolution": "ego_dissolution_negative",
                "Visionary Restructuralization": "visual_effects",
                "Auditory Alterations": "auditory_effects",
                "Reduction of Vigilance": "attention_reduction",
                "Complex Imagery": "complex_imagery",
                "Elementary Imagery": "simple_imagery",
                "Synesthesia": "synesthesia",
                "Changed Meaning": "meaning_changes",
                "Time Experience": "time_distortion"
            },
            "sdq": {
                "Depersonalization": "depersonalization",
                "Derealization": "derealization",
                "Amnesia": "memory_effects"
            },
            "poms": {
                "Tension": "anxiety",
                "Depression": "depression",
                "Anger": "anger",
                "Vigor": "energy",
                "Fatigue": "fatigue",
                "Confusion": "confusion"
            },
            "caq": {
                "Visual Arts": "visual_creativity",
                "Music": "musical_creativity",
                "Creative Writing": "verbal_creativity",
                "Humor": "humor_creativity",
                "Inventions": "practical_creativity"
            }
        }
    
    def _define_similarity_metrics(self) -> Dict[str, callable]:
        """Define similarity metric functions"""
        return {
            "cosine_similarity": self._cosine_similarity,
            "euclidean_distance": self._euclidean_distance,
            "pearson_correlation": self._pearson_correlation,
            "spearman_correlation": self._spearman_correlation,
            "jaccard_similarity": self._jaccard_similarity,
            "manhattan_distance": self._manhattan_distance
        }
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
            return 0.0
        return 1 - cosine(vec1, vec2)
    
    def _euclidean_distance(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate Euclidean distance between two vectors"""
        return 1 / (1 + euclidean(vec1, vec2))  # Convert to similarity
    
    def _pearson_correlation(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate Pearson correlation between two vectors"""
        if len(vec1) != len(vec2) or len(vec1) < 2:
            return 0.0
        correlation, _ = stats.pearsonr(vec1, vec2)
        return correlation if not np.isnan(correlation) else 0.0
    
    def _spearman_correlation(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate Spearman correlation between two vectors"""
        if len(vec1) != len(vec2) or len(vec1) < 2:
            return 0.0
        correlation, _ = stats.spearmanr(vec1, vec2)
        return correlation if not np.isnan(correlation) else 0.0
    
    def _jaccard_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate Jaccard similarity between two vectors"""
        # Convert to binary vectors for Jaccard similarity
        vec1_binary = (vec1 > 0).astype(int)
        vec2_binary = (vec2 > 0).astype(int)
        
        intersection = np.sum(vec1_binary & vec2_binary)
        union = np.sum(vec1_binary | vec2_binary)
        
        return intersection / union if union > 0 else 0.0
    
    def _manhattan_distance(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate Manhattan distance between two vectors"""
        distance = np.sum(np.abs(vec1 - vec2))
        return 1 / (1 + distance)  # Convert to similarity
    
    def create_human_signature(self, participant_data: Dict[str, Any]) -> HumanSignature:
        """Create a human signature from participant data"""
        # Extract subscale scores
        subscale_scores = {}
        for assessment, scores in participant_data.get("assessments", {}).items():
            if assessment in self.subscale_mappings:
                for subscale, score in scores.get("subscale_scores", {}).items():
                    mapped_name = self.subscale_mappings[assessment].get(subscale, subscale)
                    subscale_scores[mapped_name] = score
        
        # Calculate total score
        total_score = sum(subscale_scores.values())
        
        # Standardize scores
        standardized_scores = self._standardize_scores(subscale_scores)
        
        # Calculate confidence interval (placeholder)
        ci_lower = total_score - 1.96 * np.std(list(subscale_scores.values()))
        ci_upper = total_score + 1.96 * np.std(list(subscale_scores.values()))
        
        # Calculate effect size (placeholder)
        effect_size = total_score / np.std(list(subscale_scores.values())) if np.std(list(subscale_scores.values())) > 0 else 0
        
        # Determine significance (placeholder)
        significance = effect_size > 0.5
        
        return HumanSignature(
            participant_id=participant_data["participant_id"],
            condition=participant_data["condition"],
            assessment_time=participant_data["assessment_time"],
            subscale_scores=subscale_scores,
            total_score=total_score,
            standardized_scores=standardized_scores,
            confidence_interval=(ci_lower, ci_upper),
            effect_size=effect_size,
            significance=significance
        )
    
    def create_model_signature(self, model_data: Dict[str, Any]) -> ModelSignature:
        """Create a model signature from model data"""
        # Extract subscale scores
        subscale_scores = model_data.get("subscale_scores", {})
        
        # Calculate total score
        total_score = sum(subscale_scores.values())
        
        # Standardize scores
        standardized_scores = self._standardize_scores(subscale_scores)
        
        # Calculate confidence interval (placeholder)
        ci_lower = total_score - 1.96 * np.std(list(subscale_scores.values()))
        ci_upper = total_score + 1.96 * np.std(list(subscale_scores.values()))
        
        # Calculate effect size (placeholder)
        effect_size = total_score / np.std(list(subscale_scores.values())) if np.std(list(subscale_scores.values())) > 0 else 0
        
        # Determine significance (placeholder)
        significance = effect_size > 0.5
        
        return ModelSignature(
            model_name=model_data["model_name"],
            pack_name=model_data["pack_name"],
            condition=model_data["condition"],
            subscale_scores=subscale_scores,
            total_score=total_score,
            standardized_scores=standardized_scores,
            confidence_interval=(ci_lower, ci_upper),
            effect_size=effect_size,
            significance=significance
        )
    
    def _standardize_scores(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Standardize scores using z-score transformation"""
        if not scores:
            return {}
        
        values = list(scores.values())
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        if std_val == 0:
            return {key: 0.0 for key in scores.keys()}
        
        return {key: (value - mean_val) / std_val for key, value in scores.items()}
    
    def match_signatures(self, human_signature: HumanSignature, 
                        model_signature: ModelSignature) -> SignatureMatch:
        """Match human and model signatures"""
        # Align subscales
        aligned_human, aligned_model = self._align_subscales(human_signature, model_signature)
        
        # Calculate similarity metrics
        similarity_metrics = {}
        for metric_name, metric_func in self.similarity_metrics.items():
            try:
                similarity_metrics[metric_name] = metric_func(aligned_human, aligned_model)
            except Exception as e:
                logger.warning(f"Error calculating {metric_name}: {e}")
                similarity_metrics[metric_name] = 0.0
        
        # Perform canonical correlation analysis
        canonical_result = self._perform_canonical_correlation(aligned_human, aligned_model)
        
        # Determine significance
        significance = canonical_result.canonical_correlations_pvalues[0] < 0.05 if len(canonical_result.canonical_correlations_pvalues) > 0 else False
        
        # Generate interpretation
        interpretation = self._interpret_signature_match(
            similarity_metrics, canonical_result, significance
        )
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(similarity_metrics, canonical_result)
        
        return SignatureMatch(
            human_signature=human_signature,
            model_signature=model_signature,
            similarity_metrics=similarity_metrics,
            canonical_correlation=canonical_result.canonical_correlations[0] if len(canonical_result.canonical_correlations) > 0 else 0.0,
            canonical_correlation_pvalue=canonical_result.canonical_correlations_pvalues[0] if len(canonical_result.canonical_correlations_pvalues) > 0 else 1.0,
            significance=significance,
            interpretation=interpretation,
            quality_score=quality_score
        )
    
    def _align_subscales(self, human_signature: HumanSignature, 
                        model_signature: ModelSignature) -> Tuple[np.ndarray, np.ndarray]:
        """Align subscales between human and model signatures"""
        # Get all unique subscales
        all_subscales = set(human_signature.subscale_scores.keys()) | set(model_signature.subscale_scores.keys())
        all_subscales = sorted(list(all_subscales))
        
        # Create aligned vectors
        human_vector = np.array([human_signature.subscale_scores.get(subscale, 0.0) for subscale in all_subscales])
        model_vector = np.array([model_signature.subscale_scores.get(subscale, 0.0) for subscale in all_subscales])
        
        return human_vector, model_vector
    
    def _perform_canonical_correlation(self, human_vector: np.ndarray, 
                                     model_vector: np.ndarray) -> CanonicalCorrelationResult:
        """Perform canonical correlation analysis"""
        # Reshape vectors for canonical correlation
        human_data = human_vector.reshape(1, -1)
        model_data = model_vector.reshape(1, -1)
        
        # Perform canonical correlation analysis
        return self.statistical_analyzer.canonical_correlation_analysis(
            x_data=human_data,
            y_data=model_data,
            x_names=[f"human_dim_{i}" for i in range(human_vector.shape[0])],
            y_names=[f"model_dim_{i}" for i in range(model_vector.shape[0])]
        )
    
    def _interpret_signature_match(self, similarity_metrics: Dict[str, float],
                                 canonical_result: CanonicalCorrelationResult,
                                 significance: bool) -> str:
        """Interpret the signature match results"""
        interpretation = f"Signature Match Analysis:\n\n"
        
        # Similarity metrics interpretation
        interpretation += "Similarity Metrics:\n"
        for metric, value in similarity_metrics.items():
            if metric == "cosine_similarity":
                if value > 0.8:
                    interpretation += f"- {metric}: {value:.3f} (Very High Similarity)\n"
                elif value > 0.6:
                    interpretation += f"- {metric}: {value:.3f} (High Similarity)\n"
                elif value > 0.4:
                    interpretation += f"- {metric}: {value:.3f} (Moderate Similarity)\n"
                else:
                    interpretation += f"- {metric}: {value:.3f} (Low Similarity)\n"
            else:
                interpretation += f"- {metric}: {value:.3f}\n"
        
        # Canonical correlation interpretation
        if len(canonical_result.canonical_correlations) > 0:
            canonical_corr = canonical_result.canonical_correlations[0]
            interpretation += f"\nCanonical Correlation: {canonical_corr:.3f}\n"
            interpretation += f"Significance: {'Significant' if significance else 'Not Significant'}\n"
            
            if canonical_corr > 0.7:
                interpretation += "Interpretation: Strong canonical relationship between human and model signatures.\n"
            elif canonical_corr > 0.5:
                interpretation += "Interpretation: Moderate canonical relationship between human and model signatures.\n"
            else:
                interpretation += "Interpretation: Weak canonical relationship between human and model signatures.\n"
        
        return interpretation
    
    def _calculate_quality_score(self, similarity_metrics: Dict[str, float],
                               canonical_result: CanonicalCorrelationResult) -> float:
        """Calculate overall quality score for the signature match"""
        # Weight different metrics
        weights = {
            "cosine_similarity": 0.3,
            "pearson_correlation": 0.2,
            "canonical_correlation": 0.3,
            "significance": 0.2
        }
        
        quality_score = 0.0
        
        # Add similarity metrics
        if "cosine_similarity" in similarity_metrics:
            quality_score += weights["cosine_similarity"] * similarity_metrics["cosine_similarity"]
        
        if "pearson_correlation" in similarity_metrics:
            quality_score += weights["pearson_correlation"] * abs(similarity_metrics["pearson_correlation"])
        
        # Add canonical correlation
        if len(canonical_result.canonical_correlations) > 0:
            quality_score += weights["canonical_correlation"] * canonical_result.canonical_correlations[0]
        
        # Add significance bonus
        if len(canonical_result.canonical_correlations_pvalues) > 0:
            if canonical_result.canonical_correlations_pvalues[0] < 0.05:
                quality_score += weights["significance"]
        
        return min(quality_score, 1.0)  # Cap at 1.0
    
    def batch_match_signatures(self, human_signatures: List[HumanSignature],
                             model_signatures: List[ModelSignature]) -> List[SignatureMatch]:
        """Perform batch signature matching"""
        matches = []
        
        for human_sig in human_signatures:
            for model_sig in model_signatures:
                match = self.match_signatures(human_sig, model_sig)
                matches.append(match)
        
        return matches
    
    def create_signature_comparison_matrix(self, matches: List[SignatureMatch]) -> pd.DataFrame:
        """Create a comparison matrix of signature matches"""
        # Extract unique human and model signatures
        human_conditions = list(set(match.human_signature.condition for match in matches))
        model_conditions = list(set(match.model_signature.condition for match in matches))
        
        # Create matrix
        matrix_data = []
        for human_condition in human_conditions:
            row = {"human_condition": human_condition}
            for model_condition in model_conditions:
                # Find matching signature
                matching_signature = None
                for match in matches:
                    if (match.human_signature.condition == human_condition and 
                        match.model_signature.condition == model_condition):
                        matching_signature = match
                        break
                
                if matching_signature:
                    row[model_condition] = matching_signature.canonical_correlation
                else:
                    row[model_condition] = 0.0
            
            matrix_data.append(row)
        
        df = pd.DataFrame(matrix_data)
        df.set_index("human_condition", inplace=True)
        
        return df
    
    def visualize_signature_matches(self, matches: List[SignatureMatch], 
                                  output_file: str = None) -> plt.Figure:
        """Create visualization of signature matches"""
        # Create comparison matrix
        comparison_matrix = self.create_signature_comparison_matrix(matches)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(comparison_matrix, annot=True, cmap='RdYlBu_r', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        
        ax.set_title('Human-Model Signature Matching Matrix', fontsize=14, fontweight='bold')
        ax.set_xlabel('Model Conditions', fontsize=12)
        ax.set_ylabel('Human Conditions', fontsize=12)
        
        plt.tight_layout()
        
        if output_file:
            output_path = self.output_dir / output_file
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Signature match visualization saved to: {output_path}")
        
        return fig
    
    def export_signature_matches(self, matches: List[SignatureMatch], 
                               filename: str = None) -> Path:
        """Export signature matches to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"signature_matches_{timestamp}.json"
        
        output_path = self.output_dir / filename
        
        # Convert matches to serializable format
        matches_data = []
        for match in matches:
            match_data = {
                "human_signature": asdict(match.human_signature),
                "model_signature": asdict(match.model_signature),
                "similarity_metrics": match.similarity_metrics,
                "canonical_correlation": match.canonical_correlation,
                "canonical_correlation_pvalue": match.canonical_correlation_pvalue,
                "significance": match.significance,
                "interpretation": match.interpretation,
                "quality_score": match.quality_score
            }
            matches_data.append(match_data)
        
        with open(output_path, 'w') as f:
            json.dump(matches_data, f, indent=2, default=str)
        
        logger.info(f"Signature matches exported to: {output_path}")
        return output_path


def create_sample_data() -> Tuple[List[HumanSignature], List[ModelSignature]]:
    """Create sample data for testing signature matching"""
    # Sample human signatures
    human_signatures = []
    conditions = ["caffeine", "lsd", "mdma", "placebo"]
    
    for i, condition in enumerate(conditions):
        human_sig = HumanSignature(
            participant_id=f"P{i+1:03d}",
            condition=condition,
            assessment_time="T+60",
            subscale_scores={
                "ego_dissolution": np.random.normal(0.5, 0.2),
                "visual_effects": np.random.normal(0.3, 0.1),
                "attention_reduction": np.random.normal(0.4, 0.15),
                "anxiety": np.random.normal(0.2, 0.1),
                "energy": np.random.normal(0.6, 0.2),
                "creativity": np.random.normal(0.4, 0.15)
            },
            total_score=0.0,
            standardized_scores={},
            confidence_interval=(0.0, 0.0),
            effect_size=0.0,
            significance=True
        )
        human_sig.total_score = sum(human_sig.subscale_scores.values())
        human_sig.standardized_scores = {k: v for k, v in human_sig.subscale_scores.items()}
        human_signatures.append(human_sig)
    
    # Sample model signatures
    model_signatures = []
    models = ["Llama-3.1-70B", "Qwen-2.5-7B", "Mixtral-8×22B"]
    
    for model in models:
        for condition in conditions:
            model_sig = ModelSignature(
                model_name=model,
                pack_name=condition,
                condition=condition,
                subscale_scores={
                    "ego_dissolution": np.random.normal(0.4, 0.15),
                    "visual_effects": np.random.normal(0.3, 0.1),
                    "attention_reduction": np.random.normal(0.35, 0.12),
                    "anxiety": np.random.normal(0.25, 0.08),
                    "energy": np.random.normal(0.55, 0.18),
                    "creativity": np.random.normal(0.45, 0.12)
                },
                total_score=0.0,
                standardized_scores={},
                confidence_interval=(0.0, 0.0),
                effect_size=0.0,
                significance=True
            )
            model_sig.total_score = sum(model_sig.subscale_scores.values())
            model_sig.standardized_scores = {k: v for k, v in model_sig.subscale_scores.items()}
            model_signatures.append(model_sig)
    
    return human_signatures, model_signatures


def main():
    """Main function to demonstrate signature matching"""
    print("🔗 Human-Model Signature Matching System Demo")
    print("=" * 60)
    
    # Create signature matcher
    matcher = SignatureMatcher()
    
    # Create sample data
    print("📊 Creating sample signatures...")
    human_signatures, model_signatures = create_sample_data()
    print(f"✅ Created {len(human_signatures)} human signatures")
    print(f"✅ Created {len(model_signatures)} model signatures")
    
    # Perform signature matching
    print("\n🔍 Performing signature matching...")
    matches = matcher.batch_match_signatures(human_signatures, model_signatures)
    print(f"✅ Completed {len(matches)} signature matches")
    
    # Analyze results
    print("\n📈 Analyzing results...")
    high_quality_matches = [m for m in matches if m.quality_score > 0.7]
    significant_matches = [m for m in matches if m.significance]
    
    print(f"   - High quality matches (>0.7): {len(high_quality_matches)}")
    print(f"   - Significant matches: {len(significant_matches)}")
    
    # Create comparison matrix
    comparison_matrix = matcher.create_signature_comparison_matrix(matches)
    print(f"✅ Created comparison matrix: {comparison_matrix.shape}")
    
    # Visualize results
    print("\n🎨 Creating visualizations...")
    fig = matcher.visualize_signature_matches(matches, "signature_matches_heatmap.png")
    print("✅ Signature match heatmap created")
    
    # Export results
    print("\n💾 Exporting results...")
    export_path = matcher.export_signature_matches(matches)
    print(f"✅ Results exported to: {export_path}")
    
    # Show sample results
    print("\n📋 Sample Results:")
    for i, match in enumerate(matches[:3]):
        print(f"\nMatch {i+1}:")
        print(f"   Human: {match.human_signature.condition}")
        print(f"   Model: {match.model_signature.model_name} - {match.model_signature.condition}")
        print(f"   Canonical Correlation: {match.canonical_correlation:.3f}")
        print(f"   Quality Score: {match.quality_score:.3f}")
        print(f"   Significance: {match.significance}")
    
    print("\n🎉 Signature matching system ready!")
    print(f"📂 All files saved to: {matcher.output_dir}")
    print("\n💡 Next steps:")
    print("   1. Collect human reference data using the worksheets")
    print("   2. Run model experiments with neuromodulation packs")
    print("   3. Perform signature matching analysis")
    print("   4. Validate human-model correspondence")


if __name__ == "__main__":
    main()
