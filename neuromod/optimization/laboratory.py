"""
Drug Design Laboratory

Interactive interface for designing and optimizing neuromodulation packs
to achieve specific behavioral and emotional outcomes.
"""

import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
import time
import torch

from .targets import BehavioralTarget, TargetManager, create_joyful_social_target
from .evaluation import EvaluationFramework, BehavioralMetrics
from .pack_optimizer import PackOptimizer, OptimizationConfig, OptimizationMethod, OptimizationResult
from ..pack_system import Pack, PackManager, PackRegistry
from ..model_support import ModelSupportManager

logger = logging.getLogger(__name__)

@dataclass
class LaboratorySession:
    """Represents a drug design laboratory session"""
    session_id: str
    target: BehavioralTarget
    base_pack: Pack
    optimized_pack: Optional[Pack] = None
    optimization_result: Optional[Any] = None
    test_results: List[Dict[str, Any]] = None
    created_at: float = 0.0
    
    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()
        if self.test_results is None:
            self.test_results = []

class DrugDesignLab:
    """Interactive drug design laboratory"""
    
    def __init__(self, 
                 model_manager: ModelSupportManager = None,
                 evaluation_framework: EvaluationFramework = None,
                 model_name: str = "meta-llama/Llama-3.1-8B-Instruct"):
        """
        Initialize drug design laboratory.
        
        Args:
            model_manager: Model support manager (if None, creates one)
            evaluation_framework: Evaluation framework (if None, creates one)
            model_name: Target model for optimization (default: Llama-3.1-8B-Instruct)
                       CRITICAL: Must match the model used for final evaluation.
        """
        self.model_manager = model_manager or ModelSupportManager(test_mode=True)
        self.evaluation_framework = evaluation_framework or EvaluationFramework()
        self.target_manager = TargetManager()
        self.pack_registry = PackRegistry()
        self.sessions: Dict[str, LaboratorySession] = {}
        self.model_name = model_name  # Store target model name
        
        # Initialize optimizer with target model (model_name in config)
        config = OptimizationConfig(model_name=self.model_name)
        self.optimizer = PackOptimizer(
            self.model_manager,
            self.evaluation_framework,
            config
        )
    
    def create_session(self, 
                      target_name: str,
                      base_pack_name: str = "none",
                      session_id: str = None) -> LaboratorySession:
        """Create a new laboratory session"""
        
        if session_id is None:
            session_id = f"session_{int(time.time())}"
        
        # Get target
        target = self.target_manager.get_target(target_name)
        if not target:
            raise ValueError(f"Target not found: {target_name}")
        
        # Get base pack
        base_pack = self.pack_registry.get_pack(base_pack_name)
        if not base_pack:
            raise ValueError(f"Base pack not found: {base_pack_name}")
        
        # Create session
        session = LaboratorySession(
            session_id=session_id,
            target=target,
            base_pack=base_pack
        )
        
        self.sessions[session_id] = session
        logger.info(f"Created laboratory session: {session_id}")
        
        return session
    
    def optimize_pack(self, 
                     session_id: str,
                     method: OptimizationMethod = OptimizationMethod.RANDOM_SEARCH,
                     max_iterations: int = 50) -> OptimizationResult:
        """Optimize pack for the session target"""
        
        session = self.sessions.get(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")
        
        logger.info(f"Starting optimization for session: {session_id}")
        logger.info(f"Target: {session.target.name}")
        logger.info(f"Method: {method.value}")
        
        # Update optimizer config
        config = OptimizationConfig(
            method=method,
            max_iterations=max_iterations
        )
        self.optimizer.config = config
        
        # Run optimization
        result = self.optimizer.optimize_pack(
            session.base_pack,
            session.target,
            session.target.test_prompts
        )
        
        # Update session
        session.optimized_pack = result.optimized_pack
        session.optimization_result = result
        
        logger.info(f"Optimization completed. Final loss: {result.final_loss:.4f}")
        
        return result
    
    def test_pack(self, 
                 session_id: str,
                 test_prompts: List[str] = None,
                 custom_pack: Pack = None) -> Dict[str, Any]:
        """Test a pack with custom prompts"""
        
        session = self.sessions.get(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")
        
        if test_prompts is None:
            test_prompts = session.target.test_prompts
        
        pack_to_test = custom_pack or session.optimized_pack or session.base_pack
        
        logger.info(f"Testing pack for session: {session_id}")
        logger.info(f"Test prompts: {len(test_prompts)}")
        
        try:
            # Load model
            # CRITICAL: Use the target model, not a hardcoded test model
            # Transferability between architectures (GPT-2 vs Llama-3) is zero
            model, tokenizer, _ = self.model_manager.load_model(self.model_name)
            
            # Apply pack
            pack_manager = PackManager()
            pack_manager.apply_pack(pack_to_test, model)
            
            # Generate responses
            responses = []
            for prompt in test_prompts:
                inputs = tokenizer(prompt, return_tensors="pt")
                with torch.no_grad():
                    outputs = model.generate(
                        inputs.input_ids,
                        max_new_tokens=50,
                        num_beams=1,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=tokenizer.eos_token_id
                    )
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                responses.append(response)
            
            # Evaluate responses
            actual_metrics = self.evaluation_framework.evaluate_texts(responses)
            
            # Compute target loss
            target_loss = session.target.compute_loss(actual_metrics.get_all_metrics())
            
            # Store test results
            test_result = {
                'timestamp': time.time(),
                'test_prompts': test_prompts,
                'responses': responses,
                'metrics': actual_metrics.to_dict(),
                'target_loss': target_loss,
                'pack_name': pack_to_test.name if hasattr(pack_to_test, 'name') else 'unknown'
            }
            
            session.test_results.append(test_result)
            
            logger.info(f"Test completed. Target loss: {target_loss:.4f}")
            
            return test_result
            
        except Exception as e:
            logger.error(f"Error testing pack: {e}")
            raise
    
    def compare_packs(self, 
                     session_id: str,
                     test_prompts: List[str] = None) -> Dict[str, Any]:
        """Compare base pack vs optimized pack"""
        
        session = self.sessions.get(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")
        
        if not session.optimized_pack:
            raise ValueError("No optimized pack available for comparison")
        
        logger.info(f"Comparing packs for session: {session_id}")
        
        # Test base pack
        base_result = self.test_pack(session_id, test_prompts, session.base_pack)
        
        # Test optimized pack
        optimized_result = self.test_pack(session_id, test_prompts, session.optimized_pack)
        
        # Compare results
        comparison = {
            'base_pack': {
                'metrics': base_result['metrics'],
                'target_loss': base_result['target_loss']
            },
            'optimized_pack': {
                'metrics': optimized_result['metrics'],
                'target_loss': optimized_result['target_loss']
            },
            'improvement': {
                'loss_reduction': base_result['target_loss'] - optimized_result['target_loss'],
                'relative_improvement': (base_result['target_loss'] - optimized_result['target_loss']) / base_result['target_loss'] if base_result['target_loss'] > 0 else 0
            }
        }
        
        logger.info(f"Comparison completed. Loss reduction: {comparison['improvement']['loss_reduction']:.4f}")
        
        return comparison
    
    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get summary of a laboratory session"""
        
        session = self.sessions.get(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")
        
        summary = {
            'session_id': session.session_id,
            'target': {
                'name': session.target.name,
                'description': session.target.description,
                'targets': [t.name for t in session.target.targets]
            },
            'base_pack': session.base_pack.name if hasattr(session.base_pack, 'name') else 'unknown',
            'optimized_pack': session.optimized_pack.name if session.optimized_pack and hasattr(session.optimized_pack, 'name') else None,
            'optimization_result': {
                'final_loss': session.optimization_result.final_loss if session.optimization_result else None,
                'success': session.optimization_result.success if session.optimization_result else None,
                'iterations': session.optimization_result.convergence_iteration if session.optimization_result else None
            },
            'test_results_count': len(session.test_results),
            'created_at': session.created_at,
            'duration': time.time() - session.created_at
        }
        
        return summary
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all laboratory sessions"""
        return [self.get_session_summary(session_id) for session_id in self.sessions.keys()]
    
    def save_session(self, session_id: str, filepath: str):
        """Save session to file"""
        session = self.sessions.get(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")
        
        # Convert session to serializable format
        session_data = {
            'session_id': session.session_id,
            'target': session.target.to_dict(),
            'base_pack': session.base_pack.name if hasattr(session.base_pack, 'name') else 'unknown',
            'optimized_pack': session.optimized_pack.name if session.optimized_pack and hasattr(session.optimized_pack, 'name') else None,
            'test_results': session.test_results,
            'created_at': session.created_at
        }
        
        with open(filepath, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        logger.info(f"Session saved to: {filepath}")
    
    def load_session(self, filepath: str) -> str:
        """Load session from file"""
        with open(filepath, 'r') as f:
            session_data = json.load(f)
        
        # Recreate session
        target = BehavioralTarget.from_dict(session_data['target'])
        base_pack = self.pack_registry.get_pack(session_data['base_pack'])
        
        session = LaboratorySession(
            session_id=session_data['session_id'],
            target=target,
            base_pack=base_pack,
            test_results=session_data.get('test_results', []),
            created_at=session_data.get('created_at', time.time())
        )
        
        self.sessions[session.session_id] = session
        logger.info(f"Session loaded from: {filepath}")
        
        return session.session_id

# Convenience functions for easy lab usage
def create_lab() -> DrugDesignLab:
    """Create a new drug design laboratory"""
    return DrugDesignLab()

def design_joyful_social_drug() -> Tuple[str, OptimizationResult]:
    """Design a drug that increases joy and socialization"""
    lab = create_lab()
    session = lab.create_session("joyful_social", "none")
    result = lab.optimize_pack(session.session_id, OptimizationMethod.RANDOM_SEARCH)
    return session.session_id, result

def design_creative_focused_drug() -> Tuple[str, OptimizationResult]:
    """Design a drug that enhances creativity and focus"""
    lab = create_lab()
    session = lab.create_session("creative_focused", "none")
    result = lab.optimize_pack(session.session_id, OptimizationMethod.RANDOM_SEARCH)
    return session.session_id, result

def design_calm_reflective_drug() -> Tuple[str, OptimizationResult]:
    """Design a drug that promotes calmness and reflection"""
    lab = create_lab()
    session = lab.create_session("calm_reflective", "none")
    result = lab.optimize_pack(session.session_id, OptimizationMethod.RANDOM_SEARCH)
    return session.session_id, result
