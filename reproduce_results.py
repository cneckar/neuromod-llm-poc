#!/usr/bin/env python3
"""
Reproduction Script: The Golden Path for Reproducing Paper Results

This script reproduces the experiments described in the paper using the same
model and sample sizes as the published results.

Default behavior:
- Model: meta-llama/Llama-3.1-8B-Instruct (paper model)
- Sample size: n=126 per condition (paper sample size)

Use --test-mode for quick validation:
- Model: microsoft/DialoGPT-small (fast test model)
- Sample size: n=5 per condition (quick validation)
"""

import os
import sys
import subprocess
import json
import time
from pathlib import Path
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ReproductionRunner:
    """Main class for reproducing paper results"""
    
    def __init__(self, output_dir: str = "outputs/experiments/runs/reproduction", test_mode: bool = False):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.test_mode = test_mode
        
        # Study configuration - defaults match paper
        if test_mode:
            # Test mode: fast validation
            self.model = "microsoft/DialoGPT-small"
            self.n_samples = 5
            logger.info("🧪 Running in TEST MODE (fast validation)")
        else:
            # Production mode: paper reproduction
            self.model = "meta-llama/Llama-3.1-8B-Instruct"
            self.n_samples = 126  # Paper sample size: n=126 per condition
            logger.info("📊 Running in PRODUCTION MODE (paper reproduction)")
        
        self.packs = ["none", "caffeine", "lsd", "placebo"]  # Use 'none' instead of 'control'
        self.tests = ["adq", "cdq", "sdq", "ddq", "pdq", "edq", "pcq_pop", "didq"]
        
        # Results tracking
        self.results = {
            "start_time": datetime.now().isoformat(),
            "test_mode": test_mode,
            "model": self.model,
            "n_samples": self.n_samples,
            "phases": {},
            "errors": [],
            "success": False
        }
    
    def run_command(self, command: str, description: str) -> bool:
        """Run a command and track results"""
        logger.info(f"Running: {description}")
        logger.info(f"Command: {command}")
        
        try:
            result = subprocess.run(
                command, 
                shell=True, 
                capture_output=True, 
                text=True, 
                timeout=3600  # 1 hour timeout
            )
            
            if result.returncode == 0:
                logger.info(f"✅ {description} completed successfully")
                return True
            else:
                logger.error(f"❌ {description} failed")
                logger.error(f"Error: {result.stderr}")
                self.results["errors"].append({
                    "phase": description,
                    "command": command,
                    "error": result.stderr
                })
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"⏰ {description} timed out")
            self.results["errors"].append({
                "phase": description,
                "command": command,
                "error": "Command timed out after 1 hour"
            })
            return False
        except Exception as e:
            logger.error(f"💥 {description} failed with exception: {e}")
            self.results["errors"].append({
                "phase": description,
                "command": command,
                "error": str(e)
            })
            return False
    
    def phase_1_setup(self) -> bool:
        """Phase 1: Setup and validation"""
        logger.info("🚀 Starting Phase 1: Setup and Validation")
        
        # Get the project root directory
        project_root = Path(__file__).parent.absolute()
        
        # Record environment
        success = True
        success &= self.run_command(f"pip freeze > {self.output_dir}/freeze.txt", "Record environment dependencies")
        success &= self.run_command(f"git rev-parse HEAD > {self.output_dir}/git_sha.txt", "Record git commit hash")
        success &= self.run_command(f"cp {project_root}/analysis/plan.yaml {self.output_dir}/", "Copy study plan")
        
        # Test model loading
        test_mode_flag = "--test-mode" if self.test_mode else ""
        model_test_cmd = f"""
cd {project_root} && python -c "
import sys
sys.path.append('.')
try:
    from neuromod.model_support import ModelSupportManager
    manager = ModelSupportManager(test_mode={str(self.test_mode).lower()})
    model, tokenizer, model_info = manager.load_model('{self.model}')
    print('✅ Model loaded successfully')
except Exception as e:
    print(f'❌ Model loading failed: {{e}}')
    sys.exit(1)
"
"""
        success &= self.run_command(model_test_cmd, "Test model loading")
        
        # Test pack validation
        pack_test_cmd = f"""
cd {project_root} && python -c "
import sys
sys.path.append('.')
try:
    from neuromod.pack_system import PackRegistry
    pack_registry = PackRegistry()
    packs = {self.packs}
    for pack in packs:
        pack_data = pack_registry.get_pack(pack)
        print(f'✅ Pack {{pack}} validated')
    print('✅ All packs validated successfully')
except Exception as e:
    print(f'❌ Pack validation failed: {{e}}')
    sys.exit(1)
"
"""
        success &= self.run_command(pack_test_cmd, "Validate packs")
        
        self.results["phases"]["setup"] = {
            "success": success,
            "timestamp": datetime.now().isoformat()
        }
        
        return success
    
    def phase_2_data_collection(self) -> bool:
        """Phase 2: Data collection"""
        logger.info("📊 Starting Phase 2: Data Collection")
        
        project_root = Path(__file__).parent.absolute()
        success = True
        
        # Run psychometric tests
        psychometric_cmd = f"""
cd {project_root} && python -m neuromod.testing.test_runner \
    --model {self.model} \
    --packs {','.join(self.packs)} \
    --all
"""
        success &= self.run_command(psychometric_cmd, "Run psychometric tests")
        
        # Run cognitive tasks
        cognitive_cmd = f"""
cd {project_root} && python -m neuromod.testing.cognitive_tasks \
    --model {self.model} \
    --packs {','.join(self.packs)} \
    --n_samples {self.n_samples} \
    --output_dir {self.output_dir}/cognitive
"""
        success &= self.run_command(cognitive_cmd, "Run cognitive tasks")
        
        # Run telemetry collection
        telemetry_cmd = f"""
cd {project_root} && python -m neuromod.testing.telemetry \
    --model {self.model} \
    --packs {','.join(self.packs)} \
    --n_samples {self.n_samples} \
    --output_dir {self.output_dir}/telemetry
"""
        success &= self.run_command(telemetry_cmd, "Run telemetry collection")
        
        # Calculate endpoints
        endpoint_cmd = f"""
cd {project_root} && python scripts/calculate_endpoints.py \
    --pack caffeine \
    --model {self.model}
"""
        success &= self.run_command(endpoint_cmd, "Calculate endpoints (caffeine)")
        
        endpoint_cmd = f"""
cd {project_root} && python scripts/calculate_endpoints.py \
    --pack lsd \
    --model {self.model}
"""
        success &= self.run_command(endpoint_cmd, "Calculate endpoints (lsd)")
        
        self.results["phases"]["data_collection"] = {
            "success": success,
            "timestamp": datetime.now().isoformat()
        }
        
        return success
    
    def phase_3_analysis(self) -> bool:
        """Phase 3: Statistical analysis"""
        logger.info("📈 Starting Phase 3: Statistical Analysis")
        
        success = True
        
        # Create analysis output directory
        project_root = Path(__file__).parent.absolute()
        analysis_dir = project_root / "outputs" / "analysis" / "reproduction"
        analysis_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert endpoints to NDJSON
        export_cmd = f"""
cd {project_root} && python scripts/export_endpoints_to_ndjson.py \
    --input-dir outputs/endpoints \
    --output outputs/endpoints/reproduction_data.jsonl
"""
        success &= self.run_command(export_cmd, "Export endpoints to NDJSON")
        
        # Run statistical analysis
        stats_cmd = f"""
cd {project_root} && python scripts/analyze_endpoints.py \
    --input-dir outputs/endpoints \
    --output {analysis_dir}/statistical_results.json
"""
        success &= self.run_command(stats_cmd, "Run statistical analysis")
        
        # Run power analysis
        power_cmd = f"""
cd {project_root} && python analysis/power_analysis.py \
    --plan analysis/plan.yaml \
    --pilot outputs/endpoints/reproduction_data.jsonl \
    --output {analysis_dir}/power_analysis.json
"""
        success &= self.run_command(power_cmd, "Run power analysis")
        
        self.results["phases"]["analysis"] = {
            "success": success,
            "timestamp": datetime.now().isoformat()
        }
        
        return success
    
    def validate_outputs(self) -> bool:
        """Validate that all expected outputs were generated"""
        logger.info("🔍 Validating outputs")
        
        project_root = Path(__file__).parent.absolute()
        analysis_dir = project_root / "outputs" / "analysis" / "reproduction"
        
        expected_files = [
            # Basic files in output directory
            str(Path(self.output_dir) / "freeze.txt"),
            str(Path(self.output_dir) / "git_sha.txt"), 
            str(Path(self.output_dir) / "plan.yaml"),
            # Analysis results
            str(analysis_dir / "power_analysis.json"),
        ]
        
        missing_files = []
        for file_path in expected_files:
            if not Path(file_path).exists():
                missing_files.append(file_path)
        
        if missing_files:
            logger.error(f"❌ Missing {len(missing_files)} expected files:")
            for file_path in missing_files:
                logger.error(f"   - {file_path}")
            return False
        else:
            logger.info(f"✅ All {len(expected_files)} expected files generated")
            return True
    
    def run_reproduction(self) -> bool:
        """Run the complete reproduction study"""
        logger.info("🎯 Starting Reproduction Study")
        logger.info(f"Model: {self.model}")
        logger.info(f"Packs: {', '.join(self.packs)}")
        logger.info(f"Tests: {', '.join(self.tests)}")
        logger.info(f"Samples per condition: {self.n_samples}")
        logger.info(f"Output directory: {self.output_dir}")
        
        if self.test_mode:
            logger.info("⚠️  TEST MODE: Using fast model and small sample size for validation")
            logger.info("   For paper reproduction, run without --test-mode flag")
        else:
            logger.info("📊 PRODUCTION MODE: Reproducing paper results")
            logger.info(f"   Model: {self.model} (paper model)")
            logger.info(f"   Sample size: n={self.n_samples} per condition (paper sample size)")
        
        # Run all phases
        success = True
        success &= self.phase_1_setup()
        success &= self.phase_2_data_collection()
        success &= self.phase_3_analysis()
        success &= self.validate_outputs()
        
        # Record final results
        self.results["end_time"] = datetime.now().isoformat()
        self.results["success"] = success
        
        # Save results
        results_file = self.output_dir / "reproduction_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        if success:
            logger.info("🎉 Reproduction study completed successfully!")
            logger.info(f"Results saved to: {results_file}")
        else:
            logger.error("💥 Reproduction study failed!")
            logger.error(f"Error details saved to: {results_file}")
        
        return success


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Reproduce paper results (The Golden Path)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Reproduce paper results (default: Llama-3.1-8B, n=126)
  python reproduce_results.py

  # Quick validation (test mode: DialoGPT, n=5)
  python reproduce_results.py --test-mode

  # Custom model and sample size
  python reproduce_results.py --model meta-llama/Llama-3.1-70B-Instruct --n-samples 200
        """
    )
    parser.add_argument("--output-dir", default="outputs/experiments/runs/reproduction",
                       help="Output directory for reproduction study")
    parser.add_argument("--model", default=None,
                       help="Model to use (default: meta-llama/Llama-3.1-8B-Instruct in production, microsoft/DialoGPT-small in test mode)")
    parser.add_argument("--n-samples", type=int, default=None,
                       help="Number of samples per condition (default: 126 in production, 5 in test mode)")
    parser.add_argument("--test-mode", action="store_true",
                       help="Use test mode: fast model (DialoGPT) and small sample size (n=5) for quick validation")
    
    args = parser.parse_args()
    
    # Create and run reproduction study
    runner = ReproductionRunner(args.output_dir, test_mode=args.test_mode)
    
    # Override defaults if explicitly provided
    if args.model:
        runner.model = args.model
    if args.n_samples:
        runner.n_samples = args.n_samples
    
    success = runner.run_reproduction()
    
    if success:
        print("\n🎉 Reproduction study completed successfully!")
        if args.test_mode:
            print("⚠️  This was a TEST MODE run (fast validation).")
            print("   For paper reproduction, run without --test-mode flag.")
        else:
            print("📊 This was a PRODUCTION MODE run (paper reproduction).")
        sys.exit(0)
    else:
        print("\n💥 Reproduction study failed!")
        print("Please check the logs and fix any issues before proceeding.")
        sys.exit(1)


if __name__ == "__main__":
    main()

