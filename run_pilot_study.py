#!/usr/bin/env python3
"""
Pilot Study Execution Script

This script executes the complete pilot study to validate the neuromodulation
framework using a small, fast model.
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

class PilotStudyRunner:
    """Main class for running the pilot study"""
    
    def __init__(self, output_dir: str = "outputs/experiments/runs/pilot"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Study configuration
        self.model = "microsoft/DialoGPT-small"
        self.packs = ["none", "caffeine", "lsd", "placebo"]  # Use 'none' instead of 'control'
        self.tests = ["adq", "cdq", "sdq", "ddq", "pdq", "edq", "pcq_pop", "didq"]
        self.n_samples = 5
        
        # Results tracking
        self.results = {
            "start_time": datetime.now().isoformat(),
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
        model_test_cmd = f"""
cd {project_root} && python -c "
import sys
sys.path.append('.')
try:
    from neuromod.model_support import ModelSupportManager
    manager = ModelSupportManager(test_mode=True)
    model = manager.load_model('{self.model}')
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
        analysis_dir = project_root / "analysis" / "pilot"
        analysis_dir.mkdir(parents=True, exist_ok=True)
        
        # Run statistical analysis
        stats_cmd = f"""
cd {project_root} && python analysis/statistical_analysis.py \
    --input_dir {self.output_dir} \
    --output_dir {analysis_dir} \
    --model {self.model}
"""
        success &= self.run_command(stats_cmd, "Run statistical analysis")
        
        # Run mixed-effects analysis
        mixed_effects_cmd = f"""
cd {project_root} && python -m neuromod.testing.advanced_statistics \
    --data {analysis_dir}/aggregated_data.csv \
    --model mixed_effects \
    --output {analysis_dir}/mixed_effects_results.json
"""
        success &= self.run_command(mixed_effects_cmd, "Run mixed-effects analysis")
        
        # Run power analysis
        power_cmd = f"""
cd {project_root} && python analysis/power_analysis.py \
    --plan analysis/plan.yaml \
    --pilot {self.output_dir}/pilot_data.jsonl \
    --output {analysis_dir}/power_analysis.json
"""
        success &= self.run_command(power_cmd, "Run power analysis")
        
        self.results["phases"]["analysis"] = {
            "success": success,
            "timestamp": datetime.now().isoformat()
        }
        
        return success
    
    def phase_4_visualization(self) -> bool:
        """Phase 4: Visualization and reporting"""
        logger.info("🎨 Starting Phase 4: Visualization and Reporting")
        
        success = True
        
        project_root = Path(__file__).parent.absolute()
        analysis_dir = project_root / "analysis" / "pilot"
        reports_dir = project_root / "reports" / "pilot"
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate figures
        figures_cmd = f"""
cd {project_root} && python -m neuromod.testing.visualization \
    --data {analysis_dir} \
    --output {analysis_dir}/figures \
    --model {self.model}
"""
        success &= self.run_command(figures_cmd, "Generate figures")
        
        # Generate tables
        tables_cmd = f"""
cd {project_root} && python -m neuromod.testing.results_templates \
    --data {analysis_dir} \
    --output {analysis_dir}/tables \
    --model {self.model}
"""
        success &= self.run_command(tables_cmd, "Generate tables")
        
        # Generate reports
        reports_cmd = f"""
cd {project_root} && python analysis/reporting_system.py \
    --input {analysis_dir} \
    --output {reports_dir} \
    --format html,pdf,json
"""
        success &= self.run_command(reports_cmd, "Generate reports")
        
        self.results["phases"]["visualization"] = {
            "success": success,
            "timestamp": datetime.now().isoformat()
        }
        
        return success
    
    def validate_outputs(self) -> bool:
        """Validate that all expected outputs were generated"""
        logger.info("🔍 Validating outputs")
        
        project_root = Path(__file__).parent.absolute()
        analysis_dir = project_root / "analysis" / "pilot"
        reports_dir = project_root / "reports" / "pilot"
        
        expected_files = [
            # Basic files in output directory
            str(Path(self.output_dir) / "freeze.txt"),
            str(Path(self.output_dir) / "git_sha.txt"), 
            str(Path(self.output_dir) / "plan.yaml"),
            # Analysis results (only check what actually gets created)
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
    
    def run_pilot_study(self) -> bool:
        """Run the complete pilot study"""
        logger.info("🎯 Starting Pilot Study")
        logger.info(f"Model: {self.model}")
        logger.info(f"Packs: {', '.join(self.packs)}")
        logger.info(f"Tests: {', '.join(self.tests)}")
        logger.info(f"Samples per condition: {self.n_samples}")
        logger.info(f"Output directory: {self.output_dir}")
        
        # Run all phases
        success = True
        success &= self.phase_1_setup()
        success &= self.phase_2_data_collection()
        success &= self.phase_3_analysis()
        success &= self.phase_4_visualization()
        success &= self.validate_outputs()
        
        # Record final results
        self.results["end_time"] = datetime.now().isoformat()
        self.results["success"] = success
        
        # Save results
        results_file = self.output_dir / "pilot_study_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        if success:
            logger.info("🎉 Pilot study completed successfully!")
            logger.info(f"Results saved to: {results_file}")
        else:
            logger.error("💥 Pilot study failed!")
            logger.error(f"Error details saved to: {results_file}")
        
        return success


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run pilot study")
    parser.add_argument("--output-dir", default="outputs/experiments/runs/pilot",
                       help="Output directory for pilot study")
    parser.add_argument("--model", default="microsoft/DialoGPT-small",
                       help="Model to use for pilot study")
    parser.add_argument("--n-samples", type=int, default=5,
                       help="Number of samples per condition")
    
    args = parser.parse_args()
    
    # Create and run pilot study
    runner = PilotStudyRunner(args.output_dir)
    runner.model = args.model
    runner.n_samples = args.n_samples
    
    success = runner.run_pilot_study()
    
    if success:
        print("\n🎉 Pilot study completed successfully!")
        print("The framework is ready for full-scale study.")
        sys.exit(0)
    else:
        print("\n💥 Pilot study failed!")
        print("Please check the logs and fix any issues before proceeding.")
        sys.exit(1)


if __name__ == "__main__":
    main()
