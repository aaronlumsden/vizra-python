"""
Training runner for discovering and executing RL training routines.
"""

import importlib
import inspect
import pkgutil
from typing import Dict, List, Any, Optional, Type
from pathlib import Path
import json
from .base import BaseRLTraining


class TrainingRunner:
    """
    Runner for discovering and executing training classes.
    """
    
    def __init__(self, training_paths: Optional[List[str]] = None):
        """
        Initialize the training runner.
        
        Args:
            training_paths: List of module paths to search for training classes
        """
        self.training_paths = training_paths or ['training']
        self.trainings: Dict[str, Type[BaseRLTraining]] = {}
        self._discover_trainings()
    
    def _discover_trainings(self) -> None:
        """Discover all training classes."""
        for module_path in self.training_paths:
            try:
                # Import the module
                module = importlib.import_module(module_path)
                
                # Walk through all modules in the package
                if hasattr(module, '__path__'):
                    for importer, modname, ispkg in pkgutil.walk_packages(
                        path=module.__path__,
                        prefix=module.__name__ + '.',
                        onerror=lambda x: None
                    ):
                        try:
                            submodule = importlib.import_module(modname)
                            self._register_trainings_from_module(submodule)
                        except Exception:
                            pass
                else:
                    # Single module, not a package
                    self._register_trainings_from_module(module)
                    
            except ImportError:
                pass
    
    def _register_trainings_from_module(self, module) -> None:
        """Register all BaseRLTraining subclasses from a module."""
        for name, obj in inspect.getmembers(module):
            if (inspect.isclass(obj) and 
                issubclass(obj, BaseRLTraining) and 
                obj is not BaseRLTraining and
                hasattr(obj, 'name')):
                self.trainings[obj.name] = obj
    
    def list_trainings(self) -> List[Dict[str, Any]]:
        """
        List all discovered training routines.
        
        Returns:
            List of dicts with training info
        """
        return [
            {
                'name': train_class.name,
                'description': train_class.description,
                'agent_name': train_class.agent_name,
                'algorithm': train_class.algorithm,
                'class': f"{train_class.__module__}.{train_class.__name__}"
            }
            for train_class in self.trainings.values()
        ]
    
    def run_training(self, training_name: str) -> Dict[str, Any]:
        """
        Run a specific training routine by name.
        
        Args:
            training_name: Name of the training to run
            
        Returns:
            dict: Training results
        """
        if training_name not in self.trainings:
            available = ', '.join(self.trainings.keys())
            raise ValueError(f"Training '{training_name}' not found. Available: {available}")
        
        train_class = self.trainings[training_name]
        print(f"🔍 Found training class: {train_class}")
        print(f"🔍 Creating instance of {train_class.__name__}...")
        training = train_class()
        print(f"🔍 Instance created: {training}")
        print(f"🔍 Provider is: {getattr(training, 'provider', 'No provider attribute')}")
        return training.run()
    
    def run_all_trainings(self) -> Dict[str, Any]:
        """
        Run all discovered training routines.
        
        Returns:
            dict: Results from all trainings
        """
        results = {}
        
        print("\n" + "="*60)
        print("🚀 Running All Training Routines")
        print("="*60)
        
        for train_name in self.trainings:
            try:
                results[train_name] = self.run_training(train_name)
            except Exception as e:
                print(f"\n❌ Failed to run training '{train_name}': {e}")
                results[train_name] = {
                    'error': str(e),
                    'status': 'failed'
                }
        
        # Summary
        total_trainings = len(results)
        successful = sum(1 for r in results.values() if 'error' not in r)
        
        print("\n" + "="*60)
        print("📊 Overall Training Summary")
        print(f"   Total training routines: {total_trainings}")
        print(f"   Successful: {successful}")
        print(f"   Failed: {total_trainings - successful}")
        
        # Show individual training summaries
        for train_name, result in results.items():
            if 'error' not in result:
                best_reward = result.get('best_reward', 0)
                iterations = result.get('total_iterations', 0)
                print(f"\n   {train_name}:")
                print(f"      Best reward: {best_reward:.3f}")
                print(f"      Iterations: {iterations}")
        
        print("="*60 + "\n")
        
        return results
    
    def generate_report(self, results: Dict[str, Any], output_path: Optional[str] = None) -> str:
        """
        Generate a detailed report from training results.
        
        Args:
            results: Results from run_training or run_all_trainings
            output_path: Optional path to save the report
            
        Returns:
            str: Report content
        """
        report_lines = [
            "# Vizra Agent RL Training Report",
            "=" * 60,
            ""
        ]
        
        # Handle single training result
        if 'training_name' in results:
            results = {results['training_name']: results}
        
        # Summary section
        total_trainings = len(results)
        successful = sum(1 for r in results.values() if 'error' not in r)
        
        report_lines.extend([
            "## Summary",
            f"- Total training routines: {total_trainings}",
            f"- Successful: {successful}",
            f"- Failed: {total_trainings - successful}",
            ""
        ])
        
        # Individual training details
        for train_name, result in results.items():
            report_lines.extend([
                f"## Training: {train_name}",
                "-" * 40
            ])
            
            if 'error' in result:
                report_lines.extend([
                    f"Status: ❌ FAILED",
                    f"Error: {result['error']}",
                    ""
                ])
                continue
            
            # Hyperparameters
            hyperparams = result.get('hyperparameters', {})
            report_lines.extend([
                f"Agent: {result.get('agent_name', 'Unknown')}",
                f"Algorithm: {hyperparams.get('algorithm', 'Unknown')}",
                f"Learning rate: {hyperparams.get('learning_rate', 'N/A')}",
                f"Batch size: {hyperparams.get('batch_size', 'N/A')}",
                f"Total iterations: {result.get('total_iterations', 0)}",
                ""
            ])
            
            # Performance metrics
            report_lines.extend([
                "### Performance",
                f"Best reward: {result.get('best_reward', 0):.3f}",
            ])
            
            final_metrics = result.get('final_metrics', {})
            if final_metrics:
                report_lines.extend([
                    f"Final average reward: {final_metrics.get('avg_reward', 0):.3f}",
                    f"Final success rate: {final_metrics.get('success_rate', 0):.1%}",
                    f"Final reward range: [{final_metrics.get('min_reward', 0):.3f}, {final_metrics.get('max_reward', 0):.3f}]",
                ])
            
            # Training history summary
            if 'training_history' in result and result['training_history']:
                history = result['training_history']
                rewards = [h['avg_reward'] for h in history]
                
                report_lines.extend([
                    "",
                    "### Training Progress",
                    f"Starting reward: {rewards[0]:.3f}",
                    f"Ending reward: {rewards[-1]:.3f}",
                    f"Improvement: {(rewards[-1] - rewards[0]):.3f}",
                ])
            
            report_lines.append("")
        
        report = "\n".join(report_lines)
        
        # Save report if path provided
        if output_path:
            Path(output_path).write_text(report)
            print(f"\n📄 Report saved to: {output_path}")
        
        return report