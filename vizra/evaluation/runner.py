"""
Evaluation runner for discovering and executing evaluations.
"""

import importlib
import inspect
import pkgutil
from typing import Dict, List, Any, Optional, Type
from pathlib import Path
import json
from .base import BaseEvaluation


class EvaluationRunner:
    """
    Runner for discovering and executing evaluation classes.
    """
    
    def __init__(self, evaluation_paths: Optional[List[str]] = None):
        """
        Initialize the evaluation runner.
        
        Args:
            evaluation_paths: List of module paths to search for evaluations
        """
        self.evaluation_paths = evaluation_paths or ['evaluations']
        self.evaluations: Dict[str, Type[BaseEvaluation]] = {}
        self._discover_evaluations()
    
    def _discover_evaluations(self) -> None:
        """Discover all evaluation classes."""
        for module_path in self.evaluation_paths:
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
                            self._register_evaluations_from_module(submodule)
                        except Exception:
                            pass
                else:
                    # Single module, not a package
                    self._register_evaluations_from_module(module)
                    
            except ImportError:
                pass
    
    def _register_evaluations_from_module(self, module) -> None:
        """Register all BaseEvaluation subclasses from a module."""
        for name, obj in inspect.getmembers(module):
            if (inspect.isclass(obj) and 
                issubclass(obj, BaseEvaluation) and 
                obj is not BaseEvaluation and
                hasattr(obj, 'name')):
                self.evaluations[obj.name] = obj
    
    def list_evaluations(self) -> List[Dict[str, str]]:
        """
        List all discovered evaluations.
        
        Returns:
            List of dicts with evaluation info
        """
        return [
            {
                'name': eval_class.name,
                'description': eval_class.description,
                'agent_name': eval_class.agent_name,
                'class': f"{eval_class.__module__}.{eval_class.__name__}"
            }
            for eval_class in self.evaluations.values()
        ]
    
    def run_evaluation(self, evaluation_name: str, limit: Optional[int] = None) -> Dict[str, Any]:
        """
        Run a specific evaluation by name.
        
        Args:
            evaluation_name: Name of the evaluation to run
            limit: Optional limit on number of test cases to run
            
        Returns:
            dict: Evaluation results
        """
        if evaluation_name not in self.evaluations:
            available = ', '.join(self.evaluations.keys())
            raise ValueError(f"Evaluation '{evaluation_name}' not found. Available: {available}")
        
        eval_class = self.evaluations[evaluation_name]
        evaluation = eval_class()
        return evaluation.run(limit=limit)
    
    def run_all_evaluations(self) -> Dict[str, Any]:
        """
        Run all discovered evaluations.
        
        Returns:
            dict: Results from all evaluations
        """
        results = {}
        
        print("\n" + "="*60)
        print("🚀 Running All Evaluations")
        print("="*60)
        
        for eval_name in self.evaluations:
            try:
                results[eval_name] = self.run_evaluation(eval_name)
            except Exception as e:
                print(f"\n❌ Failed to run evaluation '{eval_name}': {e}")
                results[eval_name] = {
                    'error': str(e),
                    'status': 'failed'
                }
        
        # Summary
        total_evals = len(results)
        successful = sum(1 for r in results.values() if 'error' not in r)
        
        print("\n" + "="*60)
        print("📊 Overall Summary")
        print(f"   Total evaluations: {total_evals}")
        print(f"   Successful: {successful}")
        print(f"   Failed: {total_evals - successful}")
        
        # Show individual evaluation summaries
        for eval_name, result in results.items():
            if 'error' not in result:
                success_rate = result.get('success_rate', 0)
                print(f"\n   {eval_name}:")
                print(f"      Success rate: {success_rate:.1f}%")
                print(f"      Passed: {result.get('passed', 0)}/{result.get('total_cases', 0)}")
        
        print("="*60 + "\n")
        
        return results
    
    def generate_report(self, results: Dict[str, Any], output_path: Optional[str] = None) -> str:
        """
        Generate a detailed report from evaluation results.
        
        Args:
            results: Results from run_evaluation or run_all_evaluations
            output_path: Optional path to save the report
            
        Returns:
            str: Report content
        """
        report_lines = [
            "# Vizra Agent Evaluation Report",
            "=" * 60,
            ""
        ]
        
        # Handle single evaluation result
        if 'evaluation_name' in results:
            results = {results['evaluation_name']: results}
        
        # Summary section
        total_evals = len(results)
        successful = sum(1 for r in results.values() if 'error' not in r)
        
        report_lines.extend([
            "## Summary",
            f"- Total evaluations: {total_evals}",
            f"- Successful: {successful}",
            f"- Failed: {total_evals - successful}",
            ""
        ])
        
        # Individual evaluation details
        for eval_name, result in results.items():
            report_lines.extend([
                f"## Evaluation: {eval_name}",
                "-" * 40
            ])
            
            if 'error' in result:
                report_lines.extend([
                    f"Status: ❌ FAILED",
                    f"Error: {result['error']}",
                    ""
                ])
                continue
            
            report_lines.extend([
                f"Agent: {result.get('agent_name', 'Unknown')}",
                f"Total test cases: {result.get('total_cases', 0)}",
                f"Passed: {result.get('passed', 0)}",
                f"Failed: {result.get('failed', 0)}",
                f"Success rate: {result.get('success_rate', 0):.1f}%",
                ""
            ])
            
            # Failed cases details
            if 'results' in result:
                failed_cases = [r for r in result['results'] if not r.get('passed', False)]
                if failed_cases:
                    report_lines.append("### Failed Cases:")
                    for i, case in enumerate(failed_cases[:5]):  # Show first 5 failures
                        report_lines.extend([
                            f"\n{i+1}. Prompt: {case.get('prompt', 'N/A')[:100]}...",
                            f"   Response: {case.get('response', 'N/A')[:100]}..."
                        ])
                        
                        if 'assertions' in case:
                            failed_assertions = [a for a in case['assertions'] if a['status'] == 'failed']
                            for assertion in failed_assertions:
                                report_lines.append(f"   ❌ {assertion['assertion']}: {assertion.get('details', {})}")
                    
                    if len(failed_cases) > 5:
                        report_lines.append(f"\n   ... and {len(failed_cases) - 5} more failed cases")
            
            report_lines.append("")
        
        report = "\n".join(report_lines)
        
        # Save report if path provided
        if output_path:
            Path(output_path).write_text(report)
            print(f"\n📄 Report saved to: {output_path}")
        
        return report