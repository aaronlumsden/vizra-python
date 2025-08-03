"""
Base evaluation class for testing Vizra agents.
"""

import csv
import json
import asyncio
from typing import Dict, List, Any, Optional, Union, Type
from pathlib import Path
from datetime import datetime
import pandas as pd
from ..agent import BaseAgent
from ..context import AgentContext
from .metrics import BaseMetric


class BaseEvaluation:
    """
    Base class for creating agent evaluations.
    
    Subclass this to create custom evaluations for your agents.
    """
    
    # Class attributes to be overridden in subclasses
    name: str = "base_evaluation"
    description: str = "Base evaluation class"
    agent_name: str = ""  # Name of the agent to evaluate
    csv_path: str = ""    # Path to CSV file with test data
    metrics: List[BaseMetric] = []  # Metrics to evaluate with
    
    def __init__(self):
        """Initialize evaluation."""
        self.agent_class: Optional[type[BaseAgent]] = None
        self._load_agent()
    
    def _load_agent(self) -> None:
        """Load the agent class by name."""
        if not self.agent_name:
            raise ValueError(f"agent_name must be set in {self.__class__.__name__}")
        
        # First try already loaded subclasses
        for subclass in BaseAgent.__subclasses__():
            if subclass.name == self.agent_name:
                self.agent_class = subclass
                return
        
        # If not found, try to discover from agents directory
        import importlib
        import pkgutil
        import inspect
        
        # Try to import from 'agents' module
        try:
            agents_module = importlib.import_module('agents')
            
            # Walk through all modules in the agents package
            if hasattr(agents_module, '__path__'):
                for importer, modname, ispkg in pkgutil.walk_packages(
                    path=agents_module.__path__,
                    prefix=agents_module.__name__ + '.',
                    onerror=lambda x: None
                ):
                    try:
                        module = importlib.import_module(modname)
                        # Check all classes in the module
                        for name, obj in inspect.getmembers(module):
                            if (inspect.isclass(obj) and 
                                issubclass(obj, BaseAgent) and 
                                obj is not BaseAgent and
                                hasattr(obj, 'name') and
                                obj.name == self.agent_name):
                                self.agent_class = obj
                                return
                    except Exception:
                        pass
        except ImportError:
            pass
        
        raise ValueError(f"Agent '{self.agent_name}' not found. Make sure it's defined in the 'agents' directory with name = '{self.agent_name}'")
    
    def prepare_prompt(self, csv_row_data: Dict[str, Any]) -> str:
        """
        Extract the prompt from CSV row data.
        
        Args:
            csv_row_data: Dictionary containing row data from CSV
            
        Returns:
            str: The prompt to send to the agent
        """
        prompt_column = self.get_prompt_csv_column()
        if prompt_column not in csv_row_data:
            raise KeyError(f"Column '{prompt_column}' not found in CSV data")
        return str(csv_row_data[prompt_column])
    
    def evaluate_row(self, csv_row_data: Dict[str, Any], llm_response: str) -> Dict[str, Any]:
        """
        Evaluate a single row's response.
        
        This method automatically runs all metrics. Override in subclasses
        to add custom logic before calling super().evaluate_row().
        
        Args:
            csv_row_data: Dictionary containing row data from CSV
            llm_response: The agent's response
            
        Returns:
            dict: Evaluation results including metric outcomes
        """
        # Run all metrics
        metric_results = {}
        all_passed = True
        
        for metric in self.metrics:
            result = metric.evaluate(csv_row_data, llm_response)
            metric_results[metric.name] = result
            if not result['passed']:
                all_passed = False
        
        return {
            'prompt': self.prepare_prompt(csv_row_data),
            'response': llm_response,
            'agent_response': llm_response,  # Include both keys for compatibility
            'metrics': metric_results,
            'passed': all_passed,
            'row_data': csv_row_data
        }
    
    
    def get_prompt_csv_column(self) -> str:
        """
        Get the CSV column name containing prompts.
        
        Override this in subclasses if using a different column name.
        
        Returns:
            str: Column name for prompts (default: 'prompt')
        """
        return 'prompt'
    
    async def _run_agent_async(self, prompt: str, context: AgentContext) -> str:
        """Run agent asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.agent_class.run, prompt, context)
    
    def run(self, limit: Optional[int] = None) -> Dict[str, Any]:
        """
        Run the evaluation on CSV rows.
        
        Args:
            limit: Optional limit on number of rows to evaluate
            
        Returns:
            dict: Aggregate results with pass/fail counts and details
        """
        if not self.csv_path:
            raise ValueError(f"csv_path must be set in {self.__class__.__name__}")
        
        csv_path = Path(self.csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        # Load CSV data
        try:
            df = pd.read_csv(csv_path)
            rows = df.to_dict('records')
        except Exception as e:
            raise ValueError(f"Failed to load CSV: {e}")
        
        # Apply limit if specified
        total_rows = len(rows)
        if limit and limit > 0:
            rows = rows[:limit]
            print(f"\n⚡ Limiting evaluation to first {limit} of {total_rows} rows")
        
        # Run evaluation on each row
        results = []
        passed_count = 0
        failed_count = 0
        
        print(f"\n🧪 Running evaluation: {self.name}")
        print(f"📊 Testing {len(rows)} cases from {csv_path.name}")
        print("-" * 50)
        
        for i, row_data in enumerate(rows):
            print(f"\n[{i+1}/{len(rows)}] Evaluating...", end='', flush=True)
            
            try:
                # Prepare prompt
                prompt = self.prepare_prompt(row_data)
                
                # Create a context to capture conversation history
                eval_context = AgentContext()
                
                # Run agent with context
                if asyncio.iscoroutinefunction(self.agent_class.run):
                    response = asyncio.run(self._run_agent_async(prompt, eval_context))
                else:
                    response = self.agent_class.run(prompt, eval_context)
                
                # Add conversation history to row_data so metrics can access it
                row_data_with_history = row_data.copy()
                row_data_with_history['conversation_history'] = eval_context.messages
                
                # Evaluate response using metrics
                row_result = self.evaluate_row(row_data_with_history, response)
                
                # Add conversation history to the result
                row_result['conversation_history'] = eval_context.messages
                
                if row_result['passed']:
                    passed_count += 1
                    print(" ✅ PASSED")
                else:
                    failed_count += 1
                    print(" ❌ FAILED")
                    # Show failed metrics
                    if 'metrics' in row_result:
                        for metric_name, metric_result in row_result['metrics'].items():
                            if not metric_result['passed']:
                                print(f"   - {metric_name}: {metric_result['details']}")
                
                results.append(row_result)
                
            except Exception as e:
                print(f" ⚠️  ERROR: {str(e)}")
                failed_count += 1
                results.append({
                    'prompt': row_data.get(self.get_prompt_csv_column(), 'Unknown'),
                    'response': None,
                    'agent_response': None,  # Include both keys for compatibility
                    'error': str(e),
                    'passed': False,
                    'row_data': row_data
                })
        
        # Summary
        total = len(rows)
        success_rate = (passed_count / total * 100) if total > 0 else 0
        
        print("\n" + "=" * 50)
        print(f"📈 Evaluation Summary for '{self.name}':")
        print(f"   Total cases: {total}")
        print(f"   Passed: {passed_count} ({success_rate:.1f}%)")
        print(f"   Failed: {failed_count}")
        print("=" * 50)
        
        # Get model name from agent class
        model_name = 'unknown'
        if self.agent_class:
            model_name = getattr(self.agent_class, 'model', 'unknown')
        
        return {
            'timestamp': datetime.now().isoformat(),
            'evaluation_name': self.name,
            'agent_name': self.agent_name,
            'model': model_name,
            'total_cases': total,
            'passed': passed_count,
            'failed': failed_count,
            'success_rate': success_rate,
            'results': results
        }