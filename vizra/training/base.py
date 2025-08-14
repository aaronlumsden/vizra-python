"""
Base reinforcement learning training class for Vizra agents.
"""

import csv
import json
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
from ..agent import BaseAgent
from ..context import AgentContext
from ..config import config


class BaseRLTraining:
    """
    Base class for reinforcement learning training of agents.
    
    Subclass this to create custom training routines for your agents.
    """
    
    # Class attributes to be overridden in subclasses
    name: str = "base_training"
    description: str = "Base RL training class"
    agent_name: str = ""  # Name of the agent to train
    csv_path: str = ""    # Path to CSV file with training data
    
    # Training hyperparameters
    algorithm: str = "ppo"  # Options: ppo, dpo, reinforce, etc.
    learning_rate: float = 1e-4
    batch_size: int = 32
    n_iterations: int = 100
    
    # Optional provider for custom trajectory collection and training
    provider: Optional[Any] = None
    
    def __init__(self):
        """Initialize training."""
        # Override class attributes with config values if not already set by subclass
        if self.algorithm == "ppo":  # Default value, so check config
            self.algorithm = config('training.algorithm', 'ppo')
        if self.learning_rate == 1e-4:  # Default value, so check config
            self.learning_rate = config('training.learning_rate', 1e-4)
        if self.batch_size == 32:  # Default value, so check config
            self.batch_size = config('training.batch_size', 32)
        if self.n_iterations == 100:  # Default value, so check config
            self.n_iterations = config('training.n_iterations', 100)
        
        self.agent_class: Optional[type[BaseAgent]] = None
        self.trajectories: List[Dict[str, Any]] = []
        self.training_history: List[Dict[str, Any]] = []
        self.current_iteration = 0
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
    
    def prepare_trajectory(self, csv_row_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare a training trajectory from CSV row data.
        
        Override this method to customize how training data is prepared.
        
        Args:
            csv_row_data: Dictionary containing row data from CSV
            
        Returns:
            dict: Trajectory data including prompt, context, etc.
        """
        return {
            'prompt': csv_row_data.get('prompt', ''),
            'context': csv_row_data,
            'metadata': {
                'row_index': csv_row_data.get('_index', 0),
                'timestamp': datetime.now().isoformat()
            }
        }
    
    def calculate_reward(self, csv_row_data: Dict[str, Any], agent_response: str) -> float:
        """
        Calculate reward for an agent response.
        
        Override this method to implement custom reward functions.
        
        Args:
            csv_row_data: Dictionary containing row data from CSV
            agent_response: The agent's response
            
        Returns:
            float: Reward value between 0.0 and 1.0
        """
        # Default implementation: check if response contains expected text
        expected = csv_row_data.get('expected_response', '')
        if expected and expected.lower() in agent_response.lower():
            return 1.0
        
        # Partial credit for response length
        if len(agent_response) > 50:
            return 0.5
        
        return 0.0
    
    def should_stop_early(self, metrics: Dict[str, Any]) -> bool:
        """
        Determine if training should stop early.
        
        Override this method to implement custom early stopping logic.
        
        Args:
            metrics: Current training metrics
            
        Returns:
            bool: True if training should stop
        """
        # Default: stop if average reward is very high
        avg_reward = metrics.get('avg_reward', 0)
        if avg_reward > 0.95:
            return True
        
        # Stop if reward hasn't improved in last 10 iterations
        if len(self.training_history) > 10:
            recent_rewards = [h['avg_reward'] for h in self.training_history[-10:]]
            if max(recent_rewards) - min(recent_rewards) < 0.01:
                return True
        
        return False
    
    async def _run_agent_async(self, prompt: str, context: Optional[AgentContext] = None) -> str:
        """Run agent asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.agent_class.run, prompt, context)
    
    def collect_trajectories(self, data_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Collect trajectories by running agent on training data.
        
        Args:
            data_rows: List of training data rows
            
        Returns:
            List of trajectory dictionaries
        """
        # If provider handles trajectory collection, delegate to it
        if self.provider and hasattr(self.provider, 'collect_trajectories'):
            # Pass self to give provider access to agent, metrics, etc.
            return self.provider.collect_trajectories(
                training=self,
                data_rows=data_rows,
                agent_class=self.agent_class
            )
        
        # Otherwise use default Vizra behavior
        trajectories = []
        
        for i, row_data in enumerate(data_rows):
            print(f"\r[{i+1}/{len(data_rows)}] Collecting trajectories...", end='', flush=True)
            
            try:
                # Prepare trajectory
                trajectory = self.prepare_trajectory(row_data)
                
                # Run agent
                prompt = trajectory['prompt']
                if asyncio.iscoroutinefunction(self.agent_class.run):
                    response = asyncio.run(self._run_agent_async(prompt))
                else:
                    response = self.agent_class.run(prompt)
                
                # Calculate reward
                reward = self.calculate_reward(row_data, response)
                
                # Store complete trajectory
                trajectory.update({
                    'response': response,
                    'reward': reward,
                    'row_data': row_data
                })
                
                trajectories.append(trajectory)
                
            except Exception as e:
                print(f"\n⚠️  Error collecting trajectory {i+1}: {e}")
                trajectories.append({
                    'error': str(e),
                    'reward': 0.0,
                    'row_data': row_data
                })
        
        print()  # New line after progress
        return trajectories
    
    def compute_metrics(self, trajectories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute training metrics from trajectories.
        
        Args:
            trajectories: List of collected trajectories
            
        Returns:
            dict: Training metrics
        """
        rewards = [t['reward'] for t in trajectories if 'reward' in t]
        
        if not rewards:
            return {
                'avg_reward': 0.0,
                'min_reward': 0.0,
                'max_reward': 0.0,
                'std_reward': 0.0,
                'num_trajectories': 0
            }
        
        return {
            'avg_reward': np.mean(rewards),
            'min_reward': np.min(rewards),
            'max_reward': np.max(rewards),
            'std_reward': np.std(rewards),
            'num_trajectories': len(rewards),
            'success_rate': sum(1 for r in rewards if r > 0.5) / len(rewards)
        }
    
    def train_step(self, trajectories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Perform one training step.
        
        In a real implementation, this would update model parameters.
        For now, this is a stub that just computes metrics.
        
        Args:
            trajectories: Batch of trajectories to train on
            
        Returns:
            dict: Training step results
        """
        # If provider handles training, delegate to it
        if self.provider and hasattr(self.provider, 'train_step'):
            return self.provider.train_step(
                training=self,
                trajectories=trajectories
            )
        
        # Otherwise use default behavior
        # Compute metrics
        metrics = self.compute_metrics(trajectories)
        
        # In a real implementation, you would:
        # 1. Compute policy gradients or value estimates
        # 2. Update model parameters
        # 3. Track loss values
        
        # For now, just return metrics
        return {
            'iteration': self.current_iteration,
            'metrics': metrics,
            'algorithm': self.algorithm,
            'learning_rate': self.learning_rate
        }
    
    def run(self) -> Dict[str, Any]:
        """
        Run the RL training loop.
        
        Returns:
            dict: Training results and metrics
        """
        # Delegate entirely to provider if it supports full training ownership
        if self.provider and hasattr(self.provider, 'run_training'):
            return self.provider.run_training(self)
        
        # Otherwise continue with default Vizra training logic
        if not self.csv_path:
            raise ValueError(f"csv_path must be set in {self.__class__.__name__}")
        
        csv_path = Path(self.csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        # Load training data
        try:
            df = pd.read_csv(csv_path)
            data_rows = df.to_dict('records')
            # Add index to each row
            for i, row in enumerate(data_rows):
                row['_index'] = i
        except Exception as e:
            raise ValueError(f"Failed to load CSV: {e}")
        
        print(f"\n🎯 Starting RL Training: {self.name}")
        print(f"📊 Training on {len(data_rows)} examples from {csv_path.name}")
        print(f"🔧 Algorithm: {self.algorithm}, LR: {self.learning_rate}")
        print("-" * 50)
        
        # Training loop
        best_reward = -float('inf')
        
        for iteration in range(self.n_iterations):
            self.current_iteration = iteration + 1
            print(f"\n[Iteration {self.current_iteration}/{self.n_iterations}]")
            
            # Sample batch
            if len(data_rows) > self.batch_size:
                batch_indices = np.random.choice(len(data_rows), self.batch_size, replace=False)
                batch_data = [data_rows[i] for i in batch_indices]
            else:
                batch_data = data_rows
            
            # Collect trajectories
            trajectories = self.collect_trajectories(batch_data)
            
            # Train step
            step_result = self.train_step(trajectories)
            metrics = step_result['metrics']
            
            # Update history
            self.training_history.append({
                'iteration': self.current_iteration,
                'avg_reward': metrics['avg_reward'],
                'metrics': metrics
            })
            
            # Print metrics
            print(f"  Avg Reward: {metrics['avg_reward']:.3f}")
            print(f"  Success Rate: {metrics['success_rate']:.1%}")
            print(f"  Reward Range: [{metrics['min_reward']:.3f}, {metrics['max_reward']:.3f}]")
            
            # Track best performance
            if metrics['avg_reward'] > best_reward:
                best_reward = metrics['avg_reward']
                print(f"  🎉 New best average reward: {best_reward:.3f}")
            
            # Check early stopping
            if self.should_stop_early(metrics):
                print(f"\n✅ Early stopping at iteration {self.current_iteration}")
                break
        
        # Final summary
        print("\n" + "=" * 50)
        print(f"📈 Training Summary for '{self.name}':")
        print(f"   Total iterations: {self.current_iteration}")
        print(f"   Best average reward: {best_reward:.3f}")
        
        if self.training_history:
            final_metrics = self.training_history[-1]['metrics']
            print(f"   Final average reward: {final_metrics['avg_reward']:.3f}")
            print(f"   Final success rate: {final_metrics['success_rate']:.1%}")
        
        print("=" * 50)
        
        return {
            'training_name': self.name,
            'agent_name': self.agent_name,
            'total_iterations': self.current_iteration,
            'best_reward': best_reward,
            'final_metrics': self.training_history[-1]['metrics'] if self.training_history else {},
            'training_history': self.training_history,
            'hyperparameters': {
                'algorithm': self.algorithm,
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size,
                'n_iterations': self.n_iterations
            }
        }