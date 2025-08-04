"""
Verifiers provider for Vizra training with GRPO.

This provider owns the entire training loop when used with Vizra.
"""

import os
import json
import re
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd


class VerifiersProvider:
    """
    Provider for Verifiers GRPO integration.
    
    Uses vLLM for inference and Verifiers' GRPO trainer for weight updates.
    This provider owns the entire training loop when run_training is called.
    """
    
    def __init__(self, model_name: str, base_model: str, 
                 inference_base_url: str = "http://localhost:8000/v1", **kwargs):
        """
        Initialize Verifiers provider.
        
        Args:
            model_name: Name for the trainable model
            base_model: Base model to use (e.g., "Qwen/Qwen2.5-0.5B-Instruct")
            inference_base_url: vLLM server URL
            **kwargs: Additional configuration
        """
        self.model_name = model_name
        self.base_model = base_model
        self.inference_base_url = inference_base_url
        self.config = kwargs
        
        # Will be initialized on first use
        self.trainer = None
    
    def run_training(self, training) -> Dict[str, Any]:
        """
        Run the complete training loop using Verifiers.
        
        This method owns the entire training process and returns results
        in Vizra's expected format for CLI display.
        
        Args:
            training: The Vizra training instance with configuration
            
        Returns:
            Dict with training results in Vizra format
        """
        try:
            import verifiers as vf
        except ImportError:
            raise ImportError(
                "Verifiers is required for VerifiersProvider. "
                "Install with: pip install verifiers"
            )
        
        print(f"\n🚀 Starting Verifiers Training: {training.name}")
        print(f"📊 Model: {self.base_model}")
        print(f"🔧 Algorithm: {training.algorithm.upper()}, LR: {training.learning_rate}")
        print("-" * 50)
        
        # Load training data
        csv_path = Path(training.csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        df = pd.read_csv(csv_path)
        data_rows = df.to_dict('records')
        print(f"📊 Loaded {len(data_rows)} training examples from {csv_path.name}")
        
        # Create Verifiers environment
        print("\n🔧 Initializing Verifiers components...")
        env = VizraVerifiersEnv(training, data_rows)
        
        # Set up Verifiers configuration
        os.environ['VF_SERVING_ENGINE'] = 'vllm'
        os.environ['VF_SERVING_ENGINE_KWARGS'] = json.dumps({
            'base_url': self.inference_base_url,
            'api_key': 'dummy'  # vLLM doesn't need a real key
        })
        
        # Initialize trainer with algorithm from training config
        algorithm = training.algorithm.lower()  # Get algorithm from Vizra training class
        print(f"🚀 Initializing {algorithm.upper()} trainer...")
        
        # Prepare algorithm kwargs
        alg_kwargs = {
            'lr': training.learning_rate,
            'batch_size': training.batch_size,
        }
        
        # Add algorithm-specific parameters
        if algorithm == 'grpo':
            alg_kwargs.update({
                'mini_batch_size': min(8, training.batch_size),
                'gradient_accumulation_steps': max(1, training.batch_size // 8),
            })
        elif algorithm == 'ppo':
            alg_kwargs.update({
                'num_ppo_epochs': 4,
                'clip_range': 0.2,
            })
        # Add more algorithms as needed
        
        self.trainer = vf.Trainer(
            model_name_or_path=self.base_model,
            env=env,
            alg=algorithm,  # Use algorithm from training config
            alg_kwargs=alg_kwargs,
            vllm_server=self.inference_base_url,  # Use external vLLM
            track=False,  # Disable wandb tracking for now
        )
        
        print("✅ Verifiers trainer initialized successfully!")
        print(f"📝 Training for {training.n_iterations} iterations...")
        
        # Training history for Vizra
        training_history = []
        best_reward = -float('inf')
        best_iteration = 1
        
        # Let Verifiers handle the training loop
        try:
            # Train using Verifiers
            for iteration in range(1, training.n_iterations + 1):
                print(f"\n[Iteration {iteration}/{training.n_iterations}]")
                
                # Run one training iteration with Verifiers
                metrics = self.trainer.train_step()
                
                # Debug: Show what metrics Verifiers actually returns
                if iteration == 1:
                    print(f"\n📊 DEBUG - Verifiers metrics keys: {list(metrics.keys())[:10]}...")
                
                # Extract metrics for Vizra format - try multiple possible key names
                avg_reward = (
                    metrics.get('rollout/reward/mean', 0.0) or
                    metrics.get('reward_mean', 0.0) or 
                    metrics.get('avg_reward', 0.0) or
                    metrics.get('mean_reward', 0.0) or
                    0.0
                )
                
                # Try to extract other metrics with fallbacks
                vizra_metrics = {
                    'avg_reward': float(avg_reward),
                    'min_reward': float(
                        metrics.get('rollout/reward/min', 0.0) or
                        metrics.get('reward_min', 0.0) or
                        metrics.get('min_reward', 0.0) or
                        0.0
                    ),
                    'max_reward': float(
                        metrics.get('rollout/reward/max', 0.0) or
                        metrics.get('reward_max', 0.0) or
                        metrics.get('max_reward', 0.0) or
                        0.0
                    ),
                    'std_reward': float(
                        metrics.get('rollout/reward/std', 0.0) or
                        metrics.get('reward_std', 0.0) or
                        0.0
                    ),
                    'num_trajectories': int(
                        metrics.get('rollout/num_episodes', 0) or
                        metrics.get('num_episodes', 0) or
                        metrics.get('batch_size', training.batch_size)
                    ),
                    'success_rate': float(
                        metrics.get('rollout/success_rate', 0.0) or
                        metrics.get('success_rate', 0.0) or
                        0.0
                    )
                }
                
                # Display metrics
                print(f"📊 Iteration {iteration} Results:")
                print(f"   Average Reward: {vizra_metrics['avg_reward']:.3f}")
                print(f"   Success Rate: {vizra_metrics['success_rate']:.1%}")
                print(f"   Num Episodes: {vizra_metrics['num_trajectories']}")
                if 'loss/total' in metrics:
                    print(f"   Training Loss: {metrics['loss/total']:.4f}")
                
                # Update best reward
                if vizra_metrics['avg_reward'] > best_reward:
                    best_reward = vizra_metrics['avg_reward']
                    best_iteration = iteration
                    print(f"   🎯 New best average reward!")
                
                # Add to history
                training_history.append({
                    'iteration': iteration,
                    'avg_reward': vizra_metrics['avg_reward'],
                    'metrics': vizra_metrics
                })
                
                # Check early stopping
                if self._should_stop_early(training_history, training):
                    print(f"\n🛑 Early stopping triggered at iteration {iteration}")
                    break
                    
        except Exception as e:
            print(f"\n❌ Error during Verifiers training: {e}")
            print("Falling back to placeholder mode...")
            
            # If Verifiers fails, run placeholder training
            return self._run_placeholder_training(training, data_rows)
        
        # Final summary
        print("\n" + "=" * 50)
        print(f"✅ Training Complete!")
        print(f"🏆 Best Iteration: {best_iteration} (avg reward: {best_reward:.3f})")
        print(f"📈 Final Average Reward: {training_history[-1]['avg_reward']:.3f}")
        print("=" * 50)
        
        # Return results in Vizra format
        return {
            'status': 'completed',
            'iterations_run': len(training_history),
            'best_iteration': best_iteration,
            'best_reward': best_reward,
            'final_metrics': training_history[-1]['metrics'] if training_history else {},
            'training_history': training_history,
            'early_stopped': len(training_history) < training.n_iterations,
            'provider': 'verifiers',
            'training_mode': 'grpo'
        }
    
    def _run_placeholder_training(self, training, data_rows):
        """Fallback placeholder training if Verifiers integration fails."""
        print("\n⚠️  Running in placeholder mode (no weight updates)")
        
        from openai import AsyncOpenAI
        import asyncio
        
        # Training history
        training_history = []
        best_reward = -float('inf')
        best_iteration = 1
        
        # Create simple environment for placeholder
        env = VizraVerifiersEnv(training, data_rows)
        
        # Training loop
        for iteration in range(1, training.n_iterations + 1):
            print(f"\n[Iteration {iteration}/{training.n_iterations}]")
            
            # Sample batch
            if len(data_rows) > training.batch_size:
                batch_indices = np.random.choice(len(data_rows), training.batch_size, replace=False)
                batch_data = [data_rows[i] for i in batch_indices]
            else:
                batch_data = data_rows
            
            # Collect trajectories
            rewards = []
            for i, row_data in enumerate(batch_data):
                print(f"\r[{i+1}/{len(batch_data)}] Collecting trajectories...", end='', flush=True)
                
                # Simple reward calculation
                trajectory_data = training.prepare_trajectory(row_data)
                # Simulate some response
                response = "C Major"  # Placeholder
                reward = training.calculate_reward(row_data, response)
                rewards.append(reward)
            
            print()
            
            # Calculate metrics
            metrics = {
                'avg_reward': float(np.mean(rewards)),
                'min_reward': float(np.min(rewards)),
                'max_reward': float(np.max(rewards)),
                'std_reward': float(np.std(rewards)),
                'num_trajectories': len(rewards),
                'success_rate': sum(1 for r in rewards if r > 0.5) / len(rewards)
            }
            
            # Display metrics
            print(f"📊 Iteration {iteration} Results:")
            print(f"   Average Reward: {metrics['avg_reward']:.3f}")
            print(f"   ⚠️  No weight updates (placeholder mode)")
            
            # Update history
            training_history.append({
                'iteration': iteration,
                'avg_reward': metrics['avg_reward'],
                'metrics': metrics
            })
            
            if metrics['avg_reward'] > best_reward:
                best_reward = metrics['avg_reward']
                best_iteration = iteration
        
        return {
            'status': 'completed',
            'iterations_run': len(training_history),
            'best_iteration': best_iteration,
            'best_reward': best_reward,
            'final_metrics': training_history[-1]['metrics'],
            'training_history': training_history,
            'early_stopped': False,
            'provider': 'verifiers',
            'training_mode': 'placeholder',
            'note': 'Verifiers integration failed - ran in placeholder mode'
        }
    
    def _normalize_tools(self, tools):
        """Convert tools to a consistent dictionary format."""
        if isinstance(tools, dict):
            return tools
        elif isinstance(tools, list):
            # Convert list to dict
            normalized = {}
            for i, tool in enumerate(tools):
                if hasattr(tool, '__name__'):
                    tool_name = tool.__name__
                elif hasattr(tool, '__class__'):
                    tool_name = tool.__class__.__name__
                else:
                    tool_name = f'tool_{i}'
                
                if isinstance(tool, type):
                    try:
                        tool_instance = tool()
                        normalized[tool_name] = tool_instance
                    except Exception as e:
                        print(f"Warning: Could not instantiate tool {tool_name}: {e}")
                        continue
                else:
                    normalized[tool_name] = tool
            return normalized
        else:
            return {}
    
    def _should_stop_early(self, history, training):
        """Check if training should stop early."""
        if len(history) < 5:
            return False
        
        # Stop if reward is very high
        if history[-1]['avg_reward'] > 0.95:
            return True
        
        # Stop if no improvement in last 10 iterations
        if len(history) > 10:
            recent_rewards = [h['avg_reward'] for h in history[-10:]]
            if max(recent_rewards) - min(recent_rewards) < 0.01:
                return True
        
        return False


class VizraVerifiersEnv:
    """
    Verifiers environment that wraps Vizra agent logic.
    
    Implements the minimal interface required by Verifiers.
    """
    
    def __init__(self, training, data_rows):
        """Initialize environment with training configuration."""
        self.training = training
        self.data_rows = data_rows
        self.current_idx = 0
        
        # Get tools from agent
        raw_tools = training.agent_class._get_tools()
        self.tools = self._normalize_tools(raw_tools)
        
        # Get agent instructions
        self.instructions = training.agent_class._get_instructions()
    
    def reset(self):
        """Reset environment and return initial observation."""
        # Get next data row (cycle through data)
        row_data = self.data_rows[self.current_idx % len(self.data_rows)]
        self.current_idx += 1
        
        # Prepare trajectory
        trajectory_data = self.training.prepare_trajectory(row_data)
        prompt = trajectory_data.get('prompt', '')
        
        # Store current row for reward calculation
        self.current_row = row_data
        
        # Initial state
        self.messages = [
            {"role": "system", "content": self.instructions},
            {"role": "user", "content": prompt}
        ]
        self.turn_count = 0
        
        # Return initial observation (user message)
        return prompt
    
    def step(self, action):
        """
        Take a step in the environment.
        
        Args:
            action: The model's response
            
        Returns:
            observation: Next observation (tool result or None)
            reward: Reward for this step
            done: Whether episode is complete
            info: Additional information
        """
        self.turn_count += 1
        
        # Add assistant message
        self.messages.append({
            "role": "assistant",
            "content": action
        })
        
        # Check for tool calls in action
        tool_pattern = r'<(\w+)>(.*?)</\1>'
        tool_matches = re.findall(tool_pattern, action, re.DOTALL)
        
        if tool_matches:
            # Execute tools and get results
            tool_results = []
            for tool_name, tool_input in tool_matches:
                result = self._execute_tool(tool_name, tool_input.strip())
                tool_results.append(f"Tool {tool_name} result: {result}")
                
                # Add tool message
                self.messages.append({
                    "role": "tool",
                    "name": tool_name,
                    "content": str(result)
                })
            
            # Return tool results as observation
            observation = "\n".join(tool_results)
            reward = 0.0  # No reward yet
            done = False
            
        else:
            # No tools called - check if we have a final answer
            observation = ""  # No further observation
            
            # Calculate final reward
            reward = self.training.calculate_reward(self.current_row, action)
            
            # Episode is done if we have a chord answer
            done = any(word in action for word in ['Major', 'Minor', 'Diminished', 'Augmented'])
            
            # Also done if max turns reached
            if self.turn_count >= 10:
                done = True
        
        info = {
            'turn_count': self.turn_count,
            'messages': self.messages
        }
        
        return observation, reward, done, info
    
    def _execute_tool(self, tool_name, tool_input):
        """Execute a tool and return its result."""
        # Find tool by xml_tag or name
        tool = None
        for t_name, t_instance in self.tools.items():
            if (hasattr(t_instance, 'xml_tag') and t_instance.xml_tag == tool_name) or t_name == tool_name:
                tool = t_instance
                break
        
        if tool:
            # Execute tool
            if hasattr(tool, 'execute'):
                return tool.execute({'notes_str': tool_input})
            elif hasattr(tool, 'run'):
                return tool.run(tool_input)
            elif hasattr(tool, '__call__'):
                return tool(tool_input)
            else:
                return f"Tool {tool_name} has no execute method"
        else:
            return f"Tool {tool_name} not found"
    
    def _normalize_tools(self, tools):
        """Convert tools to dictionary format."""
        if isinstance(tools, dict):
            return tools
        elif isinstance(tools, list):
            normalized = {}
            for tool in tools:
                if hasattr(tool, '__name__'):
                    tool_name = tool.__name__
                elif hasattr(tool, '__class__'):
                    tool_name = tool.__class__.__name__
                else:
                    continue
                
                if isinstance(tool, type):
                    try:
                        normalized[tool_name] = tool()
                    except:
                        continue
                else:
                    normalized[tool_name] = tool
            return normalized
        else:
            return {}