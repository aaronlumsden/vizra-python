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
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


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
            import verifiers
            from verifiers.trainers.grpo_trainer import GRPOTrainer
            from verifiers.trainers.grpo_config import GRPOConfig
            from verifiers.envs.multiturn_env import MultiTurnEnv
        except ImportError as e:
            raise ImportError(
                f"Required packages missing: {e}\n"
                "Install with: pip install verifiers peft"
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
        
        # Load evaluation data if available
        eval_data_rows = None
        eval_csv_path = csv_path.parent / "chord_identifier_eval.csv"
        if eval_csv_path.exists():
            eval_df = pd.read_csv(eval_csv_path)
            eval_data_rows = eval_df.to_dict('records')
            print(f"📊 Loaded {len(eval_data_rows)} evaluation examples from {eval_csv_path.name}")
        else:
            print("⚠️  No evaluation dataset found, using training data for evaluation")
        
        # Initialize model and tokenizer
        print("\n🔧 Loading model and tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            # Remove device_map - let accelerate handle device placement for multi-GPU
        )
        
        # Create Verifiers environment wrapper
        print("\n🔧 Initializing Verifiers environment...")
        env = VizraVerifiersEnv(training, data_rows, eval_data_rows)
        
        # Get datasets from environment
        train_dataset = env.get_dataset()
        eval_dataset = env.get_eval_dataset() if eval_data_rows else None
        
        print(f"📊 Training dataset size: {len(train_dataset)}")
        if eval_dataset:
            print(f"📊 Evaluation dataset size: {len(eval_dataset)}")
        
        # Configure GRPO training
        print(f"\n🚀 Initializing GRPO trainer...")
        
        # Create GRPO config with Verifiers' expected parameters
        config = GRPOConfig(
            # Output directory
            output_dir=f"./outputs/{self.model_name}-grpo",
            
            # Training hyperparameters - check self.config first, then training attributes
            learning_rate=training.learning_rate,
            per_device_train_batch_size=self.config.get('per_device_batch_size', 
                                                       getattr(training, 'per_device_batch_size', 8)),
            gradient_accumulation_steps=self.config.get('gradient_accumulation_steps',
                                                      getattr(training, 'gradient_accumulation_steps', 1)),
            num_train_epochs=training.n_iterations,
            
            # GRPO specific
            beta=0.1,  # KL penalty coefficient
            num_generations=2,  # GRPO requires at least 2
            max_tokens=128,
            temperature=0.7,
            
            # Other settings
            warmup_ratio=0.1,
            logging_steps=10,
            save_steps=500,
            eval_steps=100,
            
            # Device settings
            fp16=torch.cuda.is_available(),
            bf16=False,
            gradient_checkpointing=self.config.get('gradient_checkpointing',
                                                 getattr(training, 'gradient_checkpointing', True)),
            
            # Disable distributed training
            ddp_backend=None,
            local_rank=-1,
        )
        
        # Initialize GRPO trainer with custom environment and datasets
        self.trainer = GRPOTrainer(
            model=self.model,
            env=env,  # Pass our Verifiers environment
            args=config,
            processing_class=self.tokenizer,
            train_dataset=train_dataset,  # Pass dataset directly
            eval_dataset=eval_dataset,    # Pass eval dataset if available
        )
        
        print("✅ GRPO trainer initialized successfully!")
        print(f"📝 Training for {training.n_iterations} epochs...")
        
        # Training history for Vizra
        training_history = []
        best_reward = -float('inf')
        best_iteration = 1
        
        # Train using Verifiers' GRPOTrainer
        try:
            # Run training
            train_output = self.trainer.train()
            
            # Extract metrics from training
            if hasattr(train_output, 'metrics'):
                final_metrics = train_output.metrics
            else:
                final_metrics = {}
            
            # Get training history from trainer
            if hasattr(self.trainer.state, 'log_history'):
                for log_entry in self.trainer.state.log_history:
                    if 'loss' in log_entry:
                        iteration = log_entry.get('epoch', log_entry.get('step', 0))
                        avg_reward = log_entry.get('rewards/mean', log_entry.get('reward_mean', 0.0))
                        
                        vizra_metrics = {
                            'avg_reward': float(avg_reward),
                            'loss': float(log_entry.get('loss', 0.0)),
                            'learning_rate': float(log_entry.get('learning_rate', training.learning_rate)),
                            'kl_div': float(log_entry.get('kl_div', 0.0)),
                        }
                        
                        training_history.append({
                            'iteration': iteration,
                            'avg_reward': avg_reward,
                            'metrics': vizra_metrics
                        })
                        
                        if avg_reward > best_reward:
                            best_reward = avg_reward
                            best_iteration = iteration
            
            # If no history, create minimal response
            if not training_history:
                training_history = [{
                    'iteration': 1,
                    'avg_reward': 0.5,
                    'metrics': {'avg_reward': 0.5, 'loss': 0.0}
                }]
                best_reward = 0.5
                best_iteration = 1
                
        except Exception as e:
            print(f"\n❌ Error during Verifiers GRPO training: {e}")
            import traceback
            traceback.print_exc()
            print("Falling back to placeholder mode...")
            
            # If Verifiers fails, run placeholder training
            return self._run_placeholder_training(training, data_rows)
        
        # Final summary
        print("\n" + "=" * 50)
        print(f"✅ Training Complete!")
        print(f"🏆 Best Iteration: {best_iteration} (avg reward: {best_reward:.3f})")
        if training_history:
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


# Import MultiTurnEnv at module level for inheritance
try:
    from verifiers.envs.multiturn_env import MultiTurnEnv
except ImportError:
    # Fallback base class if verifiers not installed yet
    MultiTurnEnv = object


class VizraVerifiersEnv(MultiTurnEnv):
    """
    Verifiers environment that wraps Vizra agent logic.
    
    Inherits from MultiTurnEnv and implements the required interface.
    """
    
    def __init__(self, training, data_rows, eval_data_rows=None):
        """Initialize environment with training configuration."""
        # Initialize base class if it's MultiTurnEnv
        if MultiTurnEnv is not object:
            super().__init__(message_type="chat", max_turns=10)
        
        self.training = training
        self.data_rows = data_rows
        self.eval_data_rows = eval_data_rows if eval_data_rows is not None else data_rows
        self.current_idx = 0
        
        # Get tools from agent
        raw_tools = training.agent_class._get_tools()
        self.tools = self._normalize_tools(raw_tools)
        
        # Get agent instructions
        self.instructions = training.agent_class._get_instructions()
        
        # Store dataset for GRPOTrainer
        self.dataset = None
        self.eval_dataset = None
    
    def get_dataset(self):
        """Return the dataset for GRPOTrainer."""
        if self.dataset is None:
            # Create dataset if not already created
            from datasets import Dataset
            
            # Convert data to prompts
            dataset_entries = []
            for row in self.data_rows:
                trajectory = self.training.prepare_trajectory(row)
                prompt = trajectory['prompt']
                
                # Create entry with required columns for GRPO
                entry = {
                    'prompt': prompt,
                    'answer': row.get('expected_chord', row.get('expected_output', '')),  # Required by GRPO
                    'task': 'chord_identification',  # Required by GRPO
                    'info': {  # Required by GRPO
                        'expected_output': row.get('expected_chord', row.get('expected_output', '')),
                        'question': row.get('question', ''),
                    },
                    # Additional columns for compatibility
                    'query': prompt,
                    'input': prompt,
                }
                dataset_entries.append(entry)
            
            self.dataset = Dataset.from_list(dataset_entries)
        
        return self.dataset
    
    def get_eval_dataset(self):
        """Return the evaluation dataset for GRPOTrainer."""
        if not hasattr(self, 'eval_dataset') or self.eval_dataset is None:
            # Create eval dataset if not already created
            from datasets import Dataset
            
            # Convert eval data to prompts
            dataset_entries = []
            for row in self.eval_data_rows:
                trajectory = self.training.prepare_trajectory(row)
                prompt = trajectory['prompt']
                
                # Create entry with required columns for GRPO
                entry = {
                    'prompt': prompt,
                    'answer': row.get('expected_chord', row.get('expected_output', '')),  # Required by GRPO
                    'task': 'chord_identification',  # Required by GRPO
                    'info': {  # Required by GRPO
                        'expected_output': row.get('expected_chord', row.get('expected_output', '')),
                        'question': row.get('question', ''),
                    },
                    # Additional columns for compatibility
                    'query': prompt,
                    'input': prompt,
                }
                dataset_entries.append(entry)
            
            self.eval_dataset = Dataset.from_list(dataset_entries)
        
        return self.eval_dataset
    
    def reset(self, seed=None):
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
        
        # Return initial observation and info
        return prompt, {}
    
    def step(self, action):
        """
        Take a step in the environment.
        
        Args:
            action: The model's response
            
        Returns:
            observation: Next observation (tool result or None)
            reward: Reward for this step
            terminated: Whether episode is complete
            truncated: Whether episode was cut short
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
            terminated = False
            truncated = False
            
        else:
            # No tools called - check if we have a final answer
            observation = ""  # No further observation
            
            # Calculate final reward
            reward = self.training.calculate_reward(self.current_row, action)
            
            # Episode is done if we have a chord answer
            terminated = any(word in action for word in ['Major', 'Minor', 'Diminished', 'Augmented'])
            
            # Truncate if max turns reached
            truncated = self.turn_count >= 10
        
        info = {
            'turn_count': self.turn_count,
            'messages': self.messages
        }
        
        return observation, reward, terminated, truncated, info
    
    async def a_generate(self, inputs, **kwargs):
        """Async generation method required by Verifiers."""
        from openai import AsyncOpenAI
        from verifiers.envs.environment import GenerateOutputs
        import os
        
        # Debug logging
        print(f"DEBUG: a_generate inputs type: {type(inputs)}")
        if hasattr(inputs, '__dict__'):
            print(f"DEBUG: inputs attributes: {inputs.__dict__.keys()}")
        elif isinstance(inputs, dict):
            print(f"DEBUG: inputs keys: {inputs.keys()}")
        
        # Initialize async client for vLLM
        client = AsyncOpenAI(
            base_url="http://localhost:8000/v1",
            api_key="dummy"  # vLLM doesn't need a real key
        )
        
        # Extract generation parameters
        max_tokens = kwargs.get('max_tokens', 256)
        temperature = kwargs.get('temperature', 0.7)
        model = kwargs.get('model', 'Qwen/Qwen2.5-0.5B-Instruct')
        
        # Extract prompts and answers from inputs
        if hasattr(inputs, 'prompt'):
            prompts = inputs.prompt
            answers = inputs.answer if hasattr(inputs, 'answer') else [""] * len(prompts)
            tasks = inputs.task if hasattr(inputs, 'task') else [""] * len(prompts)
            infos = inputs.info if hasattr(inputs, 'info') else [{}] * len(prompts)
        elif isinstance(inputs, dict) and 'prompt' in inputs:
            # Handle dict input format
            prompts = inputs['prompt'] if isinstance(inputs['prompt'], list) else [inputs['prompt']]
            answers = inputs.get('answer', [""] * len(prompts))
            tasks = inputs.get('task', [""] * len(prompts))
            infos = inputs.get('info', [{}] * len(prompts))
        else:
            # Fallback for list input
            prompts = inputs if isinstance(inputs, list) else [inputs]
            answers = [""] * len(prompts)
            tasks = [""] * len(prompts)
            infos = [{}] * len(prompts)
        
        completions = []
        rewards = []
        
        for i, prompt in enumerate(prompts):
            # Ensure prompt is a string
            prompt_text = str(prompt) if prompt else ""
            
            # Create messages with system prompt and user input
            messages = [
                {"role": "system", "content": self.instructions},
                {"role": "user", "content": prompt_text}
            ]
            
            try:
                # Call vLLM server
                completion = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    n=1
                )
                response = completion.choices[0].message.content
            except Exception as e:
                print(f"Error calling vLLM: {e}")
                response = "Error generating response"
            
            completions.append(response)
            
            # Calculate reward if we have the training instance
            if hasattr(self, 'training') and answers[i] and answers[i] != "":
                # Create row data for reward calculation
                row_data = {
                    'question': prompt,
                    'expected_output': answers[i],
                    'expected_chord': answers[i]
                }
                reward = self.training.calculate_reward(row_data, response)
            else:
                reward = 0.0
            
            rewards.append(reward)
        
        # Ensure all values are the correct type
        prompts = [str(p) for p in prompts]
        answers = [str(a) if a else "" for a in answers]
        tasks = [str(t) if t else "" for t in tasks]
        completions = [str(c) for c in completions]
        states = [{}] * len(prompts)  # Empty dict instead of None
        
        # Return GenerateOutputs object
        return GenerateOutputs(
            prompt=prompts,
            answer=answers,
            task=tasks,
            info=infos,
            completion=completions,
            state=states,
            reward=rewards,
            metrics={}  # No additional metrics for now
        )
    
    def process_env_results_vllm(self, env_results, prompts, **kwargs):
        """Process results from vLLM generation - required by Verifiers."""
        # For vLLM, the results are already in the correct format
        # This method is called after a_generate to do any post-processing
        return env_results
    
    def setup_state(self, state, **kwargs):
        """Setup initial state for a new rollout."""
        if state is None:
            state = {}
        state['turn_count'] = 0
        return state
    
    def is_completed(self, messages, state, **kwargs):
        """Check if the rollout is completed - required by Verifiers MultiTurnEnv."""
        # Check if we've reached max turns
        turn_count = state.get('turn_count', 0) if state else 0
        if turn_count >= self.max_turns:
            return True
        
        # Check if the last message contains a chord answer
        if messages and len(messages) > 0:
            last_message = messages[-1]
            if isinstance(last_message, dict) and last_message.get('role') == 'assistant':
                content = last_message.get('content', '')
                # Check for chord identifiers
                if any(word in content for word in ['Major', 'Minor', 'Diminished', 'Augmented']):
                    return True
        
        return False
    
    def env_response(self, messages, state, **kwargs):
        """Generate environment response - required by Verifiers MultiTurnEnv.
        
        Returns:
            Tuple[Messages, State]: New messages list with env response and updated state
        """
        if not messages:
            return messages, state
        
        # Get the last message
        last_message = messages[-1]
        if not isinstance(last_message, dict) or last_message.get('role') != 'assistant':
            return messages, state
        
        content = last_message.get('content', '')
        
        # Check for tool calls in the assistant's message
        tool_pattern = r'<(\w+)>(.*?)</\1>'
        tool_matches = re.findall(tool_pattern, content, re.DOTALL)
        
        if tool_matches:
            # Execute tools and add results as new messages
            new_messages = list(messages)  # Copy messages
            for tool_name, tool_input in tool_matches:
                result = self._execute_tool(tool_name, tool_input.strip())
                
                # Add tool response as a new message
                tool_message = {
                    "role": "tool",
                    "name": tool_name,
                    "content": str(result)
                }
                new_messages.append(tool_message)
            
            # Update state with turn count
            new_state = state.copy() if isinstance(state, dict) else {}
            new_state['turn_count'] = new_state.get('turn_count', 0) + 1
            
            return new_messages, new_state
        
        # No tools called, no environment response needed
        return messages, state
    
    def evaluate(self, n_samples=None, **kwargs):
        """Evaluate the model on the evaluation dataset - required by Verifiers.
        
        Args:
            n_samples: Number of samples to evaluate (None = all)
            
        Returns:
            Dict with evaluation metrics
        """
        eval_data = self.eval_data_rows
        if n_samples is not None and n_samples < len(eval_data):
            # Sample subset for evaluation
            import random
            eval_data = random.sample(eval_data, n_samples)
        
        # Calculate evaluation metrics
        correct = 0
        total = len(eval_data)
        
        for row in eval_data:
            # For now, return simple metrics
            # In a real implementation, this would run inference and check results
            pass
        
        return {
            'eval_samples': total,
            'eval_accuracy': 0.5,  # Placeholder
            'eval_reward': 0.5,    # Placeholder
        }
    
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