"""
Verifiers provider for Vizra training with GRPO.
"""

import asyncio
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
import numpy as np
from openai import AsyncOpenAI


class VerifiersProvider:
    """
    Provider for Verifiers GRPO integration.
    
    Uses vLLM for inference and Verifiers' GRPO trainer for weight updates.
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
        try:
            import verifiers as vf
            from verifiers import GRPOTrainer, MultiTurnEnv, tools
        except ImportError:
            raise ImportError(
                "Verifiers is required for VerifiersProvider. "
                "Install with: pip install verifiers"
            )
        
        self.model_name = model_name
        self.base_model = base_model
        self.inference_base_url = inference_base_url
        self.config = kwargs
        
        # Store imports for later use
        self.vf = vf
        self.GRPOTrainer = GRPOTrainer
        self.MultiTurnEnv = MultiTurnEnv
        self.tools = tools
        
        # Will be initialized on first use
        self.trainer = None
        self.client = None
        self.env = None
    
    def _normalize_tools(self, tools):
        """Convert tools to a consistent dictionary format."""
        if isinstance(tools, dict):
            return tools
        elif isinstance(tools, list):
            # Convert list to dict - handle both classes and instances
            normalized = {}
            for i, tool in enumerate(tools):
                # Get tool name
                if hasattr(tool, '__name__'):
                    tool_name = tool.__name__
                elif hasattr(tool, '__class__'):
                    tool_name = tool.__class__.__name__
                else:
                    tool_name = f'tool_{i}'
                
                # If it's a class, instantiate it
                if isinstance(tool, type):
                    try:
                        tool_instance = tool()
                        normalized[tool_name] = tool_instance
                    except Exception as e:
                        print(f"Warning: Could not instantiate tool {tool_name}: {e}")
                        continue
                else:
                    # It's already an instance
                    normalized[tool_name] = tool
            return normalized
        else:
            return {}
    
    def collect_trajectories(self, training, data_rows: List[Dict[str, Any]], 
                           agent_class) -> List[Dict[str, Any]]:
        """
        Collect trajectories using vLLM inference.
        
        For now, we'll use a similar approach to ARTProvider,
        collecting trajectories via OpenAI-compatible API.
        Later this can be enhanced to use Verifiers' MultiTurnEnv.
        """
        # Run async collection in sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        trajectories = loop.run_until_complete(
            self._collect_trajectories_async(training, data_rows, agent_class)
        )
        
        return trajectories
    
    async def _collect_trajectories_async(self, training, data_rows: List[Dict[str, Any]], 
                                        agent_class) -> List[Dict[str, Any]]:
        """Async implementation of trajectory collection."""
        trajectories = []
        
        # Initialize client if needed
        if not self.client:
            self.client = AsyncOpenAI(
                api_key="dummy",  # vLLM doesn't need a real key
                base_url=self.inference_base_url
            )
        
        # Get agent configuration
        instructions = agent_class._get_instructions()
        raw_tools = agent_class._get_tools()
        tools = self._normalize_tools(raw_tools)
        
        for i, row_data in enumerate(data_rows):
            print(f"\r[{i+1}/{len(data_rows)}] Collecting trajectories...", end='', flush=True)
            
            try:
                # Prepare the trajectory data
                trajectory_data = training.prepare_trajectory(row_data)
                prompt = trajectory_data.get('prompt', '')
                
                # Format tools for OpenAI/vLLM
                formatted_tools = []
                for tool_name, tool_func in tools.items():
                    # Use the tool's xml_tag if available
                    function_name = getattr(tool_func, 'xml_tag', tool_name)
                    formatted_tools.append({
                        "type": "function",
                        "function": {
                            "name": function_name,
                            "description": getattr(tool_func, '__doc__', f"Tool: {function_name}"),
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "input": {"type": "string", "description": "Input for the tool"}
                                },
                                "required": ["input"]
                            }
                        }
                    })
                
                # Create messages
                messages = [
                    {"role": "system", "content": instructions},
                    {"role": "user", "content": prompt}
                ]
                
                # Call vLLM with tools
                response = await self.client.chat.completions.create(
                    model=self.base_model,
                    messages=messages,
                    tools=formatted_tools if formatted_tools else None,
                    max_tokens=500,
                    temperature=0.7
                )
                
                # Process response
                agent_response = self._process_response(response, tools)
                full_response_for_metrics = self._format_response_for_metrics(response, tools)
                
                # Calculate reward
                reward = training.calculate_reward(row_data, full_response_for_metrics)
                
                # Create trajectory
                trajectory = {
                    'prompt': prompt,
                    'response': agent_response,
                    'reward': reward,
                    'metadata': {
                        'verifiers_messages': messages,
                        'timestamp': datetime.now().isoformat()
                    }
                }
                
                trajectories.append(trajectory)
                
            except Exception as e:
                print(f"\n⚠️  Error collecting trajectory {i+1}: {e}")
                continue
        
        print()  # New line after progress
        return trajectories
    
    def _process_response(self, response, tools) -> str:
        """Process response to extract final answer, handling tool calls."""
        if not response.choices:
            return ""
        
        choice = response.choices[0]
        
        # If there are tool calls, execute them
        if choice.message.tool_calls:
            tool_outputs = []
            for tool_call in choice.message.tool_calls:
                called_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)
                tool_input = tool_args.get('input', '')
                
                # Find and execute tool
                tool_obj = None
                for tool_class_name, tool_instance in tools.items():
                    if (hasattr(tool_instance, 'xml_tag') and tool_instance.xml_tag == called_name) or tool_class_name == called_name:
                        tool_obj = tool_instance
                        break
                
                if tool_obj:
                    if hasattr(tool_obj, 'execute'):
                        tool_result = tool_obj.execute({'notes_str': tool_input})
                    elif hasattr(tool_obj, '__call__'):
                        tool_result = tool_obj(tool_input)
                    elif hasattr(tool_obj, 'run'):
                        tool_result = tool_obj.run(tool_input)
                    else:
                        tool_result = f"Tool {called_name} has no execute method"
                else:
                    tool_result = f"Tool {called_name} not found"
                    
                tool_outputs.append(str(tool_result))
            
            return "\n".join(tool_outputs)
        
        # Otherwise return the direct response
        return choice.message.content or ""
    
    def _format_response_for_metrics(self, response, tools) -> str:
        """Format response for metrics that need to see tool usage patterns."""
        if not response.choices:
            return ""
        
        choice = response.choices[0]
        
        # If there are tool calls, format them as XML for tool usage metric
        if choice.message.tool_calls:
            tool_outputs = []
            for tool_call in choice.message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)
                tool_input = tool_args.get('input', '')
                
                # Find and execute tool
                tool_obj = None
                for tool_class_name, tool_instance in tools.items():
                    if (hasattr(tool_instance, 'xml_tag') and tool_instance.xml_tag == tool_name) or tool_class_name == tool_name:
                        tool_obj = tool_instance
                        break
                
                if tool_obj:
                    if hasattr(tool_obj, 'execute'):
                        tool_result = tool_obj.execute({'notes_str': tool_input})
                    elif hasattr(tool_obj, '__call__'):
                        tool_result = tool_obj(tool_input)
                    elif hasattr(tool_obj, 'run'):
                        tool_result = tool_obj.run(tool_input)
                    else:
                        tool_result = f"Tool {tool_name} has no execute method"
                else:
                    tool_result = f"Tool {tool_name} not found"
                    
                # Format as XML for tool usage detection
                tool_outputs.append(f"<{tool_name}>{tool_input}</{tool_name}>\n{tool_result}")
            
            return "\n".join(tool_outputs)
        
        return choice.message.content or ""
    
    def train_step(self, training, trajectories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Execute a training step using Verifiers GRPO.
        
        Note: This is a placeholder implementation. Full Verifiers integration
        would require setting up the GRPO trainer with appropriate configuration.
        """
        if not trajectories:
            return {'status': 'skipped', 'reason': 'No trajectories to train on'}
        
        # Extract rewards for metrics
        rewards = [traj['reward'] for traj in trajectories if 'reward' in traj]
        if not rewards:
            rewards = [0.0]
        
        # Calculate metrics
        metrics = {
            'avg_reward': float(np.mean(rewards)),
            'min_reward': float(np.min(rewards)),
            'max_reward': float(np.max(rewards)),
            'std_reward': float(np.std(rewards)),
            'num_trajectories': len(trajectories),
            'success_rate': sum(1 for r in rewards if r > 0.5) / len(rewards) if rewards else 0.0
        }
        
        # Show reward distribution
        print(f"\n📊 Reward distribution for this batch:")
        print(f"   Average: {metrics['avg_reward']:.3f}")
        print(f"   Min: {metrics['min_reward']:.3f}, Max: {metrics['max_reward']:.3f}")
        
        # TODO: Implement actual Verifiers GRPO training
        # This would involve:
        # 1. Initialize GRPOTrainer if not exists
        # 2. Convert trajectories to Verifiers format
        # 3. Run trainer.train_step()
        # 4. Handle any errors
        
        print(f"\n⚠️  Verifiers GRPO training not yet fully implemented")
        print(f"   This is a placeholder that shows trajectory collection works")
        print(f"   Full implementation would require Verifiers trainer setup")
        
        return {
            'iteration': training.current_iteration,
            'metrics': metrics,
            'algorithm': 'GRPO',
            'learning_rate': training.learning_rate,
            'status': 'placeholder',
            'message': 'Verifiers GRPO training placeholder - trajectory collection successful'
        }