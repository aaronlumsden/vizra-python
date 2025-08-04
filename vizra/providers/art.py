"""
OpenPipe ART provider for Vizra training with external vLLM support.
"""

import asyncio
import json
import re
from typing import List, Dict, Any, Optional
from datetime import datetime


class ARTProvider:
    """
    Provider for OpenPipe ART (Agent Reinforcement Trainer) integration.
    
    Supports both internal ART model serving and external vLLM servers.
    """
    
    def __init__(self, model_name: str, base_model: str, project: str = "vizra-training", 
                 inference_base_url: Optional[str] = None, **kwargs):
        """
        Initialize ART provider.
        
        Args:
            model_name: Name for the trainable model
            base_model: Base model to use (e.g., "Qwen/Qwen2.5-0.5B-Instruct")
            project: Project name for ART
            inference_base_url: Optional external inference URL (e.g., "http://localhost:8000/v1")
            **kwargs: Additional configuration
        """
        try:
            from art import TrainableModel, Trajectory, TrainConfig
            from art.local.backend import LocalBackend
            from art.trajectories import TrajectoryGroup
        except ImportError:
            raise ImportError(
                "OpenPipe ART is required for ARTProvider. "
                "Install with: pip install openpipe-art[backend]"
            )
        
        self.model_name = model_name
        self.base_model = base_model
        self.project = project
        self.inference_base_url = inference_base_url
        self.config = kwargs
        
        # Store imports for later use
        self.Trajectory = Trajectory
        self.TrainConfig = TrainConfig
        self.TrajectoryGroup = TrajectoryGroup
        
        # Initialize ART backend and model
        self.backend = LocalBackend()
        
        # Configure model with external inference if provided
        if inference_base_url:
            self.model = TrainableModel(
                name=model_name,
                project=project,
                base_model=base_model,
                inference_base_url=inference_base_url,
                inference_model_name=base_model
            )
            print(f"✅ ART configured with external inference at {inference_base_url}")
        else:
            self.model = TrainableModel(
                name=model_name,
                project=project,
                base_model=base_model
            )
        
        # Register model with backend (may fail with external inference)
        self._registered = False
        if not inference_base_url:
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self.model.register(self.backend))
                self._registered = True
            except Exception:
                # Registration might fail but we can still proceed
                pass
    
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
    
    def collect_trajectories(self, training, data_rows: List[Dict[str, Any]], agent_class) -> List[Dict[str, Any]]:
        """
        Collect trajectories using ART's model while respecting agent configuration.
        
        Args:
            training: The training instance
            data_rows: List of training data rows
            agent_class: The agent class
            
        Returns:
            List of trajectory dictionaries
        """
        # Run async collection in sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        trajectories = loop.run_until_complete(
            self._collect_trajectories_async(training, data_rows, agent_class)
        )
        
        return trajectories
    
    async def _collect_trajectories_async(self, training, data_rows: List[Dict[str, Any]], agent_class) -> List[Dict[str, Any]]:
        """Async implementation of trajectory collection."""
        trajectories = []
        
        # Ensure we have a registered model or external inference
        if not self._registered and not self.inference_base_url:
            try:
                await self.model.register(self.backend)
                self._registered = True
            except Exception as e:
                # If registration fails but we have external inference, continue
                if self.inference_base_url:
                    self._registered = True
                else:
                    raise RuntimeError(f"Failed to register model: {e}")
        
        # Get ART's OpenAI client
        try:
            client = self.model.openai_client()
        except Exception as e:
            if self.inference_base_url:
                # For external inference, create client directly
                from openai import AsyncOpenAI
                client = AsyncOpenAI(
                    api_key="dummy",
                    base_url=self.inference_base_url
                )
            else:
                raise RuntimeError(f"Could not obtain OpenAI client: {e}")
        
        # Get agent configuration
        instructions = agent_class._get_instructions()
        raw_tools = agent_class._get_tools()
        tools = self._normalize_tools(raw_tools)  # Normalize to dict format
        
        for i, row_data in enumerate(data_rows):
            print(f"\r[{i+1}/{len(data_rows)}] Collecting trajectories...", end='', flush=True)
            
            try:
                # Prepare the trajectory data
                trajectory_data = training.prepare_trajectory(row_data)
                prompt = trajectory_data.get('prompt', '')
                
                # Format tools for ART
                formatted_tools = []
                for tool_name, tool_func in tools.items():
                    # Use the tool's xml_tag if available, otherwise use class name
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
                
                # Create messages list
                messages = [
                    {"role": "system", "content": instructions},
                    {"role": "user", "content": prompt}
                ]
                
                # Call ART's model with tools
                response = await client.chat.completions.create(
                    model=self.model.inference_model_name or self.base_model,
                    messages=messages,
                    tools=formatted_tools if formatted_tools else None,
                    max_tokens=500,
                    temperature=0.7,
                    logprobs=True
                )
                
                # Process response to handle tool calls
                agent_response = self._process_art_response(response, tools)
                
                # Also keep the full response format for metrics that need to see tool usage
                full_response_for_metrics = self._format_response_for_metrics(response, tools)
                
                # DEBUG: Log the response and reward calculation
                print(f"\n🔍 DEBUG Trajectory {i+1}:")
                print(f"   Question: {prompt[:100]}...")
                print(f"   Raw Response: {response.choices[0].message.content}")
                print(f"   Tool Calls: {response.choices[0].message.tool_calls}")
                print(f"   Processed Response: {agent_response}")
                print(f"   Expected: {row_data.get('expected_chord', 'N/A')}")
                
                # Calculate reward using full response for tool detection
                reward = training.calculate_reward(row_data, full_response_for_metrics)
                print(f"   Reward: {reward}")
                print(f"   ---")
                
                # Create ART trajectory
                art_messages = messages.copy()
                assistant_msg = {
                    "role": "assistant",
                    "content": response.choices[0].message.content or ""
                }
                
                # Only add tool_calls if they exist
                if response.choices[0].message.tool_calls:
                    assistant_msg["tool_calls"] = response.choices[0].message.tool_calls
                
                art_messages.append(assistant_msg)
                
                # Add tool responses if any
                if response.choices[0].message.tool_calls:
                    for tool_call in response.choices[0].message.tool_calls:
                        called_name = tool_call.function.name
                        tool_args = json.loads(tool_call.function.arguments)
                        tool_input = tool_args.get('input', '')
                        
                        # Find the tool by matching xml_tag or class name
                        tool_obj = None
                        for tool_class_name, tool_instance in tools.items():
                            if (hasattr(tool_instance, 'xml_tag') and tool_instance.xml_tag == called_name) or tool_class_name == called_name:
                                tool_obj = tool_instance
                                break
                        
                        if tool_obj:
                            # For Vizra tools, use the execute method with proper arguments
                            if hasattr(tool_obj, 'execute'):
                                tool_result = tool_obj.execute({'notes_str': tool_input})
                            elif hasattr(tool_obj, '__call__'):
                                tool_result = tool_obj(tool_input)
                            elif hasattr(tool_obj, 'run'):
                                tool_result = tool_obj.run(tool_input)
                            else:
                                tool_result = f"Tool {called_name} has no execute, __call__, or run method"
                        else:
                            tool_result = f"Tool {called_name} not found"
                            
                        art_messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": str(tool_result)
                        })
                
                # Create trajectory with correct format for ART
                trajectory = self.Trajectory(
                    messages_and_choices=art_messages,
                    reward=reward
                )
                
                # Also create Vizra-compatible trajectory
                vizra_trajectory = {
                    'prompt': prompt,
                    'response': agent_response,
                    'reward': reward,
                    'metadata': {
                        'art_trajectory': trajectory,
                        'timestamp': datetime.now().isoformat()
                    }
                }
                
                trajectories.append(vizra_trajectory)
                
            except Exception as e:
                print(f"\n⚠️  Error collecting trajectory {i+1}: {e}")
                continue
        
        print()  # New line after progress
        return trajectories
    
    def _process_art_response(self, response, tools) -> str:
        """Process ART response to extract final answer, handling tool calls."""
        if not response.choices:
            return ""
        
        choice = response.choices[0]
        
        # If there are tool calls, execute them and format the response
        if choice.message.tool_calls:
            tool_outputs = []
            for tool_call in choice.message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)
                tool_input = tool_args.get('input', '')
                
                # Find the tool by matching xml_tag or class name
                tool_obj = None
                for tool_class_name, tool_instance in tools.items():
                    if (hasattr(tool_instance, 'xml_tag') and tool_instance.xml_tag == tool_name) or tool_class_name == tool_name:
                        tool_obj = tool_instance
                        break
                
                if tool_obj:
                    # For Vizra tools, use the execute method with proper arguments
                    if hasattr(tool_obj, 'execute'):
                        tool_result = tool_obj.execute({'notes_str': tool_input})
                    elif hasattr(tool_obj, '__call__'):
                        tool_result = tool_obj(tool_input)
                    elif hasattr(tool_obj, 'run'):
                        tool_result = tool_obj.run(tool_input)
                    else:
                        tool_result = f"Tool {tool_name} has no execute, __call__, or run method"
                else:
                    tool_result = f"Tool {tool_name} not found"
                    
                tool_outputs.append(str(tool_result))  # Return just the result, not the full format
            
            # Return the tool outputs as the response
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
                
                # Find the tool and execute it
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
                        tool_result = f"Tool {tool_name} has no execute, __call__, or run method"
                else:
                    tool_result = f"Tool {tool_name} not found"
                    
                # Format as XML for tool usage detection, but end with just the result
                tool_outputs.append(f"<{tool_name}>{tool_input}</{tool_name}>\n{tool_result}")
            
            return "\n".join(tool_outputs)
        
        # Otherwise return the direct response
        return choice.message.content or ""
    
    def train_step(self, training, trajectories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Execute a training step using collected trajectories.
        
        Args:
            trajectories: List of trajectories from collect_trajectories
            
        Returns:
            Training metrics
        """
        if not trajectories:
            return {'status': 'skipped', 'reason': 'No trajectories to train on'}
        
        # Extract ART trajectories from metadata
        art_trajectories = []
        for traj in trajectories:
            if 'metadata' in traj and 'art_trajectory' in traj['metadata']:
                art_trajectories.append(traj['metadata']['art_trajectory'])
        
        if not art_trajectories:
            return {'status': 'error', 'reason': 'No ART trajectories found'}
        
        # Create trajectory group
        trajectory_group = self.TrajectoryGroup(
            trajectories=art_trajectories
        )
        
        # Train model using ART
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(
                self.backend.train_on_trajectories(
                    self.model,
                    trajectory_group,
                    self.TrainConfig()
                )
            )
            
            # Calculate metrics from trajectories for compatibility with base class
            # Extract rewards from the original Vizra trajectories, not ART trajectories
            rewards = [traj['reward'] for traj in trajectories if 'reward' in traj]
            if not rewards:
                rewards = [0.0]
            
            import numpy as np
            metrics = {
                'avg_reward': float(np.mean(rewards)),
                'min_reward': float(np.min(rewards)),
                'max_reward': float(np.max(rewards)),
                'std_reward': float(np.std(rewards)),
                'num_trajectories': len(art_trajectories),
                'success_rate': sum(1 for r in rewards if r > 0.5) / len(rewards) if rewards else 0.0
            }
            
            return {
                'iteration': training.current_iteration,
                'metrics': metrics,
                'algorithm': 'GRPO',
                'learning_rate': training.learning_rate,
                'art_status': 'success',
                'trajectories_trained': len(art_trajectories),
                'message': f'Successfully sent {len(art_trajectories)} trajectories to ART for training'
            }
            
        except Exception as e:
            # Return error with empty metrics to avoid crashes
            metrics = {
                'avg_reward': 0.0,
                'min_reward': 0.0,
                'max_reward': 0.0,
                'std_reward': 0.0,
                'num_trajectories': 0,
                'success_rate': 0.0
            }
            
            return {
                'iteration': training.current_iteration,
                'metrics': metrics,
                'algorithm': 'GRPO',
                'learning_rate': training.learning_rate,
                'art_status': 'error',
                'error': str(e),
                'trajectories_attempted': len(art_trajectories)
            }