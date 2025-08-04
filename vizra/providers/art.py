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
                    formatted_tools.append({
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "description": getattr(tool_func, '__doc__', f"Tool: {tool_name}"),
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
                
                # Calculate reward
                reward = training.calculate_reward(row_data, agent_response)
                
                # Create ART trajectory
                art_messages = messages.copy()
                art_messages.append({
                    "role": "assistant",
                    "content": response.choices[0].message.content,
                    "tool_calls": response.choices[0].message.tool_calls
                })
                
                # Add tool responses if any
                if response.choices[0].message.tool_calls:
                    for tool_call in response.choices[0].message.tool_calls:
                        tool_name = tool_call.function.name
                        tool_args = json.loads(tool_call.function.arguments)
                        tool_input = tool_args.get('input', '')
                        
                        if tool_name in tools:
                            tool_result = tools[tool_name](tool_input)
                            art_messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": str(tool_result)
                            })
                
                # Create trajectory with all required fields for ART 0.3.12
                # ART expects messages_and_choices format
                messages_and_choices = [{
                    "messages": art_messages,
                    "choice": response.choices[0],
                    "logprobs": response.choices[0].logprobs
                }]
                
                trajectory = self.Trajectory(
                    messages_and_choices=messages_and_choices,
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
                
                # Execute tool if available
                if tool_name in tools:
                    tool_result = tools[tool_name](tool_input)
                    tool_outputs.append(f"<{tool_name}>{tool_input}</{tool_name}> → {tool_result}")
            
            # Return the tool outputs as the response
            return "\n".join(tool_outputs)
        
        # Otherwise return the direct response
        return choice.message.content or ""
    
    def train_step(self, trajectories: List[Dict[str, Any]]) -> Dict[str, Any]:
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
            
            return {
                'status': 'success',
                'trajectories_trained': len(art_trajectories),
                'message': f'Successfully sent {len(art_trajectories)} trajectories to ART for training'
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'trajectories_attempted': len(art_trajectories)
            }