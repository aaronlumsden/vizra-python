"""
OpenPipe ART provider for Vizra training.
"""

import asyncio
import json
import re
from typing import List, Dict, Any, Optional
from datetime import datetime


class ARTProvider:
    """
    Provider for OpenPipe ART (Agent Reinforcement Trainer) integration.
    
    This provider uses ART's internal model serving instead of LiteLLM,
    allowing for proper GRPO training with logprobs.
    """
    
    def __init__(self, model_name: str, base_model: str, project: str = "vizra-training", **kwargs):
        """
        Initialize ART provider.
        
        Args:
            model_name: Name for the trainable model
            base_model: Base model to use (e.g., "Qwen/Qwen2.5-0.5B-Instruct")
            project: Project name for ART
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
        self.config = kwargs
        
        # Store imports for later use
        self.Trajectory = Trajectory
        self.TrainConfig = TrainConfig
        self.TrajectoryGroup = TrajectoryGroup
        
        # Initialize ART backend and model
        print(f"🔧 Creating ART backend and model with name: '{model_name}'")
        self.backend = LocalBackend()
        print(f"🔧 Backend created: {self.backend}")
        
        try:
            self.model = TrainableModel(
                name=model_name,
                project=project,
                base_model=base_model
            )
            print(f"🔧 ART model created: {self.model}")
            print(f"🔧 Model attributes: name={getattr(self.model, 'name', 'N/A')}, base={getattr(self.model, 'base_model', 'N/A')}")
        except Exception as e:
            print(f"❌ Error creating TrainableModel: {e}")
            raise
        
        # Try registering immediately with better error handling
        self._registered = False
        print(f"🔧 Attempting immediate registration...")
        try:
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.model.register(self.backend))
            self._registered = True
            print(f"✅ Model registered successfully!")
        except Exception as e:
            error_msg = str(e)
            print(f"⚠️  Registration error: {error_msg}")
            
            # Special handling for known errors
            if "aimv2" in error_msg:
                print("ℹ️  Ignoring aimv2 conflict - this is a known issue with transformers version")
                # Don't mark as registered yet - we'll handle it during trajectory collection
            elif "already registered" in error_msg or "already exists" in error_msg:
                print("ℹ️  Model appears to already be registered")
                self._registered = True
    
    def collect_trajectories(self, training, data_rows: List[Dict[str, Any]], agent_class) -> List[Dict[str, Any]]:
        """
        Collect trajectories using ART's model while respecting agent configuration.
        
        Args:
            training: The training instance (for accessing methods like calculate_reward)
            data_rows: List of training data rows
            agent_class: The agent class (for accessing tools, instructions, etc.)
            
        Returns:
            List of trajectory dictionaries in Vizra format
        """
        trajectories = []
        
        # Run async collection
        loop = asyncio.new_event_loop()
        trajectories = loop.run_until_complete(
            self._collect_trajectories_async(training, data_rows, agent_class)
        )
        
        return trajectories
    
    async def _collect_trajectories_async(self, training, data_rows: List[Dict[str, Any]], agent_class) -> List[Dict[str, Any]]:
        """Async implementation of trajectory collection."""
        trajectories = []
        
        # First check if ART server is accessible
        import aiohttp
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get("http://localhost:7999/v1/models") as resp:
                    if resp.status == 200:
                        models = await resp.json()
                        print(f"🔍 ART server models available: {models}")
                    else:
                        print(f"⚠️  ART server returned status {resp.status}")
        except Exception as e:
            print(f"⚠️  Could not connect to ART server at localhost:7999: {e}")
        
        # Ensure model is registered
        if not self._registered:
            print("⚠️  Model not registered, attempting registration...")
            try:
                await self.model.register(self.backend)
                self._registered = True
                print("✅ Model registered successfully")
            except Exception as e:
                error_str = str(e)
                print(f"🔍 Registration error details: {error_str}")
                
                # Handle various known error cases
                if "already exists" in error_str or "already used" in error_str or "already registered" in error_str:
                    print("ℹ️  Model appears to be already registered, proceeding...")
                    self._registered = True
                elif "aimv2" in error_str:
                    print("ℹ️  Ignoring aimv2 conflict, attempting to proceed...")
                    self._registered = True
                elif "not yet available" in error_str:
                    # This suggests we need to register differently
                    print("⚠️  Model not available, registration may have failed")
                    raise RuntimeError(f"Failed to register model: {e}")
                else:
                    raise RuntimeError(f"Failed to register model: {e}")
        
        # Get ART's OpenAI client
        client = None
        try:
            print("🔍 Getting OpenAI client from model...")
            client = self.model.openai_client()
            print("✅ OpenAI client obtained successfully")
        except Exception as e:
            print(f"❌ Failed to get OpenAI client: {e}")
            
            # Try different approaches based on the error
            if "not yet available" in str(e):
                print("🔧 Model not available, trying registration approaches...")
                
                # Approach 1: Try force registration
                if not self._registered:
                    print("  1️⃣ Attempting force registration...")
                    try:
                        await self.model.register(self.backend)
                        self._registered = True
                        client = self.model.openai_client()
                        print("  ✅ Force registration successful")
                    except Exception as e2:
                        print(f"  ❌ Force registration failed: {e2}")
                
                # Approach 2: Try to get the client directly from backend
                if client is None:
                    print("  2️⃣ Trying to get client from backend...")
                    try:
                        # Check if backend has a method to get client
                        if hasattr(self.backend, 'get_client'):
                            client = self.backend.get_client(self.model_name)
                            print("  ✅ Got client from backend")
                        elif hasattr(self.backend, 'openai_client'):
                            client = self.backend.openai_client()
                            print("  ✅ Got OpenAI client from backend")
                    except Exception as e3:
                        print(f"  ❌ Backend client approach failed: {e3}")
                
                # Approach 3: Create a new model instance
                if client is None:
                    print("  3️⃣ Trying to create a new model instance...")
                    try:
                        # Import here to avoid circular imports
                        from art import TrainableModel
                        temp_model = TrainableModel(
                            name=f"{self.model_name}-temp",
                            project=self.project,
                            base_model=self.base_model
                        )
                        await temp_model.register(self.backend)
                        client = temp_model.openai_client()
                        self.model = temp_model  # Use the new model
                        self._registered = True
                        print("  ✅ New model instance created and registered")
                    except Exception as e4:
                        print(f"  ❌ New model instance failed: {e4}")
            
            if client is None:
                raise RuntimeError(f"Could not obtain OpenAI client after all attempts. Original error: {e}")
        
        # Get agent configuration
        instructions = agent_class._get_instructions()
        tools = agent_class._get_tools()
        
        # Debug: Check model info
        print(f"🔍 Model inference name: {getattr(self.model, 'inference_model_name', 'Not set')}")
        print(f"🔍 Model base: {self.base_model}")
        
        for i, row_data in enumerate(data_rows):
            print(f"\r[{i+1}/{len(data_rows)}] Collecting trajectories with ART...", end='', flush=True)
            
            try:
                # Prepare the trajectory data from CSV
                trajectory_data = training.prepare_trajectory(row_data)
                prompt = trajectory_data['prompt']
                
                # Create ART trajectory
                art_trajectory = await self._create_art_trajectory(
                    client, prompt, instructions, tools, row_data, training
                )
                
                # Convert to Vizra format
                vizra_trajectory = self._art_to_vizra_trajectory(
                    art_trajectory, trajectory_data, row_data
                )
                
                trajectories.append(vizra_trajectory)
                
            except Exception as e:
                print(f"\n⚠️  Error collecting trajectory {i+1} with ART: {e}")
                trajectories.append({
                    'error': str(e),
                    'reward': 0.0,
                    'row_data': row_data
                })
        
        print()  # New line after progress
        return trajectories
    
    async def _create_art_trajectory(self, client, prompt: str, instructions: str, 
                                   tools: List, row_data: Dict, training) -> Any:
        """Create a single ART trajectory."""
        # Build initial messages
        messages_and_choices = [
            {"role": "system", "content": instructions},
            {"role": "user", "content": prompt}
        ]
        
        # Generate first response
        response = await client.chat.completions.create(
            messages=[m for m in messages_and_choices if isinstance(m, dict)],
            model=self.model.inference_model_name,
            max_tokens=200,
            temperature=0.7,
            logprobs=True,  # Important for GRPO
            top_logprobs=5
        )
        
        choice = response.choices[0]
        messages_and_choices.append(choice)
        
        # Extract final response
        assistant_response = choice.message.content
        final_response = assistant_response
        
        # Check if tool was used (simple check for XML-style tools)
        if tools and '<' in assistant_response and '>' in assistant_response:
            # Handle tool execution
            final_response = await self._handle_tool_execution(
                client, messages_and_choices, assistant_response, tools, row_data
            )
        
        # Calculate reward
        reward = training.calculate_reward(row_data, final_response)
        
        # Create ART trajectory
        trajectory = self.Trajectory(
            messages_and_choices=messages_and_choices,
            reward=reward
        )
        
        return trajectory
    
    async def _handle_tool_execution(self, client, messages_and_choices, 
                                   assistant_response: str, tools: List, row_data: Dict) -> str:
        """Handle tool execution in the trajectory."""
        # Simple XML tool extraction (can be enhanced)
        for tool_class in tools:
            if hasattr(tool_class, 'xml_tag') and tool_class.xml_tag:
                pattern = f'<{tool_class.xml_tag}>(.*?)</{tool_class.xml_tag}>'
                match = re.search(pattern, assistant_response, re.DOTALL)
                
                if match:
                    # Simulate tool execution
                    tool_content = match.group(1).strip()
                    
                    # For chord tool, we can use expected output as tool result
                    tool_output = row_data.get('expected_chord', row_data.get('expected_output', ''))
                    
                    # Add tool output as user message
                    messages_and_choices.append({
                        "role": "user",
                        "content": f"Tool output: {tool_output}\n\nNow provide the final answer in the format 'RootNote Quality' (e.g., 'C Major')"
                    })
                    
                    # Get final response
                    temp_messages = []
                    for item in messages_and_choices:
                        if isinstance(item, dict) and 'role' in item:
                            temp_messages.append(item)
                        elif hasattr(item, 'message'):
                            temp_messages.append(item.message.model_dump())
                    
                    response2 = await client.chat.completions.create(
                        messages=temp_messages,
                        model=self.model.inference_model_name,
                        max_tokens=50,
                        temperature=0.3,
                        logprobs=True,
                        top_logprobs=5
                    )
                    
                    choice2 = response2.choices[0]
                    messages_and_choices.append(choice2)
                    
                    return choice2.message.content
        
        return assistant_response
    
    def _art_to_vizra_trajectory(self, art_trajectory, trajectory_data: Dict, row_data: Dict) -> Dict:
        """Convert ART trajectory to Vizra format."""
        # Extract final response from trajectory
        final_response = ""
        for item in art_trajectory.messages_and_choices:
            if hasattr(item, 'message') and hasattr(item.message, 'role') and item.message.role == 'assistant':
                final_response = item.message.content
            elif isinstance(item, dict) and item.get('role') == 'assistant':
                final_response = item.get('content', '')
        
        # Build Vizra trajectory
        vizra_trajectory = trajectory_data.copy()
        vizra_trajectory.update({
            'response': final_response,
            'reward': art_trajectory.reward,
            'row_data': row_data,
            'art_trajectory': art_trajectory  # Keep original for training
        })
        
        return vizra_trajectory
    
    def train_step(self, training, trajectories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Send trajectories to ART for training.
        
        Args:
            training: The training instance
            trajectories: List of trajectories from collect_trajectories
            
        Returns:
            Training step results
        """
        # First compute metrics using Vizra's method
        metrics = training.compute_metrics(trajectories)
        
        # Extract ART trajectories
        art_trajectories = []
        for traj in trajectories:
            if 'art_trajectory' in traj and not traj.get('error'):
                art_trajectories.append(traj['art_trajectory'])
        
        if not art_trajectories:
            print("⚠️  No valid ART trajectories to train on")
            return {
                'iteration': training.current_iteration,
                'metrics': metrics,
                'algorithm': 'grpo',
                'learning_rate': training.learning_rate,
                'art_status': 'no_trajectories'
            }
        
        # Send to ART for training
        print(f"  📊 Sending {len(art_trajectories)} trajectories to ART...")
        
        try:
            # Group trajectories for GRPO
            trajectory_groups = []
            group_size = 8  # Good size for GRPO comparison
            
            for i in range(0, len(art_trajectories), group_size):
                group = art_trajectories[i:i+group_size]
                if len(group) >= 2:  # Need at least 2 for comparison
                    trajectory_groups.append(self.TrajectoryGroup(group))
            
            if not trajectory_groups:
                # If we don't have enough for groups, make one group
                trajectory_groups = [self.TrajectoryGroup(art_trajectories)]
            
            # Train with ART
            config = self.TrainConfig(
                learning_rate=training.learning_rate,
                allow_training_without_logprobs=False  # We have logprobs
            )
            
            # Run training asynchronously
            loop = asyncio.new_event_loop()
            result = loop.run_until_complete(
                self.model.train(trajectory_groups, config=config, verbose=True)
            )
            
            print(f"  ✅ ART training complete: {result}")
            
            return {
                'iteration': training.current_iteration,
                'metrics': metrics,
                'algorithm': 'grpo',
                'learning_rate': training.learning_rate,
                'art_status': 'success',
                'art_result': str(result)
            }
            
        except Exception as e:
            print(f"  ⚠️  ART training error: {e}")
            return {
                'iteration': training.current_iteration,
                'metrics': metrics,
                'algorithm': 'grpo',
                'learning_rate': training.learning_rate,
                'art_status': 'error',
                'art_error': str(e)
            }