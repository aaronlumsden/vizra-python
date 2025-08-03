"""
Example of using ART (Adaptive Randomized Trials) for Reinforcement Learning agents.
This demonstrates how to create RL agents that can adapt their behavior based on
trial outcomes and optimize decision-making over time.
"""

import json
import random
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from vizra import BaseAgent, ToolInterface, AgentContext


@dataclass
class ARTState:
    """State representation for ART-based RL."""
    trial_count: int = 0
    successes: int = 0
    failures: int = 0
    arm_rewards: Dict[str, List[float]] = None
    exploration_rate: float = 0.3
    adaptation_rate: float = 0.1
    
    def __post_init__(self):
        if self.arm_rewards is None:
            self.arm_rewards = {}


class ExperimentDesignTool(ToolInterface):
    """
    Tool for designing adaptive experiments using ART methodology.
    """
    
    def __init__(self):
        self.experiments = {}
    
    def definition(self) -> dict:
        return {
            'name': 'design_experiment',
            'description': 'Design an adaptive randomized trial experiment',
            'parameters': {
                'type': 'object',
                'properties': {
                    'experiment_name': {
                        'type': 'string',
                        'description': 'Name of the experiment'
                    },
                    'arms': {
                        'type': 'array',
                        'description': 'List of treatment arms/actions to test',
                        'items': {'type': 'string'}
                    },
                    'objective': {
                        'type': 'string',
                        'description': 'What metric to optimize (e.g., conversion_rate, reward, success_rate)'
                    },
                    'sample_size': {
                        'type': 'integer',
                        'description': 'Initial sample size per arm',
                        'default': 100
                    },
                    'adaptation_strategy': {
                        'type': 'string',
                        'description': 'Strategy for adapting allocation',
                        'enum': ['thompson_sampling', 'ucb', 'epsilon_greedy', 'exp3'],
                        'default': 'thompson_sampling'
                    }
                },
                'required': ['experiment_name', 'arms', 'objective']
            }
        }
    
    def execute(self, arguments: dict, context: AgentContext) -> str:
        """Design and initialize an ART experiment."""
        exp_name = arguments['experiment_name']
        arms = arguments['arms']
        
        # Initialize experiment state
        self.experiments[exp_name] = {
            'arms': arms,
            'objective': arguments['objective'],
            'sample_size': arguments.get('sample_size', 100),
            'strategy': arguments.get('adaptation_strategy', 'thompson_sampling'),
            'arm_stats': {arm: {'trials': 0, 'successes': 0, 'total_reward': 0} for arm in arms},
            'allocation_probabilities': {arm: 1.0/len(arms) for arm in arms}
        }
        
        return json.dumps({
            'status': 'success',
            'experiment': exp_name,
            'arms': arms,
            'initial_allocation': self.experiments[exp_name]['allocation_probabilities'],
            'strategy': self.experiments[exp_name]['strategy'],
            'message': f'Experiment {exp_name} initialized with {len(arms)} arms'
        })


class RunTrialTool(ToolInterface):
    """
    Tool for running trials in an adaptive experiment.
    """
    
    def __init__(self):
        self.art_state = ARTState()
    
    def definition(self) -> dict:
        return {
            'name': 'run_trial',
            'description': 'Run a trial in an adaptive randomized experiment',
            'parameters': {
                'type': 'object',
                'properties': {
                    'experiment_name': {
                        'type': 'string',
                        'description': 'Name of the experiment'
                    },
                    'context_features': {
                        'type': 'object',
                        'description': 'Contextual features for the trial (optional)',
                        'additionalProperties': True
                    },
                    'num_trials': {
                        'type': 'integer',
                        'description': 'Number of trials to run',
                        'default': 1
                    }
                },
                'required': ['experiment_name']
            }
        }
    
    def execute(self, arguments: dict, context: AgentContext) -> str:
        """Run adaptive trials and collect results."""
        exp_name = arguments['experiment_name']
        num_trials = arguments.get('num_trials', 1)
        context_features = arguments.get('context_features', {})
        
        results = []
        
        for _ in range(num_trials):
            # Select arm based on current allocation strategy
            arm = self._select_arm(exp_name, context_features)
            
            # Simulate trial outcome (in real scenarios, this would be actual data)
            reward = self._simulate_outcome(arm, context_features)
            
            # Update statistics
            self._update_statistics(exp_name, arm, reward)
            
            # Adapt allocation based on results
            self._adapt_allocation(exp_name)
            
            results.append({
                'arm': arm,
                'reward': reward,
                'trial_number': self.art_state.trial_count
            })
            
            self.art_state.trial_count += 1
        
        return json.dumps({
            'status': 'success',
            'trials_run': num_trials,
            'results': results,
            'total_trials': self.art_state.trial_count,
            'current_best_arm': self._get_best_arm(exp_name)
        })
    
    def _select_arm(self, exp_name: str, context: dict) -> str:
        """Select an arm based on the adaptive strategy."""
        # For demonstration, using simple probability-based selection
        # In practice, this would implement Thompson Sampling, UCB, etc.
        arms = list(self.art_state.arm_rewards.keys())
        if not arms:
            arms = ['action_a', 'action_b', 'action_c']  # Default arms
        
        if random.random() < self.art_state.exploration_rate:
            # Explore
            return random.choice(arms)
        else:
            # Exploit best known arm
            return self._get_best_arm(exp_name)
    
    def _simulate_outcome(self, arm: str, context: dict) -> float:
        """Simulate trial outcome (replace with real data in production)."""
        # Simulated rewards for different arms
        base_rewards = {
            'action_a': 0.6,
            'action_b': 0.7,
            'action_c': 0.5
        }
        
        base = base_rewards.get(arm, 0.5)
        # Add noise
        reward = base + random.gauss(0, 0.1)
        
        # Context can influence reward
        if context.get('user_type') == 'premium':
            reward += 0.1
        
        return max(0, min(1, reward))  # Bound between 0 and 1
    
    def _update_statistics(self, exp_name: str, arm: str, reward: float):
        """Update arm statistics."""
        if arm not in self.art_state.arm_rewards:
            self.art_state.arm_rewards[arm] = []
        
        self.art_state.arm_rewards[arm].append(reward)
        
        if reward > 0.5:  # Define success threshold
            self.art_state.successes += 1
        else:
            self.art_state.failures += 1
    
    def _adapt_allocation(self, exp_name: str):
        """Adapt allocation probabilities based on performance."""
        # Simple adaptation: increase probability for better-performing arms
        total_reward = {}
        for arm, rewards in self.art_state.arm_rewards.items():
            if rewards:
                total_reward[arm] = np.mean(rewards)
            else:
                total_reward[arm] = 0.5  # Prior
        
        # Normalize to probabilities
        total = sum(total_reward.values())
        if total > 0:
            for arm in total_reward:
                # Smooth update
                old_prob = 1.0 / len(total_reward)
                new_prob = total_reward[arm] / total
                adapted_prob = (1 - self.art_state.adaptation_rate) * old_prob + self.art_state.adaptation_rate * new_prob
                total_reward[arm] = adapted_prob
    
    def _get_best_arm(self, exp_name: str) -> str:
        """Get the best performing arm so far."""
        best_arm = None
        best_reward = -float('inf')
        
        for arm, rewards in self.art_state.arm_rewards.items():
            if rewards:
                avg_reward = np.mean(rewards)
                if avg_reward > best_reward:
                    best_reward = avg_reward
                    best_arm = arm
        
        return best_arm or 'unknown'


class AnalyzeResultsTool(ToolInterface):
    """
    Tool for analyzing ART experiment results.
    """
    
    def definition(self) -> dict:
        return {
            'name': 'analyze_results',
            'description': 'Analyze results from adaptive randomized trials',
            'parameters': {
                'type': 'object',
                'properties': {
                    'experiment_name': {
                        'type': 'string',
                        'description': 'Name of the experiment to analyze'
                    },
                    'metrics': {
                        'type': 'array',
                        'description': 'Metrics to calculate',
                        'items': {
                            'type': 'string',
                            'enum': ['mean_reward', 'confidence_intervals', 'regret', 'convergence_rate']
                        },
                        'default': ['mean_reward', 'confidence_intervals']
                    }
                },
                'required': ['experiment_name']
            }
        }
    
    def execute(self, arguments: dict, context: AgentContext) -> str:
        """Analyze experiment results."""
        exp_name = arguments['experiment_name']
        metrics = arguments.get('metrics', ['mean_reward', 'confidence_intervals'])
        
        # Access the shared state (in production, this would be from a database)
        # For demo, using mock data
        mock_results = {
            'action_a': [0.6, 0.65, 0.58, 0.62, 0.61],
            'action_b': [0.7, 0.72, 0.68, 0.71, 0.73],
            'action_c': [0.5, 0.48, 0.52, 0.49, 0.51]
        }
        
        analysis = {}
        
        if 'mean_reward' in metrics:
            analysis['mean_rewards'] = {
                arm: np.mean(rewards) for arm, rewards in mock_results.items()
            }
        
        if 'confidence_intervals' in metrics:
            analysis['confidence_intervals'] = {}
            for arm, rewards in mock_results.items():
                mean = np.mean(rewards)
                std = np.std(rewards)
                n = len(rewards)
                ci = 1.96 * std / np.sqrt(n)  # 95% CI
                analysis['confidence_intervals'][arm] = {
                    'lower': mean - ci,
                    'upper': mean + ci
                }
        
        if 'regret' in metrics:
            # Calculate cumulative regret
            best_arm_mean = max(np.mean(rewards) for rewards in mock_results.values())
            analysis['regret'] = {
                arm: best_arm_mean - np.mean(rewards) 
                for arm, rewards in mock_results.items()
            }
        
        return json.dumps({
            'status': 'success',
            'experiment': exp_name,
            'analysis': analysis,
            'recommendation': 'action_b',  # Based on highest mean reward
            'trials_analyzed': sum(len(rewards) for rewards in mock_results.values())
        })


class ARTReinforcementAgent(BaseAgent):
    """
    An agent that uses Adaptive Randomized Trials for reinforcement learning.
    """
    name = 'art_rl_agent'
    description = 'RL agent using Adaptive Randomized Trials for decision optimization'
    instructions = '''You are an expert in Adaptive Randomized Trials (ART) for reinforcement learning.
    
    Your role is to:
    1. Design adaptive experiments to optimize decision-making
    2. Run trials that balance exploration and exploitation
    3. Analyze results to identify optimal actions
    4. Adapt strategies based on observed outcomes
    
    Key principles:
    - Start with exploration to gather data about all options
    - Gradually shift to exploitation of best-performing actions
    - Use statistical methods to ensure reliable conclusions
    - Consider contextual factors that might influence outcomes
    
    When users ask for help with optimization problems:
    - Set up appropriate experiments
    - Run sufficient trials to gather meaningful data
    - Provide clear analysis and recommendations
    - Explain the uncertainty in your conclusions'''
    
    model = 'gpt-4o'
    tools = [ExperimentDesignTool, RunTrialTool, AnalyzeResultsTool]
    
    def before_tool_call(self, tool_name: str, arguments: dict, context: AgentContext) -> None:
        """Log RL operations."""
        if tool_name == 'design_experiment':
            print(f"\n🔬 Designing ART experiment: {arguments.get('experiment_name')}")
        elif tool_name == 'run_trial':
            print(f"\n🎲 Running {arguments.get('num_trials', 1)} trials...")
        elif tool_name == 'analyze_results':
            print(f"\n📊 Analyzing results for: {arguments.get('experiment_name')}")


# Example usage with OpenPipe for model selection
class OpenPipeARTAgent(ARTReinforcementAgent):
    """
    ART agent configured to use OpenPipe for adaptive model selection.
    """
    model = 'openpipe:rl-specialist-v1'  # Your OpenPipe RL-optimized model
    
    def after_tool_result(self, tool_name: str, result: str, context: AgentContext) -> None:
        """Track performance for OpenPipe optimization."""
        try:
            data = json.loads(result)
            if tool_name == 'analyze_results' and 'recommendation' in data:
                # Log to OpenPipe for model improvement
                print(f"📈 Recommendation: {data['recommendation']}")
                # In production: send feedback to OpenPipe
                # openpipe.log_feedback(context, data)
        except:
            pass


if __name__ == "__main__":
    print("=== ART Reinforcement Learning Agent Example ===\n")
    
    # Create an RL agent
    agent = ARTReinforcementAgent()
    
    # Example 1: Optimize a decision problem
    response1 = agent.run("""
    I need to optimize my email marketing campaign. I have three different 
    email templates (formal, casual, personalized) and I want to find which 
    one gets the best click-through rate. Can you help me run an adaptive 
    experiment?
    """)
    print(f"\nAgent: {response1}\n")
    
    # Example 2: With context
    context = AgentContext()
    rl_agent = agent.with_context(context)
    
    # Run trials
    response2 = rl_agent.run("Please run 50 trials of the experiment")
    print(f"\nAgent: {response2}\n")
    
    # Analyze results
    response3 = rl_agent.run("What do the results show? Which template should I use?")
    print(f"\nAgent: {response3}\n")