"""
Simple Reinforcement Learning Agent Example.

This shows how to create an RL agent that:
1. Takes actions
2. Gets rewards/scores for those actions
3. Learns to take better actions over time
"""

import json
import random
from typing import List, Dict
from vizra import BaseAgent, ToolInterface, AgentContext


class ActionTool(ToolInterface):
    """
    Tool for taking actions in the environment.
    """
    
    def __init__(self):
        # Track action history for learning
        self.action_history = []
        self.reward_history = []
    
    def definition(self) -> dict:
        return {
            'name': 'take_action',
            'description': 'Take an action and observe the reward',
            'parameters': {
                'type': 'object',
                'properties': {
                    'action': {
                        'type': 'string',
                        'description': 'The action to take',
                        'enum': ['move_left', 'move_right', 'stay', 'jump']
                    },
                    'reason': {
                        'type': 'string',
                        'description': 'Why you chose this action'
                    }
                },
                'required': ['action']
            }
        }
    
    def execute(self, arguments: dict, context: AgentContext) -> str:
        """Execute action and get reward."""
        action = arguments['action']
        reason = arguments.get('reason', 'No reason given')
        
        # Simulate getting a reward for the action
        # In a real scenario, this would interact with your environment
        reward = self._get_reward(action)
        
        # Store for learning
        self.action_history.append(action)
        self.reward_history.append(reward)
        
        return json.dumps({
            'action': action,
            'reward': reward,
            'total_reward': sum(self.reward_history),
            'message': f'Took action {action}, received reward {reward}'
        })
    
    def _get_reward(self, action: str) -> float:
        """
        Simulate environment rewards.
        In practice, this would be your actual reward function.
        """
        # Simple reward structure for demonstration
        rewards = {
            'move_right': 0.8 + random.uniform(-0.2, 0.2),  # Best action
            'jump': 0.5 + random.uniform(-0.2, 0.2),        # OK action
            'stay': 0.3 + random.uniform(-0.2, 0.2),        # Poor action
            'move_left': 0.1 + random.uniform(-0.2, 0.2),   # Worst action
        }
        return rewards.get(action, 0.0)


class ScoreTool(ToolInterface):
    """
    Tool for scoring/judging a sequence of actions.
    Similar to 'ruler_score_group' but simplified.
    """
    
    def definition(self) -> dict:
        return {
            'name': 'score_actions',
            'description': 'Score a sequence of actions to evaluate performance',
            'parameters': {
                'type': 'object',
                'properties': {
                    'actions': {
                        'type': 'array',
                        'description': 'List of actions to score',
                        'items': {'type': 'string'}
                    },
                    'scoring_model': {
                        'type': 'string',
                        'description': 'Model to use for scoring',
                        'default': 'simple_scorer'
                    }
                },
                'required': ['actions']
            }
        }
    
    def execute(self, arguments: dict, context: AgentContext) -> str:
        """Score a sequence of actions."""
        actions = arguments['actions']
        model = arguments.get('scoring_model', 'simple_scorer')
        
        # Simulate scoring (like ruler_score_group)
        # In practice, this might call an LLM or evaluation model
        scores = []
        for action in actions:
            if action == 'move_right':
                score = 0.9
            elif action == 'jump':
                score = 0.6
            elif action == 'stay':
                score = 0.3
            else:
                score = 0.1
            scores.append(score)
        
        avg_score = sum(scores) / len(scores) if scores else 0
        
        return json.dumps({
            'actions': actions,
            'scores': scores,
            'average_score': avg_score,
            'best_action': actions[scores.index(max(scores))] if scores else None,
            'scoring_model': model
        })


class LearnTool(ToolInterface):
    """
    Tool for learning from past actions and rewards.
    """
    
    def definition(self) -> dict:
        return {
            'name': 'learn_from_experience',
            'description': 'Analyze past actions and learn which ones work best',
            'parameters': {
                'type': 'object',
                'properties': {
                    'num_recent': {
                        'type': 'integer',
                        'description': 'Number of recent actions to analyze',
                        'default': 10
                    }
                },
                'required': []
            }
        }
    
    def execute(self, arguments: dict, context: AgentContext) -> str:
        """Learn from recent experience."""
        # Access action history from the action tool
        # In a real implementation, this would be shared state
        
        # Mock learning results
        learning_summary = {
            'observation': 'move_right consistently gives high rewards',
            'recommendation': 'Prefer move_right action',
            'confidence': 0.85,
            'actions_analyzed': arguments.get('num_recent', 10)
        }
        
        return json.dumps(learning_summary)


class SimpleRLAgent(BaseAgent):
    """
    A simple reinforcement learning agent.
    """
    name = 'simple_rl_agent'
    description = 'A basic RL agent that learns through trial and error'
    instructions = '''You are a reinforcement learning agent trying to maximize rewards.

Your goal is to:
1. Take actions in the environment
2. Observe the rewards you get
3. Learn which actions give the best rewards
4. Improve your strategy over time

Start by exploring different actions, then gradually focus on the ones that work best.
Use the scoring tool to evaluate sequences of actions.
Use the learning tool to analyze your performance and improve.'''
    
    model = 'gpt-4o'
    tools = [ActionTool, ScoreTool, LearnTool]


# Example showing OpenPipe/scoring integration
class ScoringRLAgent(SimpleRLAgent):
    """
    RL agent that uses external scoring/judging.
    Similar to using ruler_score_group.
    """
    
    async def score_with_judge(self, actions: List[str], judge_model: str = "openai/gpt-4"):
        """
        Score actions using an external judge model.
        This is similar to: judged_group = await ruler_score_group(group, "openai/o3")
        """
        # In practice, this would call your scoring API
        # For example with OpenPipe or other evaluation service
        
        # Mock implementation
        scores = {
            'actions': actions,
            'judge_model': judge_model,
            'scores': [0.7, 0.8, 0.9],  # Mock scores
            'feedback': 'Good sequence, improving over time'
        }
        return scores


if __name__ == "__main__":
    print("=== Simple RL Agent Example ===\n")
    
    # Create the agent
    agent = SimpleRLAgent()
    
    # Initial exploration
    print("1. Initial exploration:")
    response1 = agent.run("I need to explore the environment. Try taking 3 different actions.")
    print(f"{response1}\n")
    
    # Score the actions
    print("2. Score recent actions:")
    response2 = agent.run("Score my recent actions to see how well I'm doing")
    print(f"{response2}\n")
    
    # Learn from experience
    print("3. Learn from experience:")
    response3 = agent.run("What have I learned? Which action should I prefer?")
    print(f"{response3}\n")
    
    # Exploit best action
    print("4. Use best strategy:")
    response4 = agent.run("Based on what I've learned, take the best action 5 times")
    print(f"{response4}\n")
    
    # Example of how you might use external scoring (like ruler_score_group)
    print("\n=== External Scoring Example ===")
    scoring_agent = ScoringRLAgent()
    
    # In an async context, you could do:
    # scores = await scoring_agent.score_with_judge(['move_right', 'jump', 'move_right'], "openai/gpt-4")
    # print(f"Judge scores: {scores}")