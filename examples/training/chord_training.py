"""
Example RL training for a chord identifier agent.
"""

from vizra.training import BaseRLTraining
from examples.evaluations.chord_eval import ChordIdentifierAgent


class ChordIdentifierTraining(BaseRLTraining):
    """
    RL training for improving chord identification accuracy.
    """
    name = 'chord_identifier_training'
    description = 'Train chord identifier agent using reinforcement learning'
    agent_name = 'chord_identifier'
    csv_path = 'examples/data/chord_tests.csv'
    
    # Training hyperparameters
    algorithm = 'ppo'
    learning_rate = 5e-5
    batch_size = 16
    n_iterations = 50
    
    def calculate_reward(self, csv_row_data: dict, agent_response: str) -> float:
        """
        Calculate reward based on response accuracy.
        
        Returns a value between 0.0 and 1.0.
        """
        expected = csv_row_data.get('expected_response', '').lower()
        response_lower = agent_response.lower()
        
        # Full reward for exact match
        if expected in response_lower:
            reward = 1.0
        else:
            # Partial rewards based on components
            reward = 0.0
            
            # Extract chord components
            expected_parts = expected.split()
            
            # Check root note (most important)
            if expected_parts and expected_parts[0] in response_lower:
                reward += 0.4
            
            # Check chord quality
            if 'major' in expected and 'major' in response_lower:
                reward += 0.3
            elif 'minor' in expected and 'minor' in response_lower:
                reward += 0.3
            elif '7' in expected and ('7' in response_lower or 'seventh' in response_lower):
                reward += 0.3
            
            # Penalty for wrong chord type
            if 'major' in expected and 'minor' in response_lower and 'minor' not in expected:
                reward -= 0.2
            elif 'minor' in expected and 'major' in response_lower and 'major' not in expected:
                reward -= 0.2
        
        # Bonus for clear, confident responses
        if any(phrase in response_lower for phrase in ["that's a", "this is a", "the chord is"]):
            reward += 0.1
        
        # Ensure reward is in valid range
        return max(0.0, min(1.0, reward))
    
    def prepare_trajectory(self, csv_row_data: dict) -> dict:
        """
        Prepare training trajectory with additional context.
        """
        trajectory = super().prepare_trajectory(csv_row_data)
        
        # Add difficulty level as context
        difficulty = csv_row_data.get('difficulty', 'medium')
        trajectory['difficulty'] = difficulty
        
        # Add test type
        test_type = csv_row_data.get('test_type', 'basic')
        trajectory['test_type'] = test_type
        
        return trajectory
    
    def should_stop_early(self, metrics: dict) -> bool:
        """
        Custom early stopping for chord training.
        """
        # Stop if we achieve very high accuracy
        if metrics.get('avg_reward', 0) > 0.98:
            return True
        
        # Stop if success rate is very high
        if metrics.get('success_rate', 0) > 0.95:
            return True
        
        # Otherwise use default logic
        return super().should_stop_early(metrics)