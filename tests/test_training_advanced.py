"""
Advanced tests for training base module.
"""

import pytest
import pandas as pd
import tempfile
import os
import json
from unittest import mock
from vizra.training.base import BaseRLTraining
from vizra import BaseAgent


class MockAgent(BaseAgent):
    """Mock agent for testing."""
    name = 'mock_agent'
    description = 'Mock agent'
    instructions = 'Test instructions'
    model = 'gpt-4o'
    
    @classmethod
    def run(cls, message: str, context=None) -> str:
        """Mock implementation."""
        if 'good' in message:
            return "Good response"
        elif 'bad' in message:
            return "Bad response"
        elif 'error' in message:
            raise Exception("Agent error")
        return f"Response to: {message}"


class MockTestTraining(BaseRLTraining):
    """Test training class."""
    name = 'test_training'
    description = 'Test training'
    agent_name = 'mock_agent'
    csv_path = 'test.csv'
    algorithm = 'ppo'
    learning_rate = 1e-5
    batch_size = 16
    n_iterations = 10
    
    def calculate_reward(self, csv_row_data: dict, agent_response: str) -> float:
        """Calculate reward based on response."""
        if 'good' in agent_response.lower():
            return 1.0
        elif 'bad' in agent_response.lower():
            return 0.0
        return 0.5


class TestTrainingAdvanced:
    """Advanced tests for training functionality."""
    
    def test_csv_loading_success(self):
        """Test successful CSV loading for training."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write('user_message,expected_response\n')
            f.write('Say good,Good response\n')
            f.write('Say bad,Bad response\n')
            csv_path = f.name
        
        try:
            class TempTraining(BaseRLTraining):
                name = 'temp'
                agent_name = 'mock'
                csv_path = csv_path
                
                def calculate_reward(self, csv_row_data, agent_response):
                    return 1.0
            
            training = TempTraining()
            df = training.load_training_data()
            
            assert len(df) == 2
            assert df.iloc[0]['user_message'] == 'Say good'
        finally:
            os.unlink(csv_path)
    
    def test_calculate_reward_implementation(self):
        """Test reward calculation."""
        training = MockTestTraining()
        
        # Test good response
        reward = training.calculate_reward(
            {'user_message': 'test'},
            'This is a good response'
        )
        assert reward == 1.0
        
        # Test bad response
        reward = training.calculate_reward(
            {'user_message': 'test'},
            'This is a bad response'
        )
        assert reward == 0.0
        
        # Test neutral response
        reward = training.calculate_reward(
            {'user_message': 'test'},
            'This is a neutral response'
        )
        assert reward == 0.5
    
    def test_prepare_trajectory(self):
        """Test trajectory preparation."""
        training = MockTestTraining()
        
        csv_row_data = {
            'user_message': 'Test message',
            'expected_response': 'Expected',
            'extra_field': 'Extra data'
        }
        
        trajectory = training.prepare_trajectory(csv_row_data)
        
        assert trajectory['user_message'] == 'Test message'
        assert 'agent_response' not in trajectory  # Not added yet
        assert 'reward' not in trajectory  # Not calculated yet
    
    def test_should_stop_early_default(self):
        """Test default early stopping logic."""
        training = MockTestTraining()
        
        # Should not stop with low reward
        metrics = {'avg_reward': 0.5, 'iteration': 5}
        assert not training.should_stop_early(metrics)
        
        # Should stop with very high reward
        metrics = {'avg_reward': 0.99, 'iteration': 5}
        assert training.should_stop_early(metrics)
        
        # Should not stop early in training
        metrics = {'avg_reward': 0.99, 'iteration': 2}
        assert not training.should_stop_early(metrics)
    
    def test_train_basic_flow(self):
        """Test basic training flow."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write('user_message,expected_response\n')
            f.write('Say good,Good response\n')
            csv_path = f.name
        
        try:
            class SimpleTraining(BaseRLTraining):
                name = 'simple'
                agent_name = 'mock_agent'
                csv_path = csv_path
                n_iterations = 2
                batch_size = 1
                
                def calculate_reward(self, csv_row_data, agent_response):
                    return 1.0 if 'good' in agent_response.lower() else 0.0
            
            with mock.patch('vizra.training.base.BaseRLTraining._get_agent_class') as mock_get_agent:
                mock_get_agent.return_value = MockAgent
                
                training = SimpleTraining()
                
                # Mock the RL provider
                with mock.patch.object(training, '_get_rl_provider') as mock_provider:
                    mock_rl = mock.Mock()
                    mock_rl.train.return_value = {
                        'final_metrics': {'avg_reward': 0.9},
                        'model_path': '/tmp/model.pt'
                    }
                    mock_provider.return_value = mock_rl
                    
                    results = training.train()
                    
                    assert results['training_name'] == 'simple'
                    assert results['total_iterations'] == 2
                    assert 'hyperparameters' in results
                    
                    # Verify RL provider was called
                    assert mock_rl.train.called
        finally:
            os.unlink(csv_path)
    
    def test_train_with_agent_error(self):
        """Test training when agent throws errors."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write('user_message,expected_response\n')
            f.write('Say error,Response\n')
            csv_path = f.name
        
        try:
            class ErrorTraining(BaseRLTraining):
                name = 'error'
                agent_name = 'mock_agent'
                csv_path = csv_path
                n_iterations = 1
                
                def calculate_reward(self, csv_row_data, agent_response):
                    return 0.0
            
            with mock.patch('vizra.training.base.BaseRLTraining._get_agent_class') as mock_get_agent:
                mock_get_agent.return_value = MockAgent
                
                training = ErrorTraining()
                
                with mock.patch.object(training, '_get_rl_provider') as mock_provider:
                    mock_rl = mock.Mock()
                    mock_provider.return_value = mock_rl
                    
                    # Training should handle agent errors gracefully
                    results = training.train()
                    
                    # Should complete despite errors
                    assert results['training_name'] == 'error'
        finally:
            os.unlink(csv_path)
    
    def test_get_agent_class(self):
        """Test agent class retrieval."""
        training = MockTestTraining()
        
        with mock.patch('vizra.training.base.BaseAgent.__subclasses__') as mock_subclasses:
            mock_subclasses.return_value = [MockAgent]
            
            agent_class = training._get_agent_class()
            assert agent_class == MockAgent
    
    def test_get_agent_class_not_found(self):
        """Test agent class not found."""
        class NoAgentTraining(BaseRLTraining):
            name = 'no_agent'
            agent_name = 'nonexistent'
            csv_path = 'test.csv'
            
            def calculate_reward(self, csv_row_data, agent_response):
                return 0.0
        
        training = NoAgentTraining()
        
        with mock.patch('vizra.training.base.BaseAgent.__subclasses__') as mock_subclasses:
            mock_subclasses.return_value = [MockAgent]
            
            with pytest.raises(ValueError) as exc_info:
                training._get_agent_class()
            
            assert 'Agent not found: nonexistent' in str(exc_info.value)
    
    def test_hyperparameters(self):
        """Test hyperparameter configuration."""
        training = MockTestTraining()
        
        assert training.algorithm == 'ppo'
        assert training.learning_rate == 1e-5
        assert training.batch_size == 16
        assert training.n_iterations == 10
        
        # Test custom hyperparameters
        class CustomTraining(BaseRLTraining):
            name = 'custom'
            agent_name = 'mock'
            csv_path = 'test.csv'
            algorithm = 'dqn'
            learning_rate = 0.001
            batch_size = 32
            n_iterations = 100
            temperature = 0.7  # Custom parameter
            
            def calculate_reward(self, csv_row_data, agent_response):
                return 1.0
        
        custom_training = CustomTraining()
        assert custom_training.algorithm == 'dqn'
        assert custom_training.temperature == 0.7
    
    def test_training_with_empty_csv(self):
        """Test training with empty CSV."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write('user_message,expected_response\n')
            csv_path = f.name
        
        try:
            class EmptyTraining(BaseRLTraining):
                name = 'empty'
                agent_name = 'mock'
                csv_path = csv_path
                
                def calculate_reward(self, csv_row_data, agent_response):
                    return 1.0
            
            training = EmptyTraining()
            
            with pytest.raises(ValueError) as exc_info:
                training.train()
            
            assert 'No training data' in str(exc_info.value) or 'empty' in str(exc_info.value).lower()
        finally:
            os.unlink(csv_path)
    
    def test_print_summary(self):
        """Test training summary printing."""
        training = MockTestTraining()
        
        results = {
            'training_name': 'test',
            'total_iterations': 100,
            'best_reward': 0.95,
            'final_metrics': {
                'avg_reward': 0.92,
                'success_rate': 0.88
            }
        }
        
        with mock.patch('builtins.print') as mock_print:
            training.print_summary(results)
            
            # Check summary was printed
            print_calls = [str(call) for call in mock_print.call_args_list]
            assert any('Training Results' in str(call) for call in print_calls)
            assert any('0.95' in str(call) for call in print_calls)