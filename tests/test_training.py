"""
Tests for Vizra training framework.
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from vizra.training import BaseRLTraining, TrainingRunner
from vizra.agent import BaseAgent


class TestBaseRLTraining:
    """Test BaseRLTraining class."""
    
    def test_base_training_attributes(self):
        """Test base training has required attributes."""
        # Check class attributes
        assert hasattr(BaseRLTraining, 'name')
        assert hasattr(BaseRLTraining, 'description')
        assert hasattr(BaseRLTraining, 'agent_name')
        assert hasattr(BaseRLTraining, 'csv_path')
        assert hasattr(BaseRLTraining, 'algorithm')
        assert hasattr(BaseRLTraining, 'n_iterations')
        assert hasattr(BaseRLTraining, 'batch_size')
        assert hasattr(BaseRLTraining, 'learning_rate')
        
        assert BaseRLTraining.name == "base_training"
        assert BaseRLTraining.algorithm == "ppo"
        assert BaseRLTraining.n_iterations == 100
        assert BaseRLTraining.batch_size == 32
        assert BaseRLTraining.learning_rate == 1e-4
    
    def test_training_subclass(self):
        """Test creating a proper training subclass."""
        class TestTraining(BaseRLTraining):
            name = "test_training"
            agent_name = "test_agent"
            
            def _load_agent(self):
                # Override to prevent actual agent loading
                self.agent_class = Mock(spec=BaseAgent)
            
            def get_agent(self):
                return Mock(spec=BaseAgent)
            
            def calculate_reward(self, csv_row_data, agent_response):
                return 0.5
        
        training = TestTraining()
        assert training.name == "test_training"
        assert training.agent_name == "test_agent"
        assert hasattr(training, 'trajectories')
        assert hasattr(training, 'training_history')
    
    def test_calculate_reward_default(self):
        """Test default calculate_reward implementation."""
        class TestTraining(BaseRLTraining):
            agent_name = "test_agent"
            
            def _load_agent(self):
                pass
        
        training = TestTraining()
        
        # Test with matching expected response
        row_data = {"expected_response": "Hello World"}
        reward = training.calculate_reward(row_data, "This says Hello World!")
        assert reward == 1.0
        
        # Test with non-matching response
        row_data = {"expected_response": "Hello World"}
        reward = training.calculate_reward(row_data, "Goodbye")
        assert reward == 0.0
        
        # Test with no expected response
        row_data = {}
        reward = training.calculate_reward(row_data, "Any response")
        assert reward == 0.0


class TestTrainingRunner:
    """Test TrainingRunner class."""
    
    def test_list_trainings_empty(self):
        """Test listing trainings when none exist."""
        runner = TrainingRunner()
        runner.trainings = {}
        
        trainings = runner.list_trainings()
        assert trainings == []
    
    def test_list_trainings_with_data(self):
        """Test listing training routines."""
        runner = TrainingRunner()
        
        # Add mock training
        class MockTraining(BaseRLTraining):
            name = "test_training"
            description = "Test training"
            agent_name = "test_agent"
            algorithm = "ppo"
        
        runner.trainings = {"test_training": MockTraining}
        
        trainings = runner.list_trainings()
        
        assert len(trainings) == 1
        assert trainings[0]['name'] == "test_training"
        assert trainings[0]['description'] == "Test training"
        assert trainings[0]['agent_name'] == "test_agent"
        assert trainings[0]['algorithm'] == "ppo"
    
    def test_run_training_not_found(self):
        """Test running non-existent training."""
        runner = TrainingRunner()
        runner.trainings = {}
        
        with pytest.raises(ValueError, match="Training 'nonexistent' not found"):
            runner.run_training("nonexistent")
    
    def test_generate_report(self, tmp_path):
        """Test report generation."""
        runner = TrainingRunner()
        
        results = {
            'training_name': 'test_training',
            'total_iterations': 100,
            'best_reward': 0.95,
            'final_metrics': {
                'avg_reward': 0.9,
                'success_rate': 0.85
            }
        }
        
        report_path = tmp_path / "report.txt"
        runner.generate_report(results, str(report_path))
        
        assert report_path.exists()
        content = report_path.read_text()
        assert "test_training" in content
        assert "Best reward: 0.95" in content