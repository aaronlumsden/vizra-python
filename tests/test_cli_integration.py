"""
Integration tests for Vizra CLI with evaluation and training.
"""

import pytest
import json
from pathlib import Path
from unittest import mock
from unittest.mock import Mock
from click.testing import CliRunner
from vizra.cli import cli
from vizra.evaluation import BaseEvaluation, EvaluationRunner
from vizra.evaluation.metrics import ContainsMetric
from vizra.training import BaseRLTraining, TrainingRunner
from vizra.agent import BaseAgent


class TestCLIIntegration:
    """Test CLI integration with real evaluation and training classes."""
    
    @pytest.fixture
    def setup_test_environment(self, tmp_path):
        """Set up test files and classes."""
        # Create test CSV for evaluation
        eval_csv = tmp_path / "eval_data.csv"
        eval_csv.write_text("prompt,expected\n\"What is 2+2?\",\"4\"\n\"What is 3+3?\",\"6\"")
        
        # Create test CSV for training
        train_csv = tmp_path / "train_data.csv"
        train_csv.write_text("prompt,expected\n\"Math question\",\"answer\"\n\"Science question\",\"result\"")
        
        # Define test evaluation class
        class TestEvaluation(BaseEvaluation):
            name = "integration_test_eval"
            description = "Integration test evaluation"
            agent_name = "test_agent"
            
            metrics = [
                ContainsMetric('expected')
            ]
            
            def __init__(self):
                self.csv_path = str(eval_csv)
                self.agent_class = None
                self._load_agent()
            
            def _load_agent(self):
                mock_agent = Mock(spec=BaseAgent)
                mock_agent.run.side_effect = ["4", "7"]  # First correct, second wrong
                self.agent_class = mock_agent
            
            def evaluate_row(self, csv_row_data, llm_response):
                # Base class will run metrics automatically
                return super().evaluate_row(csv_row_data, llm_response)
        
        # Define test training class
        class TestTraining(BaseRLTraining):
            name = "integration_test_training"
            description = "Integration test training"
            agent_name = "test_agent"
            algorithm = "test_ppo"
            
            def __init__(self):
                super().__init__()
                self.csv_path = str(train_csv)
                self.n_iterations = 2
                self.batch_size = 1
            
            def get_agent(self):
                mock_agent = Mock(spec=BaseAgent)
                mock_agent.chat.return_value = "response"
                return mock_agent
            
            def calculate_reward(self, csv_row_data, agent_response):
                return 0.75  # Fixed reward for testing
        
        return TestEvaluation, TestTraining, tmp_path
    
    def test_eval_run_integration(self, setup_test_environment):
        """Test running evaluation through CLI."""
        TestEvaluation, _, tmp_path = setup_test_environment
        
        runner = CliRunner()
        
        # Create a mock evaluation runner with our test evaluation
        with mock.patch('vizra.cli.eval.EvaluationRunner') as MockRunner:
            mock_instance = MockRunner.return_value
            mock_instance.evaluations = {"integration_test_eval": TestEvaluation}
            mock_instance.list_evaluations.return_value = [{
                'name': 'integration_test_eval',
                'description': 'Integration test evaluation',
                'agent_name': 'test_agent',
                'class': 'test.TestEvaluation'
            }]
            
            # Mock run_evaluation to actually run the evaluation
            def run_eval(name, limit=None):
                eval_class = TestEvaluation()
                # Simulate running evaluation
                return {
                    'evaluation_name': name,
                    'total_cases': 2,
                    'passed': 1,
                    'failed': 1,
                    'success_rate': 50.0,
                    'results': []
                }
            
            mock_instance.run_evaluation.side_effect = run_eval
            
            # Run evaluation
            result = runner.invoke(cli, ['eval', 'run', 'integration_test_eval'])
            
            # Check output
            assert result.exit_code == 1  # Exit 1 because one test failed
            assert "Running evaluation: integration_test_eval" in result.output
            assert "Total cases: 2" in result.output
            assert "Passed: 1" in result.output
            assert "Failed: 1" in result.output
            assert "Success rate: 50.0%" in result.output
    
    def test_train_run_integration(self, setup_test_environment):
        """Test running training through CLI."""
        _, TestTraining, tmp_path = setup_test_environment
        
        runner = CliRunner()
        
        # Create a mock training runner with our test training
        with mock.patch('vizra.cli.train.TrainingRunner') as MockRunner:
            mock_instance = MockRunner.return_value
            mock_instance.trainings = {"integration_test_training": TestTraining}
            mock_instance.list_trainings.return_value = [{
                'name': 'integration_test_training',
                'description': 'Integration test training',
                'agent_name': 'test_agent',
                'algorithm': 'test_ppo',
                'class': 'test.TestTraining'
            }]
            
            # Mock run_training
            mock_instance.run_training.return_value = {
                'training_name': 'integration_test_training',
                'total_iterations': 2,
                'best_reward': 0.75,
                'final_metrics': {
                    'avg_reward': 0.75,
                    'success_rate': 1.0
                },
                'hyperparameters': {
                    'algorithm': 'test_ppo',
                    'learning_rate': 0.001,
                    'batch_size': 1
                }
            }
            
            # Run training
            result = runner.invoke(cli, ['train', 'run', 'integration_test_training'])
            
            # Check output
            assert result.exit_code == 0
            assert "Starting training: integration_test_training" in result.output
            assert "Best reward:" in result.output
            assert "Training completed successfully!" in result.output
    
    def test_cli_with_empty_environment(self):
        """Test CLI commands when no evaluations or trainings exist."""
        runner = CliRunner()
        
        # Test empty evaluations
        with mock.patch('vizra.cli.eval.EvaluationRunner') as MockRunner:
            mock_instance = MockRunner.return_value
            mock_instance.list_evaluations.return_value = []
            
            # List evaluations
            result = runner.invoke(cli, ['eval', 'list'])
            assert result.exit_code == 0
            assert "No evaluations found" in result.output
            
            # Try to run non-existent evaluation
            result = runner.invoke(cli, ['eval', 'run', 'nonexistent'])
            assert result.exit_code == 1
            assert "Evaluation 'nonexistent' not found" in result.output
        
        # Test empty trainings
        with mock.patch('vizra.cli.train.TrainingRunner') as MockRunner:
            mock_instance = MockRunner.return_value
            mock_instance.list_trainings.return_value = []
            
            # List trainings
            result = runner.invoke(cli, ['train', 'list'])
            assert result.exit_code == 0
            assert "No training routines found" in result.output
            
            # Try to run non-existent training
            result = runner.invoke(cli, ['train', 'run', 'nonexistent'])
            assert result.exit_code == 1
            assert "Training 'nonexistent' not found" in result.output