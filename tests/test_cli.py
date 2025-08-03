"""
Tests for Vizra CLI.
"""

import pytest
from unittest import mock
from click.testing import CliRunner
from vizra.cli import cli
from vizra import __version__


class TestCLI:
    """Test CLI commands."""
    
    def test_version(self):
        """Test version command."""
        runner = CliRunner()
        result = runner.invoke(cli, ['--version'])
        
        assert result.exit_code == 0
        assert __version__ in result.output
    
    def test_help(self):
        """Test help command."""
        runner = CliRunner()
        result = runner.invoke(cli, ['--help'])
        
        assert result.exit_code == 0
        assert 'Vizra - AI Agent Framework' in result.output
        assert 'eval' in result.output
        assert 'train' in result.output
        assert 'status' in result.output
    
    def test_status(self):
        """Test status command."""
        runner = CliRunner()
        result = runner.invoke(cli, ['status'])
        
        assert result.exit_code == 0
        assert 'Vizra' in result.output
        assert 'installed and ready' in result.output
        assert 'Available Commands' in result.output  # Capital C in Rich output
    
    def test_eval_help(self):
        """Test eval help."""
        runner = CliRunner()
        result = runner.invoke(cli, ['eval', '--help'])
        
        assert result.exit_code == 0
        assert 'Run and manage evaluations' in result.output
        assert 'list' in result.output
        assert 'run' in result.output
    
    def test_train_help(self):
        """Test train help."""
        runner = CliRunner()
        result = runner.invoke(cli, ['train', '--help'])
        
        assert result.exit_code == 0
        assert 'Run and manage training' in result.output
        assert 'list' in result.output
        assert 'run' in result.output
    
    def test_eval_list_empty(self):
        """Test eval list with no evaluations."""
        runner = CliRunner()
        
        # Mock empty evaluation list
        with mock.patch('vizra.cli.eval.EvaluationRunner') as mock_runner:
            mock_instance = mock_runner.return_value
            mock_instance.list_evaluations.return_value = []
            
            result = runner.invoke(cli, ['eval', 'list'])
            
            assert result.exit_code == 0
            assert 'No evaluations found' in result.output
    
    def test_eval_list_with_evaluations(self):
        """Test eval list with evaluations."""
        runner = CliRunner()
        
        # Mock evaluation list
        with mock.patch('vizra.cli.eval.EvaluationRunner') as mock_runner:
            mock_instance = mock_runner.return_value
            mock_instance.list_evaluations.return_value = [
                {
                    'name': 'test_eval',
                    'description': 'Test evaluation',
                    'agent_name': 'test_agent',
                    'class': 'test.TestEval'
                }
            ]
            
            result = runner.invoke(cli, ['eval', 'list'])
            
            assert result.exit_code == 0
            assert 'test_eval' in result.output
            assert 'Test evaluation' in result.output
            assert 'test_agent' in result.output
    
    def test_eval_run_not_found(self):
        """Test eval run with non-existent evaluation."""
        runner = CliRunner()
        
        with mock.patch('vizra.cli.eval.EvaluationRunner') as mock_runner:
            mock_instance = mock_runner.return_value
            mock_instance.list_evaluations.return_value = []
            
            result = runner.invoke(cli, ['eval', 'run', 'nonexistent'])
            
            assert result.exit_code == 1
            assert "Evaluation 'nonexistent' not found" in result.output
    
    def test_eval_run_success(self):
        """Test successful eval run."""
        runner = CliRunner()
        
        with mock.patch('vizra.cli.eval.EvaluationRunner') as mock_runner:
            mock_instance = mock_runner.return_value
            mock_instance.list_evaluations.return_value = [
                {'name': 'test_eval', 'description': 'Test', 'agent_name': 'agent', 'class': 'test.Test'}
            ]
            mock_instance.run_evaluation.return_value = {
                'evaluation_name': 'test_eval',
                'total_cases': 10,
                'passed': 8,
                'failed': 2,
                'success_rate': 80.0,
                'results': []
            }
            
            result = runner.invoke(cli, ['eval', 'run', 'test_eval'])
            
            assert result.exit_code == 1  # Exit 1 because there are failures
            assert 'Running evaluation: test_eval' in result.output
            assert 'Total cases: 10' in result.output
            assert 'Passed: 8' in result.output
            assert 'Failed: 2' in result.output
            assert 'Success rate: 80.0%' in result.output
    
    def test_eval_run_with_output(self):
        """Test eval run with output file."""
        runner = CliRunner()
        
        with runner.isolated_filesystem():
            with mock.patch('vizra.cli.eval.EvaluationRunner') as mock_runner:
                mock_instance = mock_runner.return_value
                mock_instance.list_evaluations.return_value = [
                    {'name': 'test_eval', 'description': 'Test', 'agent_name': 'agent', 'class': 'test.Test'}
                ]
                mock_instance.run_evaluation.return_value = {
                    'evaluation_name': 'test_eval',
                    'total_cases': 10,
                    'passed': 10,
                    'failed': 0,
                    'success_rate': 100.0
                }
                
                result = runner.invoke(cli, ['eval', 'run', 'test_eval', '-o', 'results.json'])
                
                assert result.exit_code == 0
                assert 'Custom JSON output saved to: results.json' in result.output  # Updated message
                assert 'All tests passed!' in result.output
                
                # Check file was created
                import json
                with open('results.json') as f:
                    data = json.load(f)
                    assert data['evaluation_name'] == 'test_eval'
    
    def test_train_list_empty(self):
        """Test train list with no training routines."""
        runner = CliRunner()
        
        with mock.patch('vizra.cli.train.TrainingRunner') as mock_runner:
            mock_instance = mock_runner.return_value
            mock_instance.list_trainings.return_value = []
            
            result = runner.invoke(cli, ['train', 'list'])
            
            assert result.exit_code == 0
            assert 'No training routines found' in result.output
    
    def test_train_run_with_iterations(self):
        """Test train run with custom iterations."""
        runner = CliRunner()
        
        with mock.patch('vizra.cli.train.TrainingRunner') as mock_runner:
            mock_instance = mock_runner.return_value
            mock_instance.list_trainings.return_value = [
                {'name': 'test_training', 'description': 'Test', 'agent_name': 'agent', 'algorithm': 'ppo', 'class': 'test.Test'}
            ]
            
            # Mock the training class
            mock_train_class = mock.MagicMock()
            mock_train_class.n_iterations = 100
            mock_instance.trainings = {'test_training': mock_train_class}
            
            mock_instance.run_training.return_value = {
                'training_name': 'test_training',
                'total_iterations': 20,
                'best_reward': 0.95,
                'final_metrics': {
                    'avg_reward': 0.9,
                    'success_rate': 0.85
                },
                'hyperparameters': {
                    'algorithm': 'ppo',
                    'learning_rate': 0.001,
                    'batch_size': 32
                }
            }
            
            result = runner.invoke(cli, ['train', 'run', 'test_training', '-i', '20'])
            
            assert result.exit_code == 0
            assert 'Overriding iterations: 100 → 20' in result.output
            assert 'Best reward: 0.950' in result.output
            assert 'Training completed successfully!' in result.output