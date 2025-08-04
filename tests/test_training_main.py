"""
Tests for training module __main__ entry point.
"""

import pytest
import sys
from unittest import mock
from io import StringIO
from vizra.training.__main__ import main


class TestTrainingMain:
    """Test training CLI entry point."""
    
    def test_main_no_args(self):
        """Test main with no arguments shows help."""
        with mock.patch('sys.argv', ['vizra.training']):
            with mock.patch('sys.stdout', new=StringIO()) as mock_stdout:
                result = main()
                
                output = mock_stdout.getvalue()
                assert 'usage:' in output.lower()
                assert 'vizra.training' in output
                assert result == 1
    
    def test_list_command_empty(self):
        """Test list command with no training routines."""
        with mock.patch('sys.argv', ['vizra.training', 'list']):
            with mock.patch('vizra.training.__main__.TrainingRunner') as mock_runner_class:
                mock_runner = mock_runner_class.return_value
                mock_runner.list_trainings.return_value = []
                
                with mock.patch('sys.stdout', new=StringIO()) as mock_stdout:
                    result = main()
                    
                    output = mock_stdout.getvalue()
                    assert 'No training routines found' in output
                    assert result == 0
                    mock_runner.list_trainings.assert_called_once()
    
    def test_list_command_with_trainings(self):
        """Test list command with training routines."""
        with mock.patch('sys.argv', ['vizra.training', 'list']):
            with mock.patch('vizra.training.__main__.TrainingRunner') as mock_runner_class:
                mock_runner = mock_runner_class.return_value
                mock_runner.list_trainings.return_value = [
                    {
                        'name': 'test_training',
                        'description': 'Test training',
                        'agent_name': 'test_agent',
                        'algorithm': 'ppo',
                        'class': 'test.TestTraining'
                    },
                    {
                        'name': 'another_training',
                        'description': 'Another training',
                        'agent_name': 'another_agent',
                        'algorithm': 'dqn',
                        'class': 'test.AnotherTraining'
                    }
                ]
                
                with mock.patch('sys.stdout', new=StringIO()) as mock_stdout:
                    result = main()
                    
                    output = mock_stdout.getvalue()
                    assert 'Available Training Routines' in output
                    assert 'test_training' in output
                    assert 'Test training' in output
                    assert 'test_agent' in output
                    assert 'ppo' in output
                    assert 'another_training' in output
                    assert 'Total: 2 training routines' in output
                    assert result == 0
    
    def test_run_command_success(self):
        """Test run command with successful training."""
        with mock.patch('sys.argv', ['vizra.training', 'run', 'test_training']):
            with mock.patch('vizra.training.__main__.TrainingRunner') as mock_runner_class:
                mock_runner = mock_runner_class.return_value
                mock_runner.run_training.return_value = {
                    'training_name': 'test_training',
                    'total_iterations': 100,
                    'best_reward': 0.95,
                    'final_metrics': {
                        'avg_reward': 0.9,
                        'success_rate': 0.85
                    }
                }
                
                result = main()
                
                assert result == 0
                mock_runner.run_training.assert_called_once_with('test_training', None)
    
    def test_run_command_with_iterations(self):
        """Test run command with custom iterations."""
        with mock.patch('sys.argv', ['vizra.training', 'run', 'test_training', '--iterations', '50']):
            with mock.patch('vizra.training.__main__.TrainingRunner') as mock_runner_class:
                mock_runner = mock_runner_class.return_value
                mock_runner.run_training.return_value = {
                    'training_name': 'test_training',
                    'total_iterations': 50,
                    'best_reward': 0.92
                }
                
                result = main()
                
                assert result == 0
                mock_runner.run_training.assert_called_once_with('test_training', 50)
    
    def test_run_command_with_report(self):
        """Test run command with report generation."""
        with mock.patch('sys.argv', ['vizra.training', 'run', 'test_training', '--report', 'output.txt']):
            with mock.patch('vizra.training.__main__.TrainingRunner') as mock_runner_class:
                mock_runner = mock_runner_class.return_value
                results = {
                    'training_name': 'test_training',
                    'total_iterations': 100,
                    'best_reward': 0.95
                }
                mock_runner.run_training.return_value = results
                
                result = main()
                
                assert result == 0
                mock_runner.run_training.assert_called_once_with('test_training', None)
                mock_runner.generate_report.assert_called_once_with(results, 'output.txt')
    
    def test_run_command_error(self):
        """Test run command with error."""
        with mock.patch('sys.argv', ['vizra.training', 'run', 'test_training']):
            with mock.patch('vizra.training.__main__.TrainingRunner') as mock_runner_class:
                mock_runner = mock_runner_class.return_value
                mock_runner.run_training.side_effect = Exception("Training not found")
                
                with mock.patch('sys.stdout', new=StringIO()) as mock_stdout:
                    result = main()
                    
                    output = mock_stdout.getvalue()
                    assert 'Error: Training not found' in output
                    assert result == 1
    
    def test_run_all_command_success(self):
        """Test run-all command."""
        with mock.patch('sys.argv', ['vizra.training', 'run-all']):
            with mock.patch('vizra.training.__main__.TrainingRunner') as mock_runner_class:
                mock_runner = mock_runner_class.return_value
                mock_runner.run_all_trainings.return_value = [
                    {
                        'training_name': 'training1',
                        'total_iterations': 50,
                        'best_reward': 0.8
                    },
                    {
                        'training_name': 'training2',
                        'total_iterations': 100,
                        'best_reward': 0.9
                    }
                ]
                
                result = main()
                
                assert result == 0
                mock_runner.run_all_trainings.assert_called_once_with(None)
    
    def test_run_all_command_with_iterations(self):
        """Test run-all command with custom iterations."""
        with mock.patch('sys.argv', ['vizra.training', 'run-all', '--iterations', '25']):
            with mock.patch('vizra.training.__main__.TrainingRunner') as mock_runner_class:
                mock_runner = mock_runner_class.return_value
                mock_runner.run_all_trainings.return_value = []
                
                result = main()
                
                assert result == 0
                mock_runner.run_all_trainings.assert_called_once_with(25)
    
    def test_run_all_command_with_report(self):
        """Test run-all command with report."""
        with mock.patch('sys.argv', ['vizra.training', 'run-all', '--report', 'all_results.txt']):
            with mock.patch('vizra.training.__main__.TrainingRunner') as mock_runner_class:
                mock_runner = mock_runner_class.return_value
                results = [{'training_name': 'training1'}]
                mock_runner.run_all_trainings.return_value = results
                
                result = main()
                
                assert result == 0
                mock_runner.run_all_trainings.assert_called_once()
                mock_runner.generate_report.assert_called_once_with(results, 'all_results.txt')
    
    def test_run_all_command_error(self):
        """Test run-all command with error."""
        with mock.patch('sys.argv', ['vizra.training', 'run-all']):
            with mock.patch('vizra.training.__main__.TrainingRunner') as mock_runner_class:
                mock_runner = mock_runner_class.return_value
                mock_runner.run_all_trainings.side_effect = Exception("Something went wrong")
                
                with mock.patch('sys.stdout', new=StringIO()) as mock_stdout:
                    result = main()
                    
                    output = mock_stdout.getvalue()
                    assert 'Error: Something went wrong' in output
                    assert result == 1
    
    def test_invalid_command(self):
        """Test invalid command shows help."""
        with mock.patch('sys.argv', ['vizra.training', 'invalid']):
            with mock.patch('sys.stderr', new=StringIO()) as mock_stderr:
                with pytest.raises(SystemExit):
                    main()
                
                error_output = mock_stderr.getvalue()
                assert 'invalid choice' in error_output.lower() or 'unrecognized' in error_output.lower()
    
    def test_invalid_iterations_value(self):
        """Test run command with invalid iterations value."""
        with mock.patch('sys.argv', ['vizra.training', 'run', 'test_training', '--iterations', 'not_a_number']):
            with mock.patch('sys.stderr', new=StringIO()) as mock_stderr:
                with pytest.raises(SystemExit):
                    main()
                
                error_output = mock_stderr.getvalue()
                assert 'invalid int value' in error_output.lower()