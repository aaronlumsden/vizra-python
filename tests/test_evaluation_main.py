"""
Tests for evaluation module __main__ entry point.
"""

import pytest
import sys
from unittest import mock
from io import StringIO
from vizra.evaluation.__main__ import main


class TestEvaluationMain:
    """Test evaluation CLI entry point."""
    
    def test_main_no_args(self):
        """Test main with no arguments shows help."""
        with mock.patch('sys.argv', ['vizra.evaluation']):
            with mock.patch('sys.stdout', new=StringIO()) as mock_stdout:
                result = main()
                
                output = mock_stdout.getvalue()
                assert 'usage:' in output.lower()
                assert 'vizra.evaluation' in output
                assert result == 1
    
    def test_list_command_empty(self):
        """Test list command with no evaluations."""
        with mock.patch('sys.argv', ['vizra.evaluation', 'list']):
            with mock.patch('vizra.evaluation.__main__.EvaluationRunner') as mock_runner_class:
                mock_runner = mock_runner_class.return_value
                mock_runner.list_evaluations.return_value = []
                
                with mock.patch('sys.stdout', new=StringIO()) as mock_stdout:
                    result = main()
                    
                    output = mock_stdout.getvalue()
                    assert 'No evaluations found' in output
                    assert result == 0
                    mock_runner.list_evaluations.assert_called_once()
    
    def test_list_command_with_evaluations(self):
        """Test list command with evaluations."""
        with mock.patch('sys.argv', ['vizra.evaluation', 'list']):
            with mock.patch('vizra.evaluation.__main__.EvaluationRunner') as mock_runner_class:
                mock_runner = mock_runner_class.return_value
                mock_runner.list_evaluations.return_value = [
                    {
                        'name': 'test_eval',
                        'description': 'Test evaluation',
                        'agent_name': 'test_agent',
                        'class': 'test.TestEval'
                    },
                    {
                        'name': 'another_eval',
                        'description': 'Another test',
                        'agent_name': 'another_agent',
                        'class': 'test.AnotherEval'
                    }
                ]
                
                with mock.patch('sys.stdout', new=StringIO()) as mock_stdout:
                    result = main()
                    
                    output = mock_stdout.getvalue()
                    assert 'Available Evaluations' in output
                    assert 'test_eval' in output
                    assert 'Test evaluation' in output
                    assert 'test_agent' in output
                    assert 'another_eval' in output
                    assert 'Total: 2 evaluations' in output
                    assert result == 0
    
    def test_run_command_success(self):
        """Test run command with successful evaluation."""
        with mock.patch('sys.argv', ['vizra.evaluation', 'run', 'test_eval']):
            with mock.patch('vizra.evaluation.__main__.EvaluationRunner') as mock_runner_class:
                mock_runner = mock_runner_class.return_value
                mock_runner.run_evaluation.return_value = {
                    'evaluation_name': 'test_eval',
                    'total_cases': 10,
                    'passed': 8,
                    'failed': 2,
                    'success_rate': 80.0
                }
                
                result = main()
                
                assert result == 0
                mock_runner.run_evaluation.assert_called_once_with('test_eval')
    
    def test_run_command_with_report(self):
        """Test run command with report generation."""
        with mock.patch('sys.argv', ['vizra.evaluation', 'run', 'test_eval', '--report', 'output.txt']):
            with mock.patch('vizra.evaluation.__main__.EvaluationRunner') as mock_runner_class:
                mock_runner = mock_runner_class.return_value
                results = {
                    'evaluation_name': 'test_eval',
                    'total_cases': 10,
                    'passed': 10,
                    'failed': 0
                }
                mock_runner.run_evaluation.return_value = results
                
                result = main()
                
                assert result == 0
                mock_runner.run_evaluation.assert_called_once_with('test_eval')
                mock_runner.generate_report.assert_called_once_with(results, 'output.txt')
    
    def test_run_command_error(self):
        """Test run command with error."""
        with mock.patch('sys.argv', ['vizra.evaluation', 'run', 'test_eval']):
            with mock.patch('vizra.evaluation.__main__.EvaluationRunner') as mock_runner_class:
                mock_runner = mock_runner_class.return_value
                mock_runner.run_evaluation.side_effect = Exception("Evaluation not found")
                
                with mock.patch('sys.stdout', new=StringIO()) as mock_stdout:
                    result = main()
                    
                    output = mock_stdout.getvalue()
                    assert 'Error: Evaluation not found' in output
                    assert result == 1
    
    def test_run_all_command_success(self):
        """Test run-all command."""
        with mock.patch('sys.argv', ['vizra.evaluation', 'run-all']):
            with mock.patch('vizra.evaluation.__main__.EvaluationRunner') as mock_runner_class:
                mock_runner = mock_runner_class.return_value
                mock_runner.run_all_evaluations.return_value = [
                    {
                        'evaluation_name': 'eval1',
                        'total_cases': 5,
                        'passed': 5,
                        'failed': 0
                    },
                    {
                        'evaluation_name': 'eval2',
                        'total_cases': 10,
                        'passed': 8,
                        'failed': 2
                    }
                ]
                
                result = main()
                
                assert result == 0
                mock_runner.run_all_evaluations.assert_called_once()
    
    def test_run_all_command_with_report(self):
        """Test run-all command with report."""
        with mock.patch('sys.argv', ['vizra.evaluation', 'run-all', '--report', 'all_results.txt']):
            with mock.patch('vizra.evaluation.__main__.EvaluationRunner') as mock_runner_class:
                mock_runner = mock_runner_class.return_value
                results = [{'evaluation_name': 'eval1'}]
                mock_runner.run_all_evaluations.return_value = results
                
                result = main()
                
                assert result == 0
                mock_runner.run_all_evaluations.assert_called_once()
                mock_runner.generate_report.assert_called_once_with(results, 'all_results.txt')
    
    def test_run_all_command_error(self):
        """Test run-all command with error."""
        with mock.patch('sys.argv', ['vizra.evaluation', 'run-all']):
            with mock.patch('vizra.evaluation.__main__.EvaluationRunner') as mock_runner_class:
                mock_runner = mock_runner_class.return_value
                mock_runner.run_all_evaluations.side_effect = Exception("Something went wrong")
                
                with mock.patch('sys.stdout', new=StringIO()) as mock_stdout:
                    result = main()
                    
                    output = mock_stdout.getvalue()
                    assert 'Error: Something went wrong' in output
                    assert result == 1
    
    def test_invalid_command(self):
        """Test invalid command shows help."""
        with mock.patch('sys.argv', ['vizra.evaluation', 'invalid']):
            with mock.patch('sys.stderr', new=StringIO()) as mock_stderr:
                with pytest.raises(SystemExit):
                    main()
                
                error_output = mock_stderr.getvalue()
                assert 'invalid choice' in error_output.lower() or 'unrecognized' in error_output.lower()