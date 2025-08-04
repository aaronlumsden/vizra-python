"""
Advanced tests for evaluation runner module.
"""

import pytest
import json
import tempfile
import os
from unittest import mock
from vizra.evaluation.runner import EvaluationRunner
from vizra.evaluation.base import BaseEvaluation
from vizra.evaluation.metrics import ContainsMetric


class MockEvaluation(BaseEvaluation):
    """Mock evaluation for testing."""
    name = 'mock_eval'
    description = 'Mock evaluation'
    agent_name = 'mock_agent'
    csv_path = 'test.csv'
    metrics = [ContainsMetric('expected')]


class AnotherMockEvaluation(BaseEvaluation):
    """Another mock evaluation."""
    name = 'another_eval'
    description = 'Another evaluation'
    agent_name = 'another_agent'
    csv_path = 'test2.csv'
    metrics = []


class TestEvaluationRunnerAdvanced:
    """Advanced tests for evaluation runner."""
    
    def test_find_evaluations(self):
        """Test finding evaluation classes."""
        runner = EvaluationRunner()
        
        # Mock the subclasses
        with mock.patch.object(BaseEvaluation, '__subclasses__') as mock_subclasses:
            mock_subclasses.return_value = [MockEvaluation, AnotherMockEvaluation]
            
            runner._find_evaluations()
            
            assert 'mock_eval' in runner.evaluations
            assert 'another_eval' in runner.evaluations
            assert runner.evaluations['mock_eval'] == MockEvaluation
            assert runner.evaluations['another_eval'] == AnotherMockEvaluation
    
    def test_list_evaluations(self):
        """Test listing evaluations."""
        runner = EvaluationRunner()
        
        with mock.patch.object(BaseEvaluation, '__subclasses__') as mock_subclasses:
            mock_subclasses.return_value = [MockEvaluation, AnotherMockEvaluation]
            
            evaluations = runner.list_evaluations()
            
            assert len(evaluations) == 2
            
            # Find mock_eval in results
            mock_eval_info = next(e for e in evaluations if e['name'] == 'mock_eval')
            assert mock_eval_info['description'] == 'Mock evaluation'
            assert mock_eval_info['agent_name'] == 'mock_agent'
            assert 'MockEvaluation' in mock_eval_info['class']
    
    def test_run_evaluation_success(self):
        """Test running a specific evaluation."""
        runner = EvaluationRunner()
        
        with mock.patch.object(BaseEvaluation, '__subclasses__') as mock_subclasses:
            mock_subclasses.return_value = [MockEvaluation]
            
            # Mock the evaluation run
            mock_results = {
                'evaluation_name': 'mock_eval',
                'total_cases': 5,
                'passed': 4,
                'failed': 1,
                'success_rate': 80.0
            }
            
            with mock.patch.object(MockEvaluation, 'run') as mock_run:
                mock_run.return_value = mock_results
                
                results = runner.run_evaluation('mock_eval')
                
                assert results == mock_results
                mock_run.assert_called_once()
    
    def test_run_evaluation_not_found(self):
        """Test running non-existent evaluation."""
        runner = EvaluationRunner()
        
        with mock.patch.object(BaseEvaluation, '__subclasses__') as mock_subclasses:
            mock_subclasses.return_value = [MockEvaluation]
            
            with pytest.raises(ValueError) as exc_info:
                runner.run_evaluation('nonexistent')
            
            assert "Evaluation 'nonexistent' not found" in str(exc_info.value)
    
    def test_run_all_evaluations(self):
        """Test running all evaluations."""
        runner = EvaluationRunner()
        
        with mock.patch.object(BaseEvaluation, '__subclasses__') as mock_subclasses:
            mock_subclasses.return_value = [MockEvaluation, AnotherMockEvaluation]
            
            # Mock results for each evaluation
            mock_results1 = {
                'evaluation_name': 'mock_eval',
                'total_cases': 5,
                'passed': 5,
                'failed': 0
            }
            mock_results2 = {
                'evaluation_name': 'another_eval',
                'total_cases': 3,
                'passed': 2,
                'failed': 1
            }
            
            with mock.patch.object(MockEvaluation, 'run') as mock_run1:
                with mock.patch.object(AnotherMockEvaluation, 'run') as mock_run2:
                    mock_run1.return_value = mock_results1
                    mock_run2.return_value = mock_results2
                    
                    all_results = runner.run_all_evaluations()
                    
                    assert len(all_results) == 2
                    assert all_results[0] == mock_results1
                    assert all_results[1] == mock_results2
    
    def test_run_all_evaluations_with_error(self):
        """Test running all evaluations with one failing."""
        runner = EvaluationRunner()
        
        with mock.patch.object(BaseEvaluation, '__subclasses__') as mock_subclasses:
            mock_subclasses.return_value = [MockEvaluation, AnotherMockEvaluation]
            
            mock_results1 = {'evaluation_name': 'mock_eval', 'total_cases': 5}
            
            with mock.patch.object(MockEvaluation, 'run') as mock_run1:
                with mock.patch.object(AnotherMockEvaluation, 'run') as mock_run2:
                    mock_run1.return_value = mock_results1
                    mock_run2.side_effect = Exception("Evaluation failed")
                    
                    # Should continue despite error
                    all_results = runner.run_all_evaluations()
                    
                    # Should have results from successful evaluation
                    assert len(all_results) >= 1
                    assert any(r['evaluation_name'] == 'mock_eval' for r in all_results)
    
    def test_generate_report_text(self):
        """Test text report generation."""
        runner = EvaluationRunner()
        
        results = {
            'evaluation_name': 'test_eval',
            'total_cases': 10,
            'passed': 8,
            'failed': 2,
            'success_rate': 80.0,
            'metrics_summary': {
                'metric1': {'passed': 8, 'total': 10}
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            report_path = f.name
        
        try:
            runner.generate_report(results, report_path)
            
            # Read and verify report
            with open(report_path, 'r') as f:
                content = f.read()
            
            assert 'Evaluation Report' in content
            assert 'test_eval' in content
            assert '80.0%' in content
            assert 'metric1' in content
        finally:
            os.unlink(report_path)
    
    def test_generate_report_json(self):
        """Test JSON report generation."""
        runner = EvaluationRunner()
        
        results = {
            'evaluation_name': 'test_eval',
            'total_cases': 10,
            'passed': 8,
            'failed': 2,
            'detailed_results': [
                {'case': 1, 'passed': True},
                {'case': 2, 'passed': False}
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            report_path = f.name
        
        try:
            runner.generate_report(results, report_path)
            
            # Read and verify JSON
            with open(report_path, 'r') as f:
                data = json.load(f)
            
            assert data['evaluation_name'] == 'test_eval'
            assert data['total_cases'] == 10
            assert len(data['detailed_results']) == 2
        finally:
            os.unlink(report_path)
    
    def test_generate_report_multiple_results(self):
        """Test report generation with multiple evaluation results."""
        runner = EvaluationRunner()
        
        results = [
            {
                'evaluation_name': 'eval1',
                'total_cases': 5,
                'passed': 5,
                'failed': 0
            },
            {
                'evaluation_name': 'eval2',
                'total_cases': 10,
                'passed': 7,
                'failed': 3
            }
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            report_path = f.name
        
        try:
            runner.generate_report(results, report_path)
            
            with open(report_path, 'r') as f:
                content = f.read()
            
            # Should contain both evaluation results
            assert 'eval1' in content
            assert 'eval2' in content
            assert 'Summary' in content
        finally:
            os.unlink(report_path)
    
    def test_runner_with_no_evaluations(self):
        """Test runner when no evaluations exist."""
        runner = EvaluationRunner()
        
        with mock.patch.object(BaseEvaluation, '__subclasses__') as mock_subclasses:
            mock_subclasses.return_value = []
            
            evaluations = runner.list_evaluations()
            assert evaluations == []
            
            with pytest.raises(ValueError):
                runner.run_evaluation('any_name')
    
    def test_evaluation_class_discovery(self):
        """Test automatic discovery of evaluation classes."""
        runner = EvaluationRunner()
        
        # Test that it finds evaluations in the examples directory
        # This tests the actual discovery mechanism
        runner._find_evaluations()
        
        # Should find at least the example evaluations if they exist
        # This is more of an integration test
        assert isinstance(runner.evaluations, dict)