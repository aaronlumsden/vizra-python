"""
Tests for Vizra evaluation framework.
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from vizra.evaluation import BaseEvaluation, EvaluationRunner
from vizra.evaluation.metrics import ContainsMetric, ExactMatchMetric
from vizra.agent import BaseAgent


class TestBaseEvaluation:
    """Test BaseEvaluation class."""
    
    def test_base_evaluation_attributes(self):
        """Test base evaluation has required attributes."""
        # Check class attributes directly without instantiating
        assert hasattr(BaseEvaluation, 'name')
        assert hasattr(BaseEvaluation, 'description')
        assert hasattr(BaseEvaluation, 'agent_name')
        assert hasattr(BaseEvaluation, 'csv_path')
        
        assert BaseEvaluation.name == "base_evaluation"
        assert BaseEvaluation.description == "Base evaluation class"
        assert BaseEvaluation.agent_name == ""
        assert BaseEvaluation.csv_path == ""
    
    def test_evaluation_subclass(self):
        """Test creating a proper evaluation subclass."""
        class TestEval(BaseEvaluation):
            name = "test_eval"
            agent_name = "test_agent"
            
            def _load_agent(self):
                # Override to prevent actual agent loading
                self.agent_class = Mock(spec=BaseAgent)
                self.agent_class.run.return_value = "test response"
        
        eval_instance = TestEval()
        assert eval_instance.name == "test_eval"
        assert eval_instance.agent_name == "test_agent"
        assert hasattr(eval_instance, 'metrics')
    
    def test_prepare_prompt(self):
        """Test prepare_prompt method."""
        class TestEval(BaseEvaluation):
            agent_name = "test_agent"
            
            def _load_agent(self):
                pass
        
        eval_instance = TestEval()
        
        # Test with 'prompt' column (default)
        row_data = {"prompt": "What is 2+2?", "expected": "4"}
        prompt = eval_instance.prepare_prompt(row_data)
        assert prompt == "What is 2+2?"
        
        # Test missing prompt column
        row_data = {"question": "What is 3+3?", "expected": "6"}
        with pytest.raises(KeyError):
            eval_instance.prepare_prompt(row_data)
    
    def test_metrics_integration(self):
        """Test metrics integration."""
        class TestEval(BaseEvaluation):
            agent_name = "test_agent"
            metrics = [
                ContainsMetric('expected'),
                ExactMatchMetric('expected')
            ]
            
            def _load_agent(self):
                pass
        
        eval_instance = TestEval()
        
        # Test evaluate_row with metrics
        csv_data = {'expected': 'Hello', 'prompt': 'test'}
        result = eval_instance.evaluate_row(csv_data, "Hello world")
        
        assert 'metrics' in result
        assert 'contains_expected' in result['metrics']
        assert 'exact_match_expected' in result['metrics']
        
        # Contains should pass
        assert result['metrics']['contains_expected']['passed'] is True
        
        # Exact match should fail
        assert result['metrics']['exact_match_expected']['passed'] is False
        
        # Overall should fail since not all metrics passed
        assert result['passed'] is False
    
    def test_evaluate_row(self):
        """Test evaluate_row method."""
        class TestEval(BaseEvaluation):
            agent_name = "test_agent"
            
            def _load_agent(self):
                self.agent_class = Mock(spec=BaseAgent)
                self.agent_class.run.return_value = "4"
                
            metrics = [
                ContainsMetric('expected')
            ]
            
            def evaluate_row(self, csv_row_data, llm_response):
                # Call parent evaluate_row which runs metrics
                return super().evaluate_row(csv_row_data, llm_response)
        
        eval_instance = TestEval()
        row_data = {"prompt": "What is 2+2?", "expected": "4"}
        
        result = eval_instance.evaluate_row(row_data, "4")
        
        assert result['passed'] is True
        assert 'metrics' in result
        assert result['metrics']['contains_expected']['passed'] is True


class TestMetrics:
    """Test metrics functionality."""
    
    def test_contains_metric(self):
        """Test ContainsMetric."""
        metric = ContainsMetric('expected')
        
        row_data = {'expected': 'world'}
        result = metric.evaluate(row_data, "Hello world")
        assert result['passed'] is True
        assert result['score'] == 1.0
        
        result = metric.evaluate(row_data, "Hello universe")
        assert result['passed'] is False
        assert result['score'] == 0.0
    
    def test_exact_match_metric(self):
        """Test ExactMatchMetric."""
        metric = ExactMatchMetric('expected')
        
        row_data = {'expected': 'Hello'}
        result = metric.evaluate(row_data, "Hello")
        assert result['passed'] is True
        assert result['score'] == 1.0
        
        result = metric.evaluate(row_data, "Hello world")
        assert result['passed'] is False
        assert result['score'] == 0.0
        
        # Test case sensitivity
        metric_case = ExactMatchMetric('expected', case_sensitive=True)
        result = metric_case.evaluate(row_data, "hello")
        assert result['passed'] is False


class TestEvaluationRunner:
    """Test EvaluationRunner class."""
    
    def test_list_evaluations_empty(self):
        """Test listing evaluations when none exist."""
        runner = EvaluationRunner()
        runner.evaluations = {}
        
        evals = runner.list_evaluations()
        assert evals == []
    
    def test_list_evaluations_with_data(self):
        """Test listing evaluations."""
        runner = EvaluationRunner()
        
        # Add mock evaluation
        class MockEval(BaseEvaluation):
            name = "test_eval"
            description = "Test evaluation"
            agent_name = "test_agent"
        
        runner.evaluations = {"test_eval": MockEval}
        
        evals = runner.list_evaluations()
        
        assert len(evals) == 1
        assert evals[0]['name'] == "test_eval"
        assert evals[0]['description'] == "Test evaluation"
        assert evals[0]['agent_name'] == "test_agent"
    
    def test_run_evaluation_not_found(self):
        """Test running non-existent evaluation."""
        runner = EvaluationRunner()
        runner.evaluations = {}
        
        with pytest.raises(ValueError, match="Evaluation 'nonexistent' not found"):
            runner.run_evaluation("nonexistent")
    
    def test_generate_report(self, tmp_path):
        """Test report generation."""
        runner = EvaluationRunner()
        
        results = {
            'evaluation_name': 'test_eval',
            'total_cases': 10,
            'passed': 8,
            'failed': 2,
            'success_rate': 80.0,
            'results': []
        }
        
        report_path = tmp_path / "report.txt"
        runner.generate_report(results, str(report_path))
        
        assert report_path.exists()
        content = report_path.read_text()
        assert "test_eval" in content
        assert "Success rate: 80.0%" in content  # lowercase 'rate'