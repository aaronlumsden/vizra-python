"""
Advanced tests for evaluation base module.
"""

import pytest
import pandas as pd
import tempfile
import os
from unittest import mock
from vizra.evaluation.base import BaseEvaluation
from vizra.evaluation.metrics import ContainsMetric, ExactMatchMetric
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
        if 'error' in message:
            raise Exception("Agent error")
        return f"Response to: {message}"


class MockTestEvaluation(BaseEvaluation):
    """Test evaluation class."""
    name = 'test_eval'
    description = 'Test evaluation'
    agent_name = 'mock_agent'
    csv_path = 'test.csv'
    metrics = [ContainsMetric('expected_response')]


class TestEvaluationAdvanced:
    """Advanced tests for evaluation functionality."""
    
    def test_csv_loading_success(self):
        """Test successful CSV loading."""
        # Create temporary CSV
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write('user_message,expected_response\n')
            f.write('Hello,Hi there\n')
            f.write('How are you?,I am fine\n')
            csv_path = f.name
        
        try:
            # Create evaluation with temp CSV
            class TempEval(BaseEvaluation):
                name = 'temp'
                agent_name = 'mock'
                csv_path = csv_path
                metrics = []
            
            eval_instance = TempEval()
            df = eval_instance.load_test_cases()
            
            assert len(df) == 2
            assert df.iloc[0]['user_message'] == 'Hello'
            assert df.iloc[1]['expected_response'] == 'I am fine'
        finally:
            os.unlink(csv_path)
    
    def test_csv_loading_missing_file(self):
        """Test CSV loading with missing file."""
        class MissingCSVEval(BaseEvaluation):
            name = 'missing'
            agent_name = 'mock'
            csv_path = 'nonexistent.csv'
            metrics = []
        
        eval_instance = MissingCSVEval()
        
        with pytest.raises(FileNotFoundError):
            eval_instance.load_test_cases()
    
    def test_csv_loading_invalid_format(self):
        """Test CSV loading with invalid format."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write('wrong_column\n')
            f.write('data\n')
            csv_path = f.name
        
        try:
            class InvalidCSVEval(BaseEvaluation):
                name = 'invalid'
                agent_name = 'mock'
                csv_path = csv_path
                metrics = []
            
            eval_instance = InvalidCSVEval()
            
            with pytest.raises(ValueError) as exc_info:
                eval_instance.load_test_cases()
            
            assert 'user_message' in str(exc_info.value)
        finally:
            os.unlink(csv_path)
    
    def test_run_evaluation_basic(self):
        """Test basic evaluation run."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write('user_message,expected_response\n')
            f.write('Test message,Response\n')
            csv_path = f.name
        
        try:
            class BasicEval(BaseEvaluation):
                name = 'basic'
                agent_name = 'mock_agent'
                csv_path = csv_path
                metrics = [ContainsMetric('expected_response')]
            
            # Mock the agent
            with mock.patch('vizra.evaluation.base.BaseEvaluation._get_agent_class') as mock_get_agent:
                mock_get_agent.return_value = MockAgent
                
                eval_instance = BasicEval()
                results = eval_instance.run()
                
                assert results['evaluation_name'] == 'basic'
                assert results['total_cases'] == 1
                assert 'metrics_summary' in results
                assert 'detailed_results' in results
        finally:
            os.unlink(csv_path)
    
    def test_run_evaluation_with_multiple_metrics(self):
        """Test evaluation with multiple metrics."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write('user_message,expected_response,exact_match\n')
            f.write('Hello,Hello back,Hello back\n')
            f.write('Test,Testing,Testing\n')
            csv_path = f.name
        
        try:
            class MultiMetricEval(BaseEvaluation):
                name = 'multi'
                agent_name = 'mock_agent'
                csv_path = csv_path
                metrics = [
                    ContainsMetric('expected_response'),
                    ExactMatchMetric('exact_match')
                ]
            
            with mock.patch('vizra.evaluation.base.BaseEvaluation._get_agent_class') as mock_get_agent:
                mock_get_agent.return_value = MockAgent
                
                eval_instance = MultiMetricEval()
                results = eval_instance.run()
                
                # Should have results for both metrics
                assert len(results['metrics_summary']) == 2
                assert 'contains_expected_response' in results['metrics_summary']
                assert 'exact_match_exact_match' in results['metrics_summary']
        finally:
            os.unlink(csv_path)
    
    def test_run_evaluation_with_agent_error(self):
        """Test evaluation when agent throws error."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write('user_message,expected_response\n')
            f.write('error message,Response\n')
            csv_path = f.name
        
        try:
            class ErrorEval(BaseEvaluation):
                name = 'error'
                agent_name = 'mock_agent'
                csv_path = csv_path
                metrics = [ContainsMetric('expected_response')]
            
            with mock.patch('vizra.evaluation.base.BaseEvaluation._get_agent_class') as mock_get_agent:
                mock_get_agent.return_value = MockAgent
                
                eval_instance = ErrorEval()
                results = eval_instance.run()
                
                # Should handle error gracefully
                assert results['total_cases'] == 1
                assert results['agent_errors'] == 1
                # Error cases should fail metrics
                detailed = results['detailed_results'][0]
                assert detailed['error'] is not None
        finally:
            os.unlink(csv_path)
    
    def test_get_agent_class(self):
        """Test agent class retrieval."""
        eval_instance = MockTestEvaluation()
        
        # Mock the agent registry
        with mock.patch('vizra.evaluation.base.BaseAgent.__subclasses__') as mock_subclasses:
            mock_subclasses.return_value = [MockAgent]
            
            agent_class = eval_instance._get_agent_class()
            assert agent_class == MockAgent
    
    def test_get_agent_class_not_found(self):
        """Test agent class not found."""
        class NoAgentEval(BaseEvaluation):
            name = 'no_agent'
            agent_name = 'nonexistent_agent'
            csv_path = 'test.csv'
            metrics = []
        
        eval_instance = NoAgentEval()
        
        with mock.patch('vizra.evaluation.base.BaseAgent.__subclasses__') as mock_subclasses:
            mock_subclasses.return_value = [MockAgent]
            
            with pytest.raises(ValueError) as exc_info:
                eval_instance._get_agent_class()
            
            assert 'Agent not found: nonexistent_agent' in str(exc_info.value)
    
    def test_evaluation_with_empty_csv(self):
        """Test evaluation with empty CSV."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write('user_message,expected_response\n')
            csv_path = f.name
        
        try:
            class EmptyEval(BaseEvaluation):
                name = 'empty'
                agent_name = 'mock_agent'
                csv_path = csv_path
                metrics = []
            
            eval_instance = EmptyEval()
            df = eval_instance.load_test_cases()
            
            assert len(df) == 0
        finally:
            os.unlink(csv_path)
    
    def test_print_summary(self):
        """Test summary printing."""
        eval_instance = MockTestEvaluation()
        
        results = {
            'evaluation_name': 'test',
            'total_cases': 10,
            'passed': 8,
            'failed': 2,
            'success_rate': 80.0,
            'metrics_summary': {
                'metric1': {'passed': 8, 'total': 10},
                'metric2': {'passed': 7, 'total': 10}
            }
        }
        
        with mock.patch('builtins.print') as mock_print:
            eval_instance.print_summary(results)
            
            # Check that summary information was printed
            print_calls = [str(call) for call in mock_print.call_args_list]
            assert any('Evaluation Results' in str(call) for call in print_calls)
            assert any('80.0%' in str(call) for call in print_calls)
    
    def test_csv_with_special_characters(self):
        """Test CSV loading with special characters."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
            f.write('user_message,expected_response\n')
            f.write('"Hello, world!","Hi, there!"\n')
            f.write('Test "quotes",Response with "quotes"\n')
            csv_path = f.name
        
        try:
            class SpecialCharEval(BaseEvaluation):
                name = 'special'
                agent_name = 'mock'
                csv_path = csv_path
                metrics = []
            
            eval_instance = SpecialCharEval()
            df = eval_instance.load_test_cases()
            
            assert len(df) == 2
            assert df.iloc[0]['user_message'] == 'Hello, world!'
            assert 'quotes' in df.iloc[1]['expected_response']
        finally:
            os.unlink(csv_path)