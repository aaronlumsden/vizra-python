"""
Metrics for evaluating agent responses.

Metrics provide a flexible way to evaluate agent responses against expected outcomes.
They can be used in both evaluation and reinforcement learning contexts.
"""

import re
from typing import Dict, Any, Optional, Union, List
from abc import ABC, abstractmethod


class BaseMetric(ABC):
    """
    Base class for evaluation metrics.
    
    Subclass this to create custom metrics for evaluating agent responses.
    """
    
    name = "base_metric"  # Override in subclasses
    
    @abstractmethod
    def evaluate(self, row_data: Dict[str, Any], response: str) -> Dict[str, Any]:
        """
        Evaluate the response against the metric criteria.
        
        Args:
            row_data: Dictionary containing CSV row data
            response: The agent's response
            
        Returns:
            dict: Result dictionary with at least:
                - passed: bool indicating if metric passed
                - score: numeric score (0-1)
                - details: dict with additional information
        """
        pass


class TrainableMetric(BaseMetric):
    """
    Extended metric that can be used for reinforcement learning.
    
    Adds reward computation capabilities for training agents.
    """
    
    @abstractmethod
    def compute_reward(self, result: Dict[str, Any]) -> float:
        """
        Compute a reward value from the metric result.
        
        Args:
            result: The result from evaluate()
            
        Returns:
            float: Reward value for RL training
        """
        pass


class ExactMatchMetric(BaseMetric):
    """Checks if response exactly matches expected value."""
    
    def __init__(self, expected_column: str = 'expected_output', case_sensitive: bool = False):
        self.expected_column = expected_column
        self.case_sensitive = case_sensitive
        self.name = f"exact_match_{expected_column}"
    
    def evaluate(self, row_data: Dict[str, Any], response: str) -> Dict[str, Any]:
        expected = str(row_data.get(self.expected_column, ''))
        
        if self.case_sensitive:
            passed = response.strip() == expected.strip()
        else:
            passed = response.lower().strip() == expected.lower().strip()
        
        return {
            'passed': passed,
            'score': 1.0 if passed else 0.0,
            'details': {
                'expected': expected,
                'actual': response[:100] + '...' if len(response) > 100 else response,
                'case_sensitive': self.case_sensitive
            }
        }


class ContainsMetric(BaseMetric):
    """Checks if response contains expected substring."""
    
    def __init__(self, expected_column: str = 'expected_output', case_sensitive: bool = False):
        self.expected_column = expected_column
        self.case_sensitive = case_sensitive
        self.name = f"contains_{expected_column}"
    
    def evaluate(self, row_data: Dict[str, Any], response: str) -> Dict[str, Any]:
        expected = str(row_data.get(self.expected_column, ''))
        
        if self.case_sensitive:
            passed = expected in response
        else:
            passed = expected.lower() in response.lower()
        
        return {
            'passed': passed,
            'score': 1.0 if passed else 0.0,
            'details': {
                'expected_substring': expected,
                'found': passed,
                'response_preview': response[:100] + '...' if len(response) > 100 else response
            }
        }


class NotContainsMetric(BaseMetric):
    """Checks that response does NOT contain specified text."""
    
    def __init__(self, column_or_text: str, is_column: bool = True, case_sensitive: bool = False):
        """
        Args:
            column_or_text: Column name or literal text to check
            is_column: If True, column_or_text is a column name; if False, it's literal text
            case_sensitive: Whether to do case-sensitive matching
        """
        self.column_or_text = column_or_text
        self.is_column = is_column
        self.case_sensitive = case_sensitive
        self.name = f"not_contains_{column_or_text}" if is_column else f"not_contains_text"
    
    def evaluate(self, row_data: Dict[str, Any], response: str) -> Dict[str, Any]:
        if self.is_column:
            unexpected = str(row_data.get(self.column_or_text, ''))
        else:
            unexpected = self.column_or_text
        
        if self.case_sensitive:
            passed = unexpected not in response
        else:
            passed = unexpected.lower() not in response.lower()
        
        return {
            'passed': passed,
            'score': 1.0 if passed else 0.0,
            'details': {
                'unexpected_text': unexpected,
                'found': not passed,
                'response_preview': response[:100] + '...' if len(response) > 100 else response
            }
        }


class RegexMetric(BaseMetric):
    """Evaluates response against a regular expression pattern."""
    
    def __init__(self, pattern: str, name: Optional[str] = None, flags: int = 0):
        """
        Args:
            pattern: Regular expression pattern
            name: Optional custom name for the metric
            flags: Regular expression flags (e.g., re.IGNORECASE)
        """
        self.pattern = pattern
        self.compiled_pattern = re.compile(pattern, flags)
        self.name = name or f"regex_{pattern[:20]}"  # Truncate long patterns
    
    def evaluate(self, row_data: Dict[str, Any], response: str) -> Dict[str, Any]:
        match = self.compiled_pattern.search(response)
        passed = match is not None
        
        return {
            'passed': passed,
            'score': 1.0 if passed else 0.0,
            'details': {
                'pattern': self.pattern,
                'matched': match.group(0) if match else None,
                'response_preview': response[:100] + '...' if len(response) > 100 else response
            }
        }


class SentimentMetric(BaseMetric):
    """Evaluates sentiment of the response."""
    
    name = "sentiment"
    
    def __init__(self, expected_sentiment: str = 'positive'):
        """
        Args:
            expected_sentiment: 'positive', 'negative', or 'neutral'
        """
        self.expected_sentiment = expected_sentiment.lower()
        if self.expected_sentiment not in ['positive', 'negative', 'neutral']:
            raise ValueError("expected_sentiment must be 'positive', 'negative', or 'neutral'")
    
    def evaluate(self, row_data: Dict[str, Any], response: str) -> Dict[str, Any]:
        # Simple sentiment analysis - in production use TextBlob or similar
        positive_words = ['good', 'great', 'excellent', 'happy', 'pleased', 'wonderful', 
                         'fantastic', 'amazing', 'love', 'best', 'perfect']
        negative_words = ['bad', 'terrible', 'awful', 'sad', 'angry', 'disappointed',
                         'worst', 'hate', 'horrible', 'poor', 'failed']
        
        response_lower = response.lower()
        positive_count = sum(1 for word in positive_words if word in response_lower)
        negative_count = sum(1 for word in negative_words if word in response_lower)
        
        if positive_count > negative_count:
            detected_sentiment = 'positive'
        elif negative_count > positive_count:
            detected_sentiment = 'negative'
        else:
            detected_sentiment = 'neutral'
        
        passed = detected_sentiment == self.expected_sentiment
        
        # Score based on confidence (difference between positive and negative)
        confidence = abs(positive_count - negative_count) / max(positive_count + negative_count, 1)
        score = confidence if passed else 1 - confidence
        
        return {
            'passed': passed,
            'score': score,
            'details': {
                'expected_sentiment': self.expected_sentiment,
                'detected_sentiment': detected_sentiment,
                'positive_count': positive_count,
                'negative_count': negative_count,
                'confidence': confidence
            }
        }


class LengthMetric(BaseMetric):
    """Evaluates response length against constraints."""
    
    def __init__(self, min_length: Optional[int] = None, max_length: Optional[int] = None,
                 count_type: str = 'chars'):
        """
        Args:
            min_length: Minimum required length
            max_length: Maximum allowed length
            count_type: 'chars' for characters, 'words' for words, 'lines' for lines
        """
        self.min_length = min_length
        self.max_length = max_length
        self.count_type = count_type
        self.name = f"length_{count_type}"
    
    def evaluate(self, row_data: Dict[str, Any], response: str) -> Dict[str, Any]:
        if self.count_type == 'chars':
            actual_length = len(response)
        elif self.count_type == 'words':
            actual_length = len(response.split())
        elif self.count_type == 'lines':
            actual_length = len(response.strip().split('\n'))
        else:
            raise ValueError(f"Invalid count_type: {self.count_type}")
        
        passed = True
        issues = []
        
        if self.min_length is not None and actual_length < self.min_length:
            passed = False
            issues.append(f"too short (min: {self.min_length})")
        
        if self.max_length is not None and actual_length > self.max_length:
            passed = False
            issues.append(f"too long (max: {self.max_length})")
        
        # Score based on how far off the length is
        score = 1.0
        if not passed:
            if self.min_length and actual_length < self.min_length:
                score = actual_length / self.min_length
            elif self.max_length and actual_length > self.max_length:
                score = self.max_length / actual_length
        
        return {
            'passed': passed,
            'score': score,
            'details': {
                'actual_length': actual_length,
                'min_length': self.min_length,
                'max_length': self.max_length,
                'count_type': self.count_type,
                'issues': issues
            }
        }


# Trainable versions of metrics for RL
class TrainableExactMatchMetric(ExactMatchMetric, TrainableMetric):
    """Exact match metric with reward computation."""
    
    def compute_reward(self, result: Dict[str, Any]) -> float:
        """Simple binary reward: 1.0 for match, -1.0 for no match."""
        return 1.0 if result['passed'] else -1.0


class TrainableContainsMetric(ContainsMetric, TrainableMetric):
    """Contains metric with reward computation."""
    
    def compute_reward(self, result: Dict[str, Any]) -> float:
        """Binary reward with partial credit considered in future versions."""
        return 1.0 if result['passed'] else -0.5


class TrainableSentimentMetric(SentimentMetric, TrainableMetric):
    """Sentiment metric with reward computation."""
    
    def compute_reward(self, result: Dict[str, Any]) -> float:
        """Reward based on sentiment match and confidence."""
        base_reward = 1.0 if result['passed'] else -1.0
        confidence = result['details']['confidence']
        return base_reward * (0.5 + 0.5 * confidence)  # Scale by confidence


class ToolUsageMetric(BaseMetric):
    """Checks if specific tools were used during the agent interaction."""
    
    def __init__(self, expected_tools: Union[str, List[str]], require_all: bool = False):
        """
        Args:
            expected_tools: Tool name(s) to check for
                - For OpenAI: the function name from definition
                - For XML: the xml_tag value
            require_all: If True, ALL listed tools must be used
                        If False, ANY listed tool counts as success
        """
        self.expected_tools = [expected_tools] if isinstance(expected_tools, str) else expected_tools
        self.require_all = require_all
        self.name = f"tool_usage_{'_'.join(self.expected_tools)}"
    
    def evaluate(self, row_data: Dict[str, Any], response: str) -> Dict[str, Any]:
        tools_used = set()
        tool_calls_details = []
        
        if 'conversation_history' in row_data:
            history = row_data.get('conversation_history', [])
            
            for msg in history:
                if isinstance(msg, dict):
                    # OpenAI-style tool calls
                    if msg.get('role') == 'assistant' and 'tool_calls' in msg:
                        for call in msg['tool_calls']:
                            tool_name = call.get('function', {}).get('name')
                            if tool_name:
                                tools_used.add(tool_name)
                                tool_calls_details.append({
                                    'type': 'openai',
                                    'name': tool_name
                                })
                    
                    # XML tool calls in assistant messages
                    elif msg.get('role') == 'assistant' and msg.get('content'):
                        for expected_tool in self.expected_tools:
                            pattern = f'<{expected_tool}>(.*?)</{expected_tool}>'
                            if re.search(pattern, msg['content'], re.DOTALL):
                                tools_used.add(expected_tool)
                                tool_calls_details.append({
                                    'type': 'xml',
                                    'name': expected_tool
                                })
                    
                    # Also check for tool results (XML tools in Vizra)
                    elif msg.get('role') == 'user' and msg.get('content'):
                        content = msg['content']
                        if 'Tool results:' in content:
                            # Parse tool result tags like <chord_tool_result>
                            for expected_tool in self.expected_tools:
                                result_pattern = f'<{expected_tool}_result>(.*?)</{expected_tool}_result>'
                                if re.search(result_pattern, content, re.DOTALL):
                                    tools_used.add(expected_tool)
                                    # Don't duplicate if already found in assistant message
                                    if not any(d['name'] == expected_tool and d['type'] == 'xml' 
                                              for d in tool_calls_details):
                                        tool_calls_details.append({
                                            'type': 'xml',
                                            'name': expected_tool
                                        })
        
        # Check if requirements are met
        if self.require_all:
            passed = all(tool in tools_used for tool in self.expected_tools)
        else:
            passed = any(tool in tools_used for tool in self.expected_tools)
        
        # Calculate partial score for require_all case
        score = 1.0 if passed else 0.0
        if self.require_all and self.expected_tools:
            score = len([t for t in self.expected_tools if t in tools_used]) / len(self.expected_tools)
        
        return {
            'passed': passed,
            'score': score,
            'details': {
                'expected_tools': self.expected_tools,
                'tools_used': list(tools_used),
                'require_all': self.require_all,
                'tool_calls': tool_calls_details,
                'response_preview': response[:100] + '...' if len(response) > 100 else response
            }
        }