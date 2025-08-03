"""
Example evaluation for a chord identifier agent.
"""

from vizra.evaluation import BaseEvaluation
from vizra.evaluation.metrics import ContainsMetric, BaseMetric
from vizra import BaseAgent


class ChordIdentifierAgent(BaseAgent):
    """
    Example agent that identifies musical chords.
    This is a mock agent for demonstration purposes.
    """
    name = 'chord_identifier'
    description = 'Identifies musical chords from note names'
    instructions = '''You are a music theory expert specializing in chord identification.
    When given a set of notes, identify the chord they form.
    Be precise and use standard chord notation.'''
    model = 'gpt-4o'
    
    @classmethod  
    def run(cls, message: str, context=None) -> str:
        """Mock implementation for testing."""
        # In a real implementation, this would use the LLM
        # For testing, we'll use simple pattern matching
        
        message_lower = message.lower()
        
        # Basic major chords
        if 'c e g' in message_lower:
            return "That's a C major chord."
        elif 'd f# a' in message_lower or 'd f♯ a' in message_lower:
            return "That's a D major chord."
        elif 'g b d' in message_lower:
            return "That's a G major chord."
        elif 'f a c' in message_lower:
            return "That's an F major chord."
        elif 'a c# e' in message_lower or 'a c♯ e' in message_lower:
            return "That's an A major chord."
        elif 'e♭ g b♭' in message_lower or 'eb g bb' in message_lower:
            return "That's an E♭ major chord."
            
        # Minor chords
        elif 'c e♭ g' in message_lower or 'c eb g' in message_lower:
            return "That's a C minor chord."
        elif 'd f a' in message_lower:
            return "That's a D minor chord."
            
        # Seventh chords
        elif 'c e g b♭' in message_lower or 'c e g bb' in message_lower:
            return "That's a C7 (C dominant seventh) chord."
        elif 'g b d f' in message_lower:
            return "That's a G7 (G dominant seventh) chord."
            
        else:
            return "I need to analyze those notes more carefully."


class ChordQualityMetric(BaseMetric):
    """Custom metric to check chord quality based on test type."""
    
    name = "chord_quality"
    
    def evaluate(self, row_data: dict, response: str) -> dict:
        test_type = row_data.get('test_type', '')
        expected = row_data.get('expected_response', '')
        
        if test_type == 'basic' and 'major' in expected.lower():
            passed = 'major' in response.lower()
            return {
                'passed': passed,
                'score': 1.0 if passed else 0.0,
                'details': {
                    'expected': 'major',
                    'found': passed
                }
            }
        
        elif test_type == 'minor':
            passed = 'minor' in response.lower()
            return {
                'passed': passed,
                'score': 1.0 if passed else 0.0,
                'details': {
                    'expected': 'minor',
                    'found': passed
                }
            }
        
        elif test_type in ['seventh', 'minor_seventh']:
            if '7' in expected:
                has_seven = '7' in response or 'seventh' in response.lower()
                return {
                    'passed': has_seven,
                    'score': 1.0 if has_seven else 0.0,
                    'details': {
                        'expected': 'Should mention 7 or seventh',
                        'found': has_seven
                    }
                }
        
        # Default pass for other types
        return {
            'passed': True,
            'score': 1.0,
            'details': {'test_type': test_type}
        }


class NoWrongChordTypeMetric(BaseMetric):
    """Check that response doesn't contain wrong chord types."""
    
    name = "no_wrong_chord_type"
    
    def evaluate(self, row_data: dict, response: str) -> dict:
        expected = row_data.get('expected_response', '')
        
        # If expecting major (not minor), shouldn't contain minor
        if 'major' in expected.lower() and 'minor' not in expected.lower():
            contains_minor = 'minor' in response.lower()
            return {
                'passed': not contains_minor,
                'score': 1.0 if not contains_minor else 0.0,
                'details': {
                    'unexpected': 'minor',
                    'found': contains_minor
                }
            }
        
        # Default pass
        return {
            'passed': True,
            'score': 1.0,
            'details': {}
        }


class ChordIdentifierEvaluation(BaseEvaluation):
    """
    Evaluation for testing chord identification accuracy.
    """
    name = 'chord_identifier_eval'
    description = 'Evaluate chord identification accuracy'
    agent_name = 'chord_identifier'
    csv_path = 'examples/data/chord_tests.csv'
    
    # Define metrics
    metrics = [
        ContainsMetric('expected_response'),
        ChordQualityMetric(),
        NoWrongChordTypeMetric()
    ]