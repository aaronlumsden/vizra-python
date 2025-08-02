class VizraException(Exception):
    """Base exception for Vizra framework."""
    pass


class ToolExecutionError(VizraException):
    """Raised when a tool execution fails."""
    pass


class AgentExecutionError(VizraException):
    """Raised when agent execution fails."""
    pass


class MaxIterationsReachedError(VizraException):
    """Raised when maximum tool iterations are reached."""
    pass