"""
Configuration system for Vizra.

Provides a simple, pythonic way to manage configuration using Python dictionaries.
Users can create a vizra_config.py file in their project root to customize settings.
"""

import os
from typing import Any, Optional


class Config:
    """Configuration object with dot notation access."""
    
    def __init__(self, config_dict: dict):
        """Initialize with a configuration dictionary."""
        self._config = config_dict or {}
        
    def get(self, path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            path: Dot-separated path to config value (e.g., 'llm.model')
            default: Default value if path not found
            
        Returns:
            Configuration value or default
            
        Examples:
            >>> config.get('llm.model', 'gpt-4o')
            >>> config.get('training.batch_size', 32)
        """
        keys = path.split('.')
        value = self._config
        
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
                if value is None:
                    return default
            else:
                return default
                
        return value if value is not None else default
    
    def set(self, path: str, value: Any) -> None:
        """
        Set configuration value using dot notation.
        
        Args:
            path: Dot-separated path to config value
            value: Value to set
        """
        keys = path.split('.')
        config = self._config
        
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
            
        config[keys[-1]] = value
    
    def update(self, config_dict: dict) -> None:
        """Update configuration with new values."""
        self._config.update(config_dict)


def load_config() -> Config:
    """
    Load configuration from vizra_config.py if it exists.
    
    Returns:
        Config object with loaded settings or empty config
    """
    config_dict = {}
    
    # Try to load from vizra_config.py in current directory
    try:
        import vizra_config
        if hasattr(vizra_config, 'settings'):
            config_dict = vizra_config.settings
    except ImportError:
        pass
    
    # Also check for VIZRA_CONFIG environment variable pointing to a config file
    config_path = os.getenv('VIZRA_CONFIG')
    if config_path and os.path.exists(config_path):
        import importlib.util
        spec = importlib.util.spec_from_file_location("vizra_config_env", config_path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            if hasattr(module, 'settings'):
                config_dict.update(module.settings)
    
    return Config(config_dict)


# Global config instance
_config = load_config()


def config(path: str, default: Any = None) -> Any:
    """
    Get configuration value using dot notation.
    
    This is the main function users should use to access configuration.
    
    Args:
        path: Dot-separated path to config value (e.g., 'llm.model')
        default: Default value if path not found
        
    Returns:
        Configuration value or default
        
    Examples:
        >>> from vizra.config import config
        >>> model = config('llm.model', 'gpt-4o')
        >>> api_key = config('api.openai.key')
    """
    return _config.get(path, default)


def get_config() -> Config:
    """Get the global Config instance for advanced usage."""
    return _config


def reload_config() -> None:
    """Reload configuration from file (useful for testing)."""
    global _config
    _config = load_config()