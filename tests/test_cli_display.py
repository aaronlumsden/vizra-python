"""
Tests for CLI display module.
"""

import pytest
from unittest import mock
from io import StringIO
from vizra.cli.display import (
    console, show_welcome, print_success, print_error,
    print_warning, print_info, create_table, create_panel, create_progress_bar,
    print_json, EMOJIS, COLORS
)


class TestDisplayFunctions:
    """Test display utility functions."""
    
    def test_show_welcome(self):
        """Test welcome banner display."""
        with mock.patch.object(console, 'print') as mock_print:
            show_welcome()
            
            # Check that print was called
            assert mock_print.called
            # Check that it contains Vizra branding
            mock_print.assert_called()
            args = mock_print.call_args[0][0]
            assert 'Vizra' in str(args) or 'VIZRA' in str(args)
    
    
    def test_print_success(self):
        """Test success message printing."""
        with mock.patch.object(console, 'print') as mock_print:
            print_success("Operation completed")
            
            mock_print.assert_called_once()
            args = mock_print.call_args[0][0]
            assert 'Operation completed' in str(args)
            assert 'green' in str(args) or EMOJIS['checkmark'] in str(args)
    
    def test_print_error(self):
        """Test error message printing."""
        with mock.patch.object(console, 'print') as mock_print:
            print_error("Something went wrong")
            
            mock_print.assert_called_once()
            args = mock_print.call_args[0][0]
            assert 'Something went wrong' in str(args)
            assert 'red' in str(args) or EMOJIS['cross'] in str(args)
    
    def test_print_warning(self):
        """Test warning message printing."""
        with mock.patch.object(console, 'print') as mock_print:
            print_warning("This is a warning")
            
            mock_print.assert_called_once()
            args = mock_print.call_args[0][0]
            assert 'This is a warning' in str(args)
            assert 'yellow' in str(args) or EMOJIS['warning'] in str(args)
    
    def test_print_info(self):
        """Test info message printing."""
        with mock.patch.object(console, 'print') as mock_print:
            print_info("Information message")
            
            mock_print.assert_called_once()
            args = mock_print.call_args[0][0]
            assert 'Information message' in str(args)
            assert 'blue' in str(args) or EMOJIS['info'] in str(args)
    
    def test_create_table(self):
        """Test table creation."""
        # Test table creation
        table = create_table(
            title="Test Table",
            columns=["Name", "Value"],
            rows=[["Test", "123"], ["Another", "456"]]
        )
        assert table.title == "Test Table"
        assert len(table.columns) == 2
    
    def test_create_panel(self):
        """Test panel creation."""
        # Test basic panel
        panel = create_panel("Panel content")
        assert panel.renderable == "Panel content"
        
        # Test with title
        panel = create_panel("Content", title="Title")
        assert panel.title == "Title"
    
    def test_create_progress_bar(self):
        """Test progress bar creation."""
        progress = create_progress_bar("Processing")
        
        # Should return a Progress instance
        assert hasattr(progress, 'start')
        assert hasattr(progress, 'stop')
        assert hasattr(progress, 'add_task')
    
    
    def test_print_json(self):
        """Test JSON printing."""
        # Test with dict
        data = {"name": "test", "value": 123}
        with mock.patch.object(console, 'print') as mock_print:
            print_json(data)
            mock_print.assert_called()
            # Should have called with JSON syntax highlighting
    
    
    def test_emojis_dict(self):
        """Test EMOJIS dictionary."""
        # Check key emojis exist
        assert 'checkmark' in EMOJIS
        assert 'cross' in EMOJIS
        assert 'warning' in EMOJIS
        assert 'info' in EMOJIS
        assert 'rocket' in EMOJIS
        assert 'sparkles' in EMOJIS
        
        # All values should be strings
        for key, value in EMOJIS.items():
            assert isinstance(value, str)
    
    def test_colors_dict(self):
        """Test COLORS dictionary."""
        # Check key colors exist
        assert 'primary' in COLORS
        assert 'success' in COLORS
        assert 'error' in COLORS
        assert 'warning' in COLORS
        assert 'info' in COLORS
        
        # All values should be strings
        for key, value in COLORS.items():
            assert isinstance(value, str)
    
    def test_console_instance(self):
        """Test console instance is properly configured."""
        # Console should exist
        assert console is not None
        
        # Should have key methods
        assert hasattr(console, 'print')
        assert hasattr(console, 'log')
        assert hasattr(console, 'status')