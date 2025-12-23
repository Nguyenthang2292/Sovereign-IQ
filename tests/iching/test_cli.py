"""
Tests for I Ching CLI.
"""

import pytest
import sys
from unittest.mock import Mock, patch, MagicMock

from modules.iching.cli.main import main


class TestMain:
    """Test main function."""
    
    @patch('modules.iching.cli.main.fill_web_form')
    @patch('modules.iching.cli.main.prepare_hexagram')
    @patch('modules.iching.cli.main.clean_images_folder')
    @patch('modules.iching.cli.main.log_info')
    @patch('modules.iching.cli.main.log_error')
    def test_main_success(self, mock_log_error, mock_log_info, mock_clean, 
                         mock_prepare, mock_fill):
        """Test successful main execution."""
        # Setup mocks
        mock_clean.return_value = 0
        mock_prepare.return_value = [{"is_solid": True, "is_red": False}] * 6
        mock_fill.return_value = None
        
        # Should not raise exception
        main(auto_close=True)
        
        # Verify calls
        mock_clean.assert_called_once()
        mock_prepare.assert_called_once()
        mock_fill.assert_called_once()
    
    @patch('modules.iching.cli.main.fill_web_form')
    @patch('modules.iching.cli.main.prepare_hexagram')
    @patch('modules.iching.cli.main.clean_images_folder')
    @patch('modules.iching.cli.main.log_info')
    @patch('modules.iching.cli.main.log_error')
    @patch('sys.exit')
    def test_main_clean_error(self, mock_exit, mock_log_error, mock_log_info, 
                             mock_clean, mock_prepare, mock_fill):
        """Test main with clean_images_folder error."""
        mock_clean.side_effect = Exception("Clean error")
        
        main(auto_close=True)
        
        # Verify error was logged with the exception message
        mock_log_error.assert_called_once()
        error_call_args = mock_log_error.call_args[0][0]
        assert "Clean error" in error_call_args
        # Ensure log_info was not called with success messages after error
        assert not any("Đã xóa" in str(call) or "Folder images đã sạch" in str(call) 
                      for call in mock_log_info.call_args_list)
        
        # Should exit on error
        mock_exit.assert_called_once_with(1)
        mock_prepare.assert_not_called()
        mock_fill.assert_not_called()
    
    @patch('modules.iching.cli.main.fill_web_form')
    @patch('modules.iching.cli.main.prepare_hexagram')
    @patch('modules.iching.cli.main.clean_images_folder')
    @patch('modules.iching.cli.main.log_info')
    @patch('modules.iching.cli.main.log_error')
    @patch('sys.exit')
    def test_main_prepare_error(self, mock_exit, mock_log_error, mock_log_info, 
                                mock_clean, mock_prepare, mock_fill):
        """Test main with prepare_hexagram error."""
        mock_clean.return_value = 0
        mock_prepare.side_effect = ValueError("Prepare error")
        
        main(auto_close=True)
        
        # Verify error was logged with the exception message
        mock_log_error.assert_called_once()
        error_call_args = mock_log_error.call_args[0][0]
        assert "Prepare error" in error_call_args
        # Should exit on error
        mock_exit.assert_called_once_with(1)
        mock_fill.assert_not_called()
    
    @patch('modules.iching.cli.main.fill_web_form')
    @patch('modules.iching.cli.main.prepare_hexagram')
    @patch('modules.iching.cli.main.clean_images_folder')
    @patch('modules.iching.cli.main.log_info')
    @patch('modules.iching.cli.main.log_error')
    @patch('sys.exit')
    def test_main_fill_error(self, mock_exit, mock_log_error, mock_log_info, 
                             mock_clean, mock_prepare, mock_fill):
        """Test main with fill_web_form error."""
        mock_clean.return_value = 0
        mock_prepare.return_value = [{"is_solid": True, "is_red": False}] * 6
        mock_fill.side_effect = RuntimeError("Fill error")
        
        main(auto_close=True)
        
        # Verify error was logged with the exception message
        mock_log_error.assert_called_once()
        error_call_args = mock_log_error.call_args[0][0]
        assert "Fill error" in error_call_args
        # Should exit on error
        mock_exit.assert_called_once_with(1)
    
    @patch('modules.iching.cli.main.fill_web_form')
    @patch('modules.iching.cli.main.prepare_hexagram')
    @patch('modules.iching.cli.main.clean_images_folder')
    @patch('modules.iching.cli.main.log_info')
    def test_main_with_deleted_files(self, mock_log_info, mock_clean, 
                                     mock_prepare, mock_fill):
        """Test main when files are deleted."""
        mock_clean.return_value = 5  # 5 files deleted
        mock_prepare.return_value = [{"is_solid": True, "is_red": False}] * 6
        mock_fill.return_value = None
        
        main(auto_close=True)
        
        # Should log deleted count with exact message
        mock_log_info.assert_any_call("Đã xóa 5 file trong folder images")

