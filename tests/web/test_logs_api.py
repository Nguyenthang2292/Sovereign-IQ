"""
Tests for Logs API endpoints (web/api/logs.py).

Tests cover:
- GET /api/logs/{session_id} (read logs)
- Error handling
- Offset handling
- Different command types
"""

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

# Import app - project root is added to path in conftest, so use absolute import
from web.app import app


@pytest.fixture
def client():
    """Create FastAPI TestClient instance."""
    return TestClient(app)


@pytest.fixture
def log_manager(tmp_path):
    """Create LogFileManager instance for testing."""
    from web.utils.log_manager import LogFileManager
    manager = LogFileManager(logs_dir=str(tmp_path))
    return manager


class TestGetLogsEndpoint:
    """Test GET /api/logs/{session_id} endpoint."""
    
    def test_get_logs_success(self, client, log_manager, tmp_path):
        """Test successfully getting logs."""
        session_id = "test-session-123"
        command_type = "scan"
        
        # Create log file with content
        log_path = log_manager.create_log_file(session_id, command_type)
        log_manager.write_log(session_id, "Line 1", command_type)
        log_manager.write_log(session_id, "Line 2", command_type)
        
        with patch('web.api.logs.get_log_manager', return_value=log_manager):
            response = client.get(f"/api/logs/{session_id}?command_type={command_type}")
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert "logs" in data
            assert "offset" in data
            assert "has_more" in data
            assert "file_size" in data
            assert "Line 1" in data["logs"]
            assert "Line 2" in data["logs"]
    
    def test_get_logs_with_offset(self, client, log_manager, tmp_path):
        """Test getting logs with offset."""
        session_id = "test-session-123"
        command_type = "scan"
        
        # Create log file
        log_manager.create_log_file(session_id, command_type)
        log_manager.write_log(session_id, "Line 1", command_type)
        
        # Get offset after first line
        _, first_offset = log_manager.read_log(session_id, offset=0, command_type=command_type)
        
        # Write more
        log_manager.write_log(session_id, "Line 2", command_type)
        log_manager.write_log(session_id, "Line 3", command_type)
        
        with patch('web.api.logs.get_log_manager', return_value=log_manager):
            response = client.get(
                f"/api/logs/{session_id}?offset={first_offset}&command_type={command_type}"
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "Line 1" not in data["logs"]
            assert "Line 2" in data["logs"]
            assert "Line 3" in data["logs"]
            assert data["offset"] > first_offset
    
    def test_get_logs_nonexistent_file(self, client, log_manager):
        """Test getting logs from non-existent file."""
        session_id = "nonexistent-session"
        command_type = "scan"
        
        with patch('web.api.logs.get_log_manager', return_value=log_manager):
            response = client.get(f"/api/logs/{session_id}?command_type={command_type}")
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["logs"] == ""
            assert data["offset"] == 0
            assert data["file_size"] == 0
    
    def test_get_logs_different_command_type(self, client, log_manager, tmp_path):
        """Test getting logs with different command type."""
        session_id = "test-session-123"
        
        # Create logs for both types
        log_manager.create_log_file(session_id, "scan")
        log_manager.create_log_file(session_id, "analyze")
        
        log_manager.write_log(session_id, "Scan log", "scan")
        log_manager.write_log(session_id, "Analyze log", "analyze")
        
        with patch('web.api.logs.get_log_manager', return_value=log_manager):
            # Get scan logs
            scan_response = client.get(f"/api/logs/{session_id}?command_type=scan")
            assert scan_response.status_code == 200
            scan_data = scan_response.json()
            assert "Scan log" in scan_data["logs"]
            assert "Analyze log" not in scan_data["logs"]
            
            # Get analyze logs
            analyze_response = client.get(f"/api/logs/{session_id}?command_type=analyze")
            assert analyze_response.status_code == 200
            analyze_data = analyze_response.json()
            assert "Analyze log" in analyze_data["logs"]
            assert "Scan log" not in analyze_data["logs"]
    
    def test_get_logs_invalid_offset(self, client, log_manager):
        """Test getting logs with invalid offset (negative)."""
        session_id = "test-session-123"
        command_type = "scan"
        
        with patch('web.api.logs.get_log_manager', return_value=log_manager):
            # FastAPI should validate offset >= 0
            response = client.get(f"/api/logs/{session_id}?offset=-1&command_type={command_type}")
            assert response.status_code == 422  # Validation error
    
    def test_get_logs_has_more_flag(self, client, log_manager, tmp_path):
        """Test has_more flag when reading partial content."""
        session_id = "test-session-123"
        command_type = "scan"
        
        # Create log file
        log_manager.create_log_file(session_id, command_type)
        log_manager.write_log(session_id, "Line 1", command_type)
        log_manager.write_log(session_id, "Line 2", command_type)
        
        file_size = log_manager.get_log_size(session_id, command_type)
        
        with patch('web.api.logs.get_log_manager', return_value=log_manager):
            # Read from beginning
            response = client.get(f"/api/logs/{session_id}?offset=0&command_type={command_type}")
            assert response.status_code == 200
            data = response.json()
            
            # If we read everything, has_more should be False
            if data["offset"] >= file_size:
                assert data["has_more"] is False
            else:
                assert data["has_more"] is True
    
    def test_get_logs_error_handling(self, client):
        """Test error handling in logs endpoint."""
        session_id = "test-session-123"
        command_type = "scan"
        
        # Mock log_manager to raise exception
        mock_manager = MagicMock()
        mock_manager.read_log.side_effect = Exception("Test error")
        
        with patch('web.api.logs.get_log_manager', return_value=mock_manager):
            response = client.get(f"/api/logs/{session_id}?command_type={command_type}")
            
            assert response.status_code == 500
            data = response.json()
            assert "detail" in data
            assert data["detail"]  # Check that detail is non-empty
            # Verify error is sanitized (should contain translated message)
            assert "Error reading logs" in data["detail"] or "Lỗi khi đọc logs" in data["detail"]
    
    def test_get_logs_error_sanitization(self, client):
        """Test that errors with sensitive info are sanitized."""
        session_id = "test-session-123"
        command_type = "scan"
        
        # Mock log_manager to raise exception with file path
        mock_manager = MagicMock()
        error_with_path = FileNotFoundError("C:\\Users\\secret\\file.txt")
        mock_manager.read_log.side_effect = error_with_path
        
        with patch('web.api.logs.get_log_manager', return_value=mock_manager):
            response = client.get(f"/api/logs/{session_id}?command_type={command_type}")
            
            assert response.status_code == 500
            data = response.json()
            detail = data["detail"]
            # Verify sensitive info is removed
            assert "C:" not in detail
            assert "Users" not in detail
            assert "secret" not in detail
            # Should contain sanitized error message
            assert "File not found" in detail or "Error reading logs" in detail

