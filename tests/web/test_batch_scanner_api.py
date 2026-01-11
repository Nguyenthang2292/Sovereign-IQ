
from datetime import datetime
from unittest.mock import MagicMock, Mock, patch
import json
import os
import time
import uuid

from fastapi.testclient import TestClient
import pytest

from web.app import app

from web.app import app

"""
Tests for Batch Scanner API endpoints (web/api/batch_scanner.py).

Tests cover:
- POST /api/batch/scan (batch market scanning - old blocking version)
- POST /api/batch/scan (new background thread version)
- GET /api/batch/scan/{session_id}/status (get scan status)
- GET /api/batch/results/{filename} (get saved results)
- GET /api/batch/list (list all results)
"""



# Import app - project root is added to path in conftest, so use absolute import
from web.app import app


@pytest.fixture
def client():
    """Create FastAPI TestClient instance."""
    return TestClient(app)


class TestBatchScanEndpoint:
    """Test POST /api/batch/scan endpoint"""

    def test_success_single_timeframe_mode(self, client, tmp_path):
        """Test successful batch scan in single timeframe mode (background thread)."""
        results_file = tmp_path / "results.json"
        results_file.write_text('{"test": "data"}')

        mock_scanner_result = {
            "summary": {"total_scanned": 10, "long_count": 3, "short_count": 2, "none_count": 5},
            "long_symbols": ["BTC/USDT", "ETH/USDT"],
            "short_symbols": ["ADA/USDT"],
            "long_symbols_with_confidence": [{"symbol": "BTC/USDT", "confidence": 0.8}],
            "short_symbols_with_confidence": [{"symbol": "ADA/USDT", "confidence": 0.7}],
            "all_results": {"BTC/USDT": {"signal": "LONG"}},
            "results_file": str(results_file),
        }

        with (
            patch("web.api.batch_scanner.MarketBatchScanner") as mock_scanner_class,
            patch("web.api.batch_scanner.get_log_manager") as mock_log_mgr,
            patch("web.api.batch_scanner.get_task_manager") as mock_task_mgr,
        ):
            mock_scanner = Mock()
            mock_scanner.scan_market.return_value = mock_scanner_result
            mock_scanner_class.return_value = mock_scanner

            mock_log_manager = Mock()
            mock_log_manager.create_log_file.return_value = tmp_path / "test.log"
            mock_log_mgr.return_value = mock_log_manager

            mock_task_manager = Mock()
            mock_task_mgr.return_value = mock_task_manager

            request_data = {"timeframe": "1h", "max_symbols": 10}
            response = client.post("/api/batch/scan", json=request_data)

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert "session_id" in data
            assert data["status"] == "running"
            assert "message" in data

            # Verify task_manager.start_task was called with correct parameters
            mock_task_manager.start_task.assert_called_once()
            call_args = mock_task_manager.start_task.call_args
            assert call_args[0][0] == data["session_id"]  # session_id
            assert callable(call_args[0][1])  # task_func (run_scan)
            assert call_args[0][2] == "scan"  # command_type as positional arg

    def test_success_multi_timeframe_mode(self, client, tmp_path):
        """Test successful batch scan in multi-timeframe mode (background thread)."""
        results_file = tmp_path / "results.json"
        results_file.write_text('{"test": "data"}')

        mock_scanner_result = {
            "summary": {"total_scanned": 5, "long_count": 2, "short_count": 1, "none_count": 2},
            "long_symbols": ["BTC/USDT"],
            "short_symbols": [],
            "long_symbols_with_confidence": [],
            "short_symbols_with_confidence": [],
            "all_results": {},
            "results_file": str(results_file),
        }

        with (
            patch("web.api.batch_scanner.MarketBatchScanner") as mock_scanner_class,
            patch("web.api.batch_scanner.get_log_manager") as mock_log_mgr,
            patch("web.api.batch_scanner.get_task_manager") as mock_task_mgr,
        ):
            mock_scanner = Mock()
            mock_scanner.scan_market.return_value = mock_scanner_result
            mock_scanner_class.return_value = mock_scanner

            mock_log_manager = Mock()
            mock_log_manager.create_log_file.return_value = tmp_path / "test.log"
            mock_log_mgr.return_value = mock_log_manager

            mock_task_manager = Mock()
            mock_task_mgr.return_value = mock_task_manager

            with (
                patch(
                    "modules.gemini_chart_analyzer.core.utils.chart_paths.get_analysis_results_dir",
                    return_value=tmp_path,
                ),
                patch("os.path.exists", return_value=True),
            ):
                request_data = {"timeframes": ["1h", "4h"], "max_symbols": 5}
                response = client.post("/api/batch/scan", json=request_data)

                assert response.status_code == 200
                data = response.json()
                assert data["success"] is True
                assert "session_id" in data
                assert data["status"] == "running"

    def test_success_with_custom_parameters(self, client, tmp_path):
        """Test batch scan with custom cooldown and charts_per_batch."""
        results_file = tmp_path / "results.json"
        results_file.write_text('{"test": "data"}')

        mock_scanner_result = {
            "summary": {"total_scanned": 1, "long_count": 0, "short_count": 0, "none_count": 1},
            "long_symbols": [],
            "short_symbols": [],
            "long_symbols_with_confidence": [],
            "short_symbols_with_confidence": [],
            "all_results": {},
            "results_file": str(results_file),
        }

        with (
            patch("web.api.batch_scanner.MarketBatchScanner") as mock_scanner_class,
            patch("web.api.batch_scanner.get_log_manager") as mock_log_mgr,
            patch("web.api.batch_scanner.get_task_manager") as mock_task_mgr,
        ):
            mock_scanner = Mock()
            mock_scanner.scan_market.return_value = mock_scanner_result
            mock_scanner_class.return_value = mock_scanner

            mock_log_manager = Mock()
            mock_log_manager.create_log_file.return_value = tmp_path / "test.log"
            mock_log_mgr.return_value = mock_log_manager

            mock_task_manager = Mock()
            mock_task_mgr.return_value = mock_task_manager

            with (
                patch(
                    "modules.gemini_chart_analyzer.core.utils.chart_paths.get_analysis_results_dir",
                    return_value=tmp_path,
                ),
                patch("os.path.exists", return_value=True),
            ):
                request_data = {
                    "timeframe": "1h",
                    "cooldown": 5.0,
                    "charts_per_batch": 50,
                    "quote_currency": "BTC",
                    "exchange_name": "kraken",
                }
                response = client.post("/api/batch/scan", json=request_data)

                assert response.status_code == 200
                data = response.json()
                assert "session_id" in data
                assert data["status"] == "running"
                # MarketBatchScanner is called in background thread, not immediately
                # Verify that start_task was called instead
                mock_task_manager.start_task.assert_called_once()

    def test_success_results_url_generation(self, client, tmp_path):
        """Test results URL generation."""
        batch_scan_dir = tmp_path / "batch_scan"
        batch_scan_dir.mkdir(parents=True, exist_ok=True)
        results_file = batch_scan_dir / "results.json"
        results_file.write_text('{"test": "data"}')

        mock_scanner_result = {
            "summary": {},
            "long_symbols": [],
            "short_symbols": [],
            "long_symbols_with_confidence": [],
            "short_symbols_with_confidence": [],
            "all_results": {},
            "results_file": str(results_file),
        }

        with (
            patch("web.api.batch_scanner.MarketBatchScanner") as mock_scanner_class,
            patch("web.api.batch_scanner.get_log_manager") as mock_log_mgr,
            patch("web.api.batch_scanner.get_task_manager") as mock_task_mgr,
        ):
            mock_scanner = Mock()
            mock_scanner.scan_market.return_value = mock_scanner_result
            mock_scanner_class.return_value = mock_scanner

            mock_log_manager = Mock()
            mock_log_manager.create_log_file.return_value = tmp_path / "test.log"
            mock_log_mgr.return_value = mock_log_manager

            mock_task_manager = Mock()
            mock_task_manager.is_cancelled.return_value = False
            mock_task_manager._lock = MagicMock()
            mock_task_manager._tasks = {}
            mock_task_mgr.return_value = mock_task_manager

            with (
                patch(
                    "modules.gemini_chart_analyzer.core.utils.chart_paths.get_analysis_results_dir",
                    return_value=tmp_path,
                ),
                patch("os.path.exists", return_value=True),
            ):
                request_data = {"timeframe": "1h"}
                response = client.post("/api/batch/scan", json=request_data)

                assert response.status_code == 200
                data = response.json()
                # results_url is only available after task completes, not in immediate response
                assert "session_id" in data
                assert data["status"] == "running"

    def test_error_no_timeframe_or_timeframes(self, client):
        """Test error when neither timeframe nor timeframes is provided."""
        request_data = {}
        response = client.post("/api/batch/scan", json=request_data)

        assert response.status_code == 400
        assert "Either 'timeframe' (single) or 'timeframes' (multi) must be provided" in response.json()["detail"]

    def test_error_invalid_timeframes(self, client):
        """Test error when timeframes are invalid."""
        # Patch at the source location where it's actually imported and used
        with patch(
            "modules.gemini_chart_analyzer.core.utils.validate_timeframes",
            return_value=(False, "Invalid timeframe format"),
        ):
            request_data = {"timeframes": ["invalid"]}
            response = client.post("/api/batch/scan", json=request_data)

            assert response.status_code == 400
            assert "Invalid" in response.json()["detail"]

    def test_error_scanner_failure(self, client, tmp_path):
        """Test error when scanner fails - error occurs in background thread."""
        with (
            patch("web.api.batch_scanner.MarketBatchScanner") as mock_scanner_class,
            patch("web.api.batch_scanner.get_log_manager") as mock_log_mgr,
            patch("web.api.batch_scanner.get_task_manager") as mock_task_mgr,
        ):
            mock_scanner = Mock()
            mock_scanner.scan_market.side_effect = Exception("Scanner error")
            mock_scanner_class.return_value = mock_scanner

            mock_log_manager = Mock()
            mock_log_manager.create_log_file.return_value = tmp_path / "test.log"
            mock_log_mgr.return_value = mock_log_manager

            mock_task_manager = Mock()
            mock_task_mgr.return_value = mock_task_manager

            request_data = {"timeframe": "1h"}
            # Scanner error occurs in background thread, so immediate response is 200 with status "running"
            response = client.post("/api/batch/scan", json=request_data)

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "running"
            # Error will be set in background thread via task_manager.set_error()

    def test_success_edge_case_max_symbols_one(self, client, tmp_path):
        """Test edge case with max_symbols=1."""
        results_file = tmp_path / "results.json"
        results_file.write_text('{"test": "data"}')

        mock_scanner_result = {
            "summary": {"total_scanned": 1, "long_count": 0, "short_count": 0, "none_count": 1},
            "long_symbols": [],
            "short_symbols": [],
            "long_symbols_with_confidence": [],
            "short_symbols_with_confidence": [],
            "all_results": {},
            "results_file": str(results_file),
        }

        with (
            patch("web.api.batch_scanner.MarketBatchScanner") as mock_scanner_class,
            patch("web.api.batch_scanner.get_log_manager") as mock_log_mgr,
            patch("web.api.batch_scanner.get_task_manager") as mock_task_mgr,
        ):
            mock_scanner = Mock()
            mock_scanner.scan_market.return_value = mock_scanner_result
            mock_scanner_class.return_value = mock_scanner

            mock_log_manager = Mock()
            mock_log_manager.create_log_file.return_value = tmp_path / "test.log"
            mock_log_mgr.return_value = mock_log_manager

            mock_task_manager = Mock()
            mock_task_mgr.return_value = mock_task_manager

            with (
                patch(
                    "modules.gemini_chart_analyzer.core.utils.chart_paths.get_analysis_results_dir",
                    return_value=tmp_path,
                ),
                patch("os.path.exists", return_value=True),
            ):
                request_data = {"timeframe": "1h", "max_symbols": 1}
                response = client.post("/api/batch/scan", json=request_data)

                assert response.status_code == 200
                data = response.json()
                assert data["success"] is True
                assert data["status"] == "running"


class TestGetBatchResultsEndpoint:
    """Test GET /api/batch/results/{filename} endpoint"""

    def test_success_read_valid_json(self, client, tmp_path):
        """Test successfully reading a valid JSON results file."""
        batch_scan_dir = tmp_path / "batch_scan"
        batch_scan_dir.mkdir(parents=True, exist_ok=True)

        results_file = batch_scan_dir / "test_results.json"
        results_data = {
            "summary": {"total_scanned": 10, "long_count": 3},
            "long_symbols": ["BTC/USDT", "ETH/USDT"],
            "timestamp": datetime.now().isoformat(),
        }
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results_data, f)

        with patch(
            "modules.gemini_chart_analyzer.core.utils.chart_paths.get_analysis_results_dir", return_value=tmp_path
        ):
            response = client.get("/api/batch/results/test_results.json")

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["filename"] == "test_results.json"
            assert "results" in data
            assert data["results"]["summary"]["total_scanned"] == 10

    def test_error_file_not_found(self, client, tmp_path):
        """Test error when file does not exist."""
        batch_scan_dir = tmp_path / "batch_scan"
        batch_scan_dir.mkdir(parents=True, exist_ok=True)
        # Don't create the file

        with patch(
            "modules.gemini_chart_analyzer.core.utils.chart_paths.get_analysis_results_dir", return_value=tmp_path
        ):
            response = client.get("/api/batch/results/nonexistent.json")

            # File doesn't exist, should return 404 (not 400)
            assert response.status_code == 404
            assert "not found" in response.json()["detail"].lower()

    def test_error_not_json_file(self, client):
        """Test error when filename is not a JSON file."""
        response = client.get("/api/batch/results/test.txt")

        assert response.status_code == 400
        assert "Only JSON files are allowed" in response.json()["detail"]

    def test_error_directory_traversal_attempt(self, client, tmp_path):
        """Test error when directory traversal is attempted.

        Tests various directory traversal patterns:
        - Relative paths with '..'
        - Absolute paths (Unix and Windows)
        - Paths with and without .json extension (to verify traversal check takes priority)
        - URL-encoded paths to ensure FastAPI decodes correctly
        """
        import sys
        from urllib.parse import quote

        # Create batch_scan directory structure
        with patch(
            "modules.gemini_chart_analyzer.core.utils.chart_paths.get_analysis_results_dir", return_value=tmp_path
        ):
            batch_scan_dir = tmp_path / "batch_scan"
            batch_scan_dir.mkdir(parents=True, exist_ok=True)

            # Test cases for directory traversal detection
            # Note: "test/../test.json" normalizes to "test.json" which is a valid filename,
            # but the API policy rejects any filename containing ".." regardless of normalization.
            # This case is tested separately in test_normalized_path_with_traversal_pattern below.
            test_cases = [
                # Relative paths with '..' (with .json extension)
                "../test.json",
                "../../test.json",
                "../../../test.json",
                "test/../test.json",
                "subdir/../../test.json",
                # Relative paths with '..' (without .json extension - should still be caught by traversal check)
                "../test",
                "../../test",
                "test/../test",
            ]

            # Add absolute path test cases based on platform
            if sys.platform != "win32":
                # Unix-like systems: test absolute paths
                test_cases.extend(
                    [
                        "/etc/passwd",  # Without .json
                        "/etc/passwd.json",  # With .json (should still be rejected)
                        "/root/.ssh/id_rsa",
                        "/var/log/syslog",
                    ]
                )
            else:
                # Windows: test Windows absolute paths
                test_cases.extend(
                    [
                        "C:/Windows/System32/config/sam",  # Without .json
                        "C:/Windows/System32/config/sam.json",  # With .json (should still be rejected)
                        "C:\\Windows\\System32\\config\\sam",  # Backslash (should normalize but still be caught)
                        "D:/test.json",
                    ]
                )

            # Test each case
            for filename in test_cases:
                # Normalize backslashes to forward slashes for URL (Windows paths)
                # FastAPI path parameters handle forward slashes, backslashes may cause issues
                url_safe_filename = filename.replace("\\", "/")

                # URL-encode the filename to ensure FastAPI handles special characters correctly
                # FastAPI automatically decodes URL-encoded path parameters
                encoded_filename = quote(url_safe_filename, safe="")

                # Test with URL-encoded path (most realistic scenario)
                response = client.get(f"/api/batch/results/{encoded_filename}")
                assert response.status_code == 400, (
                    f"Expected 400 for traversal attempt '{filename}', got {response.status_code}"
                )
                detail = response.json()["detail"]
                assert "Directory traversal detected" in detail, (
                    f"Expected 'Directory traversal detected' in error message for '{filename}', got: {detail}"
                )

                # Note: Raw path testing is skipped because HTTP clients (including FastAPI TestClient)
                # automatically normalize paths containing ".." before sending the request.
                # This means the API never sees the ".." pattern in raw paths, so the security check
                # cannot detect traversal attempts. URL encoding is required to prevent normalization.
                # The encoded test above is sufficient to verify the security check works correctly.

    def test_normalized_path_with_traversal_pattern(self, client, tmp_path):
        """Test behavior for paths that normalize to valid filenames but contain traversal patterns.

        The path "test/../test.json" normalizes to "test.json" which is a valid filename.
        This test verifies the API policy: whether such paths are allowed (after normalization)
        or rejected (because they contain "..").
        """
        from urllib.parse import quote

        # Create batch_scan directory structure
        with patch(
            "modules.gemini_chart_analyzer.core.utils.chart_paths.get_analysis_results_dir", return_value=tmp_path
        ):
            batch_scan_dir = tmp_path / "batch_scan"
            batch_scan_dir.mkdir(parents=True, exist_ok=True)

            # Create a test file that the normalized path would point to
            test_file = batch_scan_dir / "test.json"
            test_file.write_text('{"test": "data"}')

            # Test cases: paths that normalize to valid filenames but contain traversal patterns
            test_cases = [
                # Basic normalized path
                "test/../test.json",
                # Multiple traversal patterns
                "a/b/../../test.json",
                "test/../subdir/../test.json",
                # With encoded slashes (URL encoding)
                quote("test/../test.json", safe=""),
                quote("a/b/../../test.json", safe=""),
                # Mixed encoding
                "test" + quote("/../", safe="") + "test.json",
                # Unicode and special characters in path
                "test/../test file.json",
                "test/../test-file.json",
                # Edge case: path starting with traversal
                "../test.json",
                # Edge case: multiple consecutive traversals
                "test/../../test.json",
                # Edge case: traversal at end (should still be caught by .. check)
                "test.json/..",
            ]

            for normalized_path in test_cases:
                # URL encode the path to prevent HTTP client from normalizing it
                # This ensures the ".." pattern reaches the API endpoint
                encoded_path = quote(normalized_path, safe="")
                response = client.get(f"/api/batch/results/{encoded_path}")

                # API policy: reject paths containing ".." even if they normalize to valid paths
                assert response.status_code == 400, (
                    f"Expected 400 for path '{normalized_path}' (encoded: '{encoded_path}'), got {response.status_code}"
                )
                detail = response.json()["detail"]
                assert "Directory traversal detected" in detail, (
                    f"Expected 'Directory traversal detected' for path '{normalized_path}', got: {detail}"
                )

    def test_url_encoded_traversal_patterns(self, client, tmp_path):
        """Test URL-encoded traversal patterns to ensure they are properly detected."""

        # Create batch_scan directory structure
        with patch(
            "modules.gemini_chart_analyzer.core.utils.chart_paths.get_analysis_results_dir", return_value=tmp_path
        ):
            batch_scan_dir = tmp_path / "batch_scan"
            batch_scan_dir.mkdir(parents=True, exist_ok=True)

            # Create a test file
            test_file = batch_scan_dir / "test.json"
            test_file.write_text('{"test": "data"}')

            # Test cases with various URL encoding schemes
            test_cases = [
                # Fully encoded path
                ("test%2F..%2Ftest.json", "test/../test.json"),
                # Encoded dots
                ("test/%2E%2E/test.json", "test/../test.json"),
                # Mixed encoding
                ("test%2F..%2Ftest%2Ejson", "test/../test.json"),
                # Double encoding (should be decoded by FastAPI)
                ("test%252F..%252Ftest.json", "test%2F..%2Ftest.json"),  # After first decode
            ]

            for encoded_path, expected_decoded in test_cases:
                response = client.get(f"/api/batch/results/{encoded_path}")

                # Should reject because after URL decoding, it contains ".."
                assert response.status_code == 400, (
                    f"Expected 400 for encoded path '{encoded_path}', got {response.status_code}"
                )
                detail = response.json()["detail"]
                assert "Directory traversal detected" in detail, (
                    f"Expected 'Directory traversal detected' for encoded path '{encoded_path}', got: {detail}"
                )

    def test_edge_cases_traversal_detection(self, client, tmp_path):
        """Test edge cases for traversal detection."""

        # Create batch_scan directory structure
        with patch(
            "modules.gemini_chart_analyzer.core.utils.chart_paths.get_analysis_results_dir", return_value=tmp_path
        ):
            batch_scan_dir = tmp_path / "batch_scan"
            batch_scan_dir.mkdir(parents=True, exist_ok=True)

            # Create a test file
            test_file = batch_scan_dir / "test.json"
            test_file.write_text('{"test": "data"}')

            # Edge cases that should be rejected
            edge_cases = [
                # Windows absolute path
                "C:/test.json",
                "C:\\test.json",
                # Unix absolute path
                "/etc/passwd.json",
                "/root/test.json",
                # Path with only dots (should fail JSON check but test traversal first)
                "..",
                "...",
                # Path with dots at different positions
                "test..json",  # Should pass traversal check but fail JSON check
                "..test.json",  # Should be caught
                "test....json",  # Multiple double dots but not traversal pattern
                "test//test.json",  # Should pass traversal check
                # Path with spaces and traversal
                "test / ../test.json",
                # Path with unicode
                "test/../测试.json",
            ]

            for path in edge_cases:
                response = client.get(f"/api/batch/results/{path}")

                # Check if it's caught by traversal detection or JSON validation
                assert response.status_code in [400, 404], (
                    f"Expected 400 or 404 for path '{path}', got {response.status_code}"
                )

                if response.status_code == 400:
                    detail = response.json()["detail"]
                    # Should be either traversal detection or JSON validation error
                    assert "Directory traversal detected" in detail or "Only JSON files are allowed" in detail, (
                        f"Unexpected error message for path '{path}': {detail}"
                    )

    def test_valid_paths_should_pass(self, client, tmp_path):
        """Test that valid paths without traversal patterns are accepted."""

        # Create batch_scan directory structure
        with patch(
            "modules.gemini_chart_analyzer.core.utils.chart_paths.get_analysis_results_dir", return_value=tmp_path
        ):
            batch_scan_dir = tmp_path / "batch_scan"
            batch_scan_dir.mkdir(parents=True, exist_ok=True)

            # Valid test cases (should work if files exist)
            # Note: API only allows files directly in batch_scan_dir, not subdirectories
            valid_cases = [
                ("simple.json", '{"test": "data1"}'),
                ("test-file.json", '{"test": "data2"}'),
                ("test_file.json", '{"test": "data3"}'),
                ("test123.json", '{"test": "data4"}'),
            ]

            for filename, content in valid_cases:
                # Create file
                file_path = batch_scan_dir / filename
                file_path.write_text(content)

                response = client.get(f"/api/batch/results/{filename}")

                # Should succeed for valid files
                assert response.status_code == 200, (
                    f"Expected 200 for valid path '{filename}', got {response.status_code}"
                )
                data = response.json()
                assert data["success"] is True, f"Expected success=True for valid path '{filename}'"
                assert data["filename"] == filename, (
                    f"Expected filename '{filename}' in response, got '{data['filename']}'"
                )

    def test_error_invalid_json_content(self, client, tmp_path):
        """Test error when file contains invalid JSON."""
        batch_scan_dir = tmp_path / "batch_scan"
        batch_scan_dir.mkdir(parents=True, exist_ok=True)

        results_file = batch_scan_dir / "invalid.json"
        results_file.write_text("This is not valid JSON {")

        with patch(
            "modules.gemini_chart_analyzer.core.utils.chart_paths.get_analysis_results_dir", return_value=tmp_path
        ):
            response = client.get("/api/batch/results/invalid.json")

            # Should return 500 error when trying to parse invalid JSON
            assert response.status_code == 500
            assert "Lỗi khi đọc kết quả" in response.json()["detail"]

    def test_success_empty_json_file(self, client, tmp_path):
        """Test reading an empty JSON file (edge case)."""
        batch_scan_dir = tmp_path / "batch_scan"
        batch_scan_dir.mkdir(parents=True, exist_ok=True)

        results_file = batch_scan_dir / "empty.json"
        results_file.write_text("{}")

        with patch(
            "modules.gemini_chart_analyzer.core.utils.chart_paths.get_analysis_results_dir", return_value=tmp_path
        ):
            response = client.get("/api/batch/results/empty.json")

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["results"] == {}


class TestListBatchResultsEndpoint:
    """Test GET /api/batch/list endpoint"""

    def test_success_list_with_files(self, client, tmp_path):
        """Test listing results when files exist."""
        batch_scan_dir = tmp_path / "batch_scan"
        batch_scan_dir.mkdir(parents=True, exist_ok=True)

        # Create multiple result files
        for i in range(3):
            results_file = batch_scan_dir / f"results_{i}.json"
            results_data = {
                "summary": {"total_scanned": 10 + i, "long_count": i},
                "timestamp": datetime.now().isoformat(),
            }
            with open(results_file, "w", encoding="utf-8") as f:
                json.dump(results_data, f)

        with patch(
            "modules.gemini_chart_analyzer.core.utils.chart_paths.get_analysis_results_dir", return_value=tmp_path
        ):
            response = client.get("/api/batch/list")

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["count"] == 3
            assert len(data["results"]) == 3
            # Verify each result has required fields
            for result in data["results"]:
                assert "filename" in result
                assert "size" in result
                assert "modified" in result
                assert "summary" in result
                assert "url" in result
                assert result["url"].startswith("/api/batch/results/")

    def test_success_list_empty_directory(self, client, tmp_path):
        """Test listing when directory is empty."""
        batch_scan_dir = tmp_path / "batch_scan"
        batch_scan_dir.mkdir(parents=True, exist_ok=True)
        # Don't create any files

        with patch(
            "modules.gemini_chart_analyzer.core.utils.chart_paths.get_analysis_results_dir", return_value=tmp_path
        ):
            response = client.get("/api/batch/list")

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["count"] == 0
            assert data["results"] == []

    def test_success_list_directory_not_exists(self, client, tmp_path):
        """Test listing when directory does not exist."""
        # Don't create batch_scan directory

        with patch(
            "modules.gemini_chart_analyzer.core.utils.chart_paths.get_analysis_results_dir", return_value=tmp_path
        ):
            response = client.get("/api/batch/list")

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["count"] == 0
            assert data["results"] == []

    def test_success_sorting_by_modified_time(self, client, tmp_path):
        """Test that results are sorted by modified time (newest first)."""
        batch_scan_dir = tmp_path / "batch_scan"
        batch_scan_dir.mkdir(parents=True, exist_ok=True)

        # Create files with explicitly set modification times (deterministic, increasing)
        base_time = time.time()
        for i in range(3):
            results_file = batch_scan_dir / f"results_{i}.json"
            results_data = {"summary": {"total_scanned": i}}
            with open(results_file, "w", encoding="utf-8") as f:
                json.dump(results_data, f)
            # Set explicit modification time: older files first (i=0 is oldest, i=2 is newest)
            # Use increasing timestamps with 10 second intervals for clarity
            file_mtime = base_time + (i * 10)
            os.utime(results_file, (file_mtime, file_mtime))

        with patch(
            "modules.gemini_chart_analyzer.core.utils.chart_paths.get_analysis_results_dir", return_value=tmp_path
        ):
            response = client.get("/api/batch/list")

            assert response.status_code == 200
            data = response.json()
            # Verify sorting (newest first)
            assert len(data["results"]) == 3
            modified_times = [r["modified"] for r in data["results"]]
            # Results should be sorted newest first (results_2.json, results_1.json, results_0.json)
            assert modified_times == sorted(modified_times, reverse=True)
            # Verify filenames are in correct order (newest first)
            filenames = [r["filename"] for r in data["results"]]
            assert filenames == ["results_2.json", "results_1.json", "results_0.json"]

    def test_success_files_with_and_without_summary(self, client, tmp_path):
        """Test listing files with and without summary (edge case)."""
        batch_scan_dir = tmp_path / "batch_scan"
        batch_scan_dir.mkdir(parents=True, exist_ok=True)

        # File with summary
        results_file1 = batch_scan_dir / "with_summary.json"
        with open(results_file1, "w", encoding="utf-8") as f:
            json.dump({"summary": {"total_scanned": 10}}, f)

        # File without summary
        results_file2 = batch_scan_dir / "no_summary.json"
        with open(results_file2, "w", encoding="utf-8") as f:
            json.dump({"long_symbols": []}, f)

        with patch(
            "modules.gemini_chart_analyzer.core.utils.chart_paths.get_analysis_results_dir", return_value=tmp_path
        ):
            response = client.get("/api/batch/list")

            assert response.status_code == 200
            data = response.json()
            assert data["count"] == 2
            # Find files by filename
            with_summary = next(r for r in data["results"] if r["filename"] == "with_summary.json")
            no_summary = next(r for r in data["results"] if r["filename"] == "no_summary.json")
            assert "total_scanned" in with_summary["summary"]  # summary dict has total_scanned
            assert no_summary["summary"] == {}  # Empty summary dict

    def test_success_skip_unreadable_files(self, client, tmp_path):
        """Test that unreadable files are skipped."""
        batch_scan_dir = tmp_path / "batch_scan"
        batch_scan_dir.mkdir(parents=True, exist_ok=True)

        # Valid file
        valid_file = batch_scan_dir / "valid.json"
        with open(valid_file, "w", encoding="utf-8") as f:
            json.dump({"summary": {}}, f)

        # Invalid JSON file (will cause error when reading)
        invalid_file = batch_scan_dir / "invalid.json"
        invalid_file.write_text("Not valid JSON {")

        with patch(
            "modules.gemini_chart_analyzer.core.utils.chart_paths.get_analysis_results_dir", return_value=tmp_path
        ):
            response = client.get("/api/batch/list")

            assert response.status_code == 200
            data = response.json()
            # Should only include valid file
            assert data["count"] == 1
            assert data["results"][0]["filename"] == "valid.json"

    def test_error_list_failure(self, client, tmp_path):
        """Test error when listing fails."""
        batch_scan_dir = tmp_path / "batch_scan"
        batch_scan_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists so glob is called

        with (
            patch(
                "modules.gemini_chart_analyzer.core.utils.chart_paths.get_analysis_results_dir", return_value=tmp_path
            ),
            patch("pathlib.Path.glob", side_effect=Exception("File system error")),
        ):
            response = client.get("/api/batch/list")

            assert response.status_code == 500
            assert "Lỗi khi liệt kê kết quả" in response.json()["detail"]


class TestBatchScanBackgroundThread:
    """Test POST /api/batch/scan with background thread (new implementation)."""

    def test_batch_scan_returns_session_id_immediately(self, client, tmp_path):
        """Test that batch scan returns session_id immediately without blocking."""
        import time

        mock_scanner_result = {
            "summary": {"total_scanned": 10, "long_count": 3, "short_count": 2, "none_count": 5},
            "long_symbols": ["BTC/USDT"],
            "short_symbols": ["ADA/USDT"],
            "long_symbols_with_confidence": [],
            "short_symbols_with_confidence": [],
            "all_results": {},
            "results_file": str(tmp_path / "results.json"),
        }

        with (
            patch("web.api.batch_scanner.MarketBatchScanner") as mock_scanner_class,
            patch("web.api.batch_scanner.get_log_manager") as mock_log_mgr,
            patch("web.api.batch_scanner.get_task_manager") as mock_task_mgr,
        ):
            # Setup mocks
            mock_scanner = Mock()
            mock_scanner.scan_market.return_value = mock_scanner_result
            mock_scanner_class.return_value = mock_scanner

            mock_log_manager = Mock()
            mock_log_manager.create_log_file.return_value = tmp_path / "test.log"
            mock_log_mgr.return_value = mock_log_manager

            mock_task_manager = Mock()
            mock_task_mgr.return_value = mock_task_manager

            # Make request
            start_time = time.time()
            request_data = {"timeframe": "1h", "max_symbols": 2}
            response = client.post("/api/batch/scan", json=request_data)
            elapsed = time.time() - start_time

            # Should return quickly - use configurable threshold for CI/slower machines
            # Default to 0.5s, allow override via BATCH_SCAN_TIMEOUT_THRESHOLD env var
            timeout_threshold = float(os.getenv("BATCH_SCAN_TIMEOUT_THRESHOLD", "0.5"))
            assert elapsed < timeout_threshold, f"Request took {elapsed:.3f}s, expected < {timeout_threshold}s"
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert "session_id" in data
            assert data["status"] == "running"
            assert "message" in data

            # Verify task was started
            mock_task_manager.start_task.assert_called_once()
            call_args = mock_task_manager.start_task.call_args
            assert call_args[0][1] is not None  # task_func
            assert call_args[0][2] == "scan"  # command_type as positional arg

    def test_batch_scan_creates_log_file(self, client, tmp_path):
        """Test that batch scan creates log file."""
        mock_scanner_result = {
            "summary": {"total_scanned": 1},
            "long_symbols": [],
            "short_symbols": [],
            "long_symbols_with_confidence": [],
            "short_symbols_with_confidence": [],
            "all_results": {},
            "results_file": str(tmp_path / "results.json"),
        }

        with (
            patch("web.api.batch_scanner.MarketBatchScanner") as mock_scanner_class,
            patch("web.api.batch_scanner.get_log_manager") as mock_log_mgr,
            patch("web.api.batch_scanner.get_task_manager") as mock_task_mgr,
        ):
            mock_scanner = Mock()
            mock_scanner.scan_market.return_value = mock_scanner_result
            mock_scanner_class.return_value = mock_scanner

            mock_log_manager = Mock()
            log_path = tmp_path / "test.log"
            mock_log_manager.create_log_file.return_value = log_path
            mock_log_mgr.return_value = mock_log_manager

            mock_task_manager = Mock()
            mock_task_mgr.return_value = mock_task_manager

            request_data = {"timeframe": "1h"}
            response = client.post("/api/batch/scan", json=request_data)

            assert response.status_code == 200
            # Verify log file was created
            mock_log_manager.create_log_file.assert_called_once()


class TestBatchScanStatusEndpoint:
    """Test GET /api/batch/scan/{session_id}/status endpoint."""

    def test_get_status_running(self, client):
        """Test getting status of running task."""
        from web.utils.task_manager import get_task_manager

        task_manager = get_task_manager()
        session_id = f"test-session-{uuid.uuid4()}"

        try:
            # Start a task
            def long_task():
                import time

                time.sleep(0.2)
                return {"result": "done"}

            task_manager.start_task(session_id, long_task, "scan")

            # Get status immediately (should be running)
            response = client.get(f"/api/batch/scan/{session_id}/status")

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["session_id"] == session_id
            assert data["status"] == "running"
            assert "started_at" in data
        finally:
            # Cleanup to prevent test pollution
            task_manager.cleanup_task(session_id)

    def test_get_status_completed(self, client):
        """Test getting status of completed task."""
        from web.utils.task_manager import get_task_manager

        task_manager = get_task_manager()
        session_id = f"test-session-{uuid.uuid4()}"

        try:
            # Start and complete a task
            def quick_task():
                return {"success": True, "summary": {"total_scanned": 5}, "long_symbols": ["BTC/USDT"]}

            task_manager.start_task(session_id, quick_task, "scan")

            # Wait for completion
            import time

            time.sleep(0.1)

            response = client.get(f"/api/batch/scan/{session_id}/status")

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["status"] == "completed"
            assert "result" in data
            assert data["result"]["success"] is True
            assert "completed_at" in data
        finally:
            # Cleanup to prevent test pollution
            task_manager.cleanup_task(session_id)

    def test_get_status_error(self, client):
        """Test getting status of failed task."""
        from web.utils.task_manager import get_task_manager

        task_manager = get_task_manager()
        session_id = f"test-session-{uuid.uuid4()}"

        try:
            # Start a failing task
            def failing_task():
                raise ValueError("Test error message")

            task_manager.start_task(session_id, failing_task, "scan")

            # Wait for error
            import time

            time.sleep(0.1)

            response = client.get(f"/api/batch/scan/{session_id}/status")

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["status"] == "error"
            assert "error" in data
            assert "Test error message" in data["error"]
        finally:
            # Cleanup to prevent test pollution
            task_manager.cleanup_task(session_id)

    def test_get_status_nonexistent(self, client):
        """Test getting status of non-existent session."""
        response = client.get("/api/batch/scan/nonexistent-session/status")

        assert response.status_code == 404
        data = response.json()
        assert "not found" in data["detail"].lower()

    def test_get_status_completed_with_set_result(self, client):
        """Test getting status when result is set via set_result() (not from task_func return).

        This test covers the race condition fix where:
        1. set_result() is called to set the result (simulating batch_scanner behavior)
        2. task_func() returns None (simulating run_scan() which doesn't return)
        3. Result should NOT be overwritten by run_task()
        4. Status endpoint should return the result that was set via set_result()
        """
        from web.utils.task_manager import get_task_manager

        task_manager = get_task_manager()
        session_id = f"test-session-{uuid.uuid4()}"

        try:
            # Start a task that returns None (simulating run_scan())
            def task_that_returns_none():
                import time

                time.sleep(0.05)  # Simulate some work
                return None  # run_scan() doesn't return anything

            task_manager.start_task(session_id, task_that_returns_none, "scan")

            # Wait a bit for task to start
            import time

            time.sleep(0.01)

            # Manually set result (simulating what batch_scanner does via set_result())
            result_data = {
                "success": True,
                "mode": "multi-timeframe",
                "summary": {"total_scanned": 10, "long_count": 3, "short_count": 2, "none_count": 5},
                "long_symbols": ["BTC/USDT", "ETH/USDT"],
                "short_symbols": ["ADA/USDT"],
                "long_symbols_with_confidence": [{"symbol": "BTC/USDT", "confidence": 0.8}],
                "short_symbols_with_confidence": [{"symbol": "ADA/USDT", "confidence": 0.7}],
                "all_results": {},
                "results_file": "/path/to/results.json",
            }
            task_manager.set_result(session_id, result_data)

            # Wait for task to complete
            time.sleep(0.1)

            # Get status - should have the result from set_result(), not None
            response = client.get(f"/api/batch/scan/{session_id}/status")

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["status"] == "completed"
            assert "result" in data
            # Verify result is the one from set_result(), not None
            assert data["result"] == result_data
            assert data["result"]["summary"]["total_scanned"] == 10
            assert len(data["result"]["long_symbols"]) == 2
            assert "completed_at" in data
        finally:
            # Cleanup to prevent test pollution
            task_manager.cleanup_task(session_id)


class TestCancelBatchScanEndpoint:
    """Test POST /api/batch/scan/{session_id}/cancel endpoint."""

    def test_cancel_scan_success(self, client):
        """Test successfully cancelling a running scan."""
        from web.utils.task_manager import get_task_manager

        task_manager = get_task_manager()
        session_id = f"test-session-{uuid.uuid4()}"

        try:
            # Start a long-running task
            def long_task():
                import time

                time.sleep(1.0)  # Long task
                return {"result": "done"}

            task_manager.start_task(session_id, long_task, "scan")

            # Wait a bit to ensure task is running
            import time

            time.sleep(0.05)

            # Cancel the task
            response = client.post(f"/api/batch/scan/{session_id}/cancel")

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["session_id"] == session_id
            assert data["status"] == "cancelled"
            assert "message" in data

            # Verify task is marked as cancelled
            status = task_manager.get_status(session_id)
            assert status is not None
            assert task_manager.is_cancelled(session_id) is True
        finally:
            # Cleanup to prevent test pollution
            task_manager.cleanup_task(session_id)

    def test_cancel_scan_not_found(self, client):
        """Test cancelling a non-existent session."""
        response = client.post("/api/batch/scan/nonexistent-session/cancel")

        assert response.status_code == 404
        data = response.json()
        assert "not found" in data["detail"].lower()

    def test_cancel_scan_already_completed(self, client):
        """Test cancelling a scan that is already completed (should fail)."""
        from web.utils.task_manager import get_task_manager

        task_manager = get_task_manager()
        session_id = f"test-session-{uuid.uuid4()}"

        try:
            # Start and complete a task
            def quick_task():
                return {"result": "done"}

            task_manager.start_task(session_id, quick_task, "scan")

            # Wait for completion
            import time

            time.sleep(0.1)

            # Try to cancel (should fail)
            response = client.post(f"/api/batch/scan/{session_id}/cancel")

            assert response.status_code == 400
            data = response.json()
            assert "cannot cancel" in data["detail"].lower() or "only running" in data["detail"].lower()
        finally:
            # Cleanup to prevent test pollution
            task_manager.cleanup_task(session_id)

    def test_cancel_scan_already_cancelled(self, client):
        """Test cancelling a scan that is already cancelled (should fail)."""
        from web.utils.task_manager import get_task_manager

        task_manager = get_task_manager()
        session_id = f"test-session-{uuid.uuid4()}"

        try:
            # Start a task
            def long_task():
                import time

                time.sleep(1.0)
                return {"result": "done"}

            task_manager.start_task(session_id, long_task, "scan")

            # Cancel it once
            import time

            time.sleep(0.05)
            task_manager.cancel_task(session_id)

            # Try to cancel again (should fail)
            response = client.post(f"/api/batch/scan/{session_id}/cancel")

            assert response.status_code == 400
            data = response.json()
            assert "cannot cancel" in data["detail"].lower() or "only running" in data["detail"].lower()
        finally:
            # Cleanup to prevent test pollution
            task_manager.cleanup_task(session_id)


class TestAutoCleanupIntegration:
    """Test auto-cleanup integration with API endpoints."""

    def test_api_endpoint_triggers_auto_cleanup(self, client, tmp_path):
        """Test that calling API endpoint triggers auto-cleanup of old logs."""
        import os

        # Set up logs directory
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()

        # Create old log file (2 hours ago) that should be cleaned up
        old_log = logs_dir / "scan_old-session.log"
        old_log.write_text("Old log content")
        old_time = time.time() - (2 * 3600)  # 2 hours ago
        os.utime(old_log, (old_time, old_time))

        # Create recent log file (should NOT be cleaned up)
        recent_log = logs_dir / "scan_recent-session.log"
        recent_log.write_text("Recent log content")
        recent_time = time.time() - (30 * 60)  # 30 minutes ago
        os.utime(recent_log, (recent_time, recent_time))

        # Patch get_log_manager to use real instance with cleanup enabled
        from web.utils.log_manager import LogFileManager

        real_manager = LogFileManager(
            logs_dir=str(logs_dir),
            auto_cleanup_before_new=True,
            max_log_age_hours=1,  # Clean up logs older than 1 hour
            start_cleanup_thread=False,  # Avoid background thread in tests
        )

        # Reset last cleanup time to allow cleanup to run
        real_manager.reset_cleanup_timer_for_testing()

        # Mock other dependencies but use real log manager
        with (
            patch("web.api.batch_scanner.MarketBatchScanner") as mock_scanner_class,
            patch("web.api.batch_scanner.get_log_manager", return_value=real_manager),
            patch("web.api.batch_scanner.get_task_manager") as mock_task_mgr,
        ):
            # Setup mocks
            mock_scanner = Mock()
            mock_scanner.scan_market.return_value = {
                "summary": {"total_scanned": 10, "long_count": 3, "short_count": 2, "none_count": 5},
                "long_symbols": ["BTC/USDT"],
                "short_symbols": ["ETH/USDT"],
                "timeframe": "1h",
                "results_file": str(tmp_path / "results.json"),
            }
            mock_scanner_class.return_value = mock_scanner

            mock_task_manager = Mock()
            mock_task_mgr.return_value = mock_task_manager

            # Call API endpoint - this should trigger cleanup
            response = client.post("/api/batch/scan", json={"timeframe": "1h", "max_symbols": 10})

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert "session_id" in data

            # Verify old log was deleted (older than 1 hour)
            assert not old_log.exists(), "Old log file should have been cleaned up"

            # Verify recent log still exists (newer than 1 hour)
            assert recent_log.exists(), "Recent log file should not be cleaned up"

            # Verify new log file was created
            session_id = data["session_id"]
            new_log = logs_dir / f"scan_{session_id}.log"
            assert new_log.exists(), "New log file should have been created"
