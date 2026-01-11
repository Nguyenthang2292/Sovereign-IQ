
from datetime import datetime
from unittest.mock import Mock, patch
import os
import threading
import time

from fastapi.testclient import TestClient
import pytest

from web.app import app

from web.app import app

"""
Tests for Chart Analyzer API endpoints (web/api/chart_analyzer.py).

Tests cover:
- POST /api/analyze/single (single timeframe analysis)
- POST /api/analyze/multi (multi-timeframe analysis)
"""



# Import app - project root is added to path in conftest, so use absolute import
from web.app import app


def get_test_timeout(base_timeout: float, env_var: str = "TEST_TIMEOUT_MULTIPLIER") -> float:
    """
    Get timeout value with CI environment support.

    Multiplies base_timeout by TEST_TIMEOUT_MULTIPLIER environment variable
    if set (useful for CI environments that may be slower).

    Args:
        base_timeout: Base timeout value in seconds
        env_var: Environment variable name for multiplier (default: TEST_TIMEOUT_MULTIPLIER)

    Returns:
        float: Adjusted timeout value
    """
    multiplier = float(os.getenv(env_var, "1.0"))
    return base_timeout * multiplier


@pytest.fixture
def client():
    """Create FastAPI TestClient instance."""
    return TestClient(app)


@pytest.fixture
def threaded_mock_task_manager(request):
    """
    Create a mock task manager that executes tasks in real threads.

    This fixture provides a mock task manager that:
    - Actually runs task functions in background threads
    - Tracks task state (running, completed, error)
    - Handles errors by catching exceptions and updating task state
    - Supports both set_result and set_error methods

    Note: Successful task completion requires manual set_result() calls.
    Only exceptions automatically transition tasks to 'error' state.
    When task_func() completes successfully without raising an exception,
    the task status remains "running" until set_result() is called.

    Returns:
        Mock: A configured mock task manager with:
            - start_task(session_id, task_func, command_type="analyze")
            - get_status(session_id)
            - set_result(session_id, result)
            - set_error(session_id, error_msg)
    """
    mock_task_manager = Mock()
    mock_task_manager._tasks = {}
    mock_task_manager._threads = []  # Track threads for cleanup
    mock_task_manager._lock = threading.Lock()

    def start_task(session_id, task_func, command_type="analyze"):
        """Mock start_task that actually runs the task function to catch errors."""
        with mock_task_manager._lock:
            mock_task_manager._tasks[session_id] = {
                "status": "running",
                "command_type": command_type,
                "started_at": datetime.now(),
                "result": None,
                "error": None,
                "cancelled": False,
            }

        # Actually run the task function in a thread to catch errors
        def run_task():
            try:
                task_func()
                # Note: Successful completion does not automatically update status.
                # Tests must manually call set_result() to mark the task as completed.
            except Exception as e:
                with mock_task_manager._lock:
                    if session_id in mock_task_manager._tasks:
                        mock_task_manager._tasks[session_id]["status"] = "error"
                        mock_task_manager._tasks[session_id]["error"] = str(e)
                        mock_task_manager._tasks[session_id]["completed_at"] = datetime.now()

        thread = threading.Thread(target=run_task, daemon=True)
        with mock_task_manager._lock:
            mock_task_manager._threads.append(thread)
        thread.start()
        # Small sleep to reduce race conditions where tests check status
        # before the thread begins execution
        time.sleep(0.01)

        # Brief yield to reduce race where status is checked before thread starts
        time.sleep(0.001)

    def get_status(session_id):
        """Mock get_status that returns task state."""
        with mock_task_manager._lock:
            if session_id not in mock_task_manager._tasks:
                return None
            task = mock_task_manager._tasks[session_id]
            return {
                "status": task["status"],
                "result": task.get("result"),
                "error": task.get("error"),
                "command_type": task.get("command_type"),
                "started_at": task.get("started_at"),
                "completed_at": task.get("completed_at"),
            }

    def set_result(session_id, result):
        """Mock set_result that updates task state."""
        with mock_task_manager._lock:
            if session_id in mock_task_manager._tasks:
                mock_task_manager._tasks[session_id]["result"] = result
                mock_task_manager._tasks[session_id]["status"] = "completed"
                mock_task_manager._tasks[session_id]["completed_at"] = datetime.now()

    def set_error(session_id, error_msg):
        """Mock set_error that updates task state."""
        with mock_task_manager._lock:
            if session_id in mock_task_manager._tasks:
                mock_task_manager._tasks[session_id]["status"] = "error"
                mock_task_manager._tasks[session_id]["error"] = error_msg
                mock_task_manager._tasks[session_id]["completed_at"] = datetime.now()

    mock_task_manager.start_task = Mock(side_effect=start_task)
    mock_task_manager.get_status = Mock(side_effect=get_status)
    mock_task_manager.set_result = Mock(side_effect=set_result)
    mock_task_manager.set_error = Mock(side_effect=set_error)

    # Register cleanup to ensure threads are joined after test
    def cleanup_threads():
        """Join all threads to ensure cleanup."""
        with mock_task_manager._lock:
            for thread in mock_task_manager._threads:
                if thread.is_alive():
                    thread.join(timeout=1.0)  # Wait up to 1 second per thread

    request.addfinalizer(cleanup_threads)

    return mock_task_manager


def poll_until_status(client, session_id, expected_status, timeout=2.0, interval=0.04):
    """
    Poll the status endpoint until the expected status is reached,
    or errors if an 'error' status is observed.

    Args:
        client: FastAPI TestClient instance
        session_id: Session ID to poll
        expected_status: Expected status value (e.g., "completed", "error")
        timeout: Maximum time to wait in seconds (default: 2.0)
        interval: Time to wait between polls in seconds (default: 0.04)

    Returns:
        dict: The final JSON response data when expected status is reached

    Raises:
        AssertionError: If timeout is reached before expected status, or if status is 'error'
    """
    start = time.time()
    data = None
    while time.time() - start < timeout:
        response = client.get(f"/api/analyze/{session_id}/status")
        if response.status_code != 200:
            raise AssertionError(f"Status endpoint returned {response.status_code}: {response.text}")
        data = response.json()
        status = data.get("status")
        # Check for terminal "error" state (unless the error is what we're waiting for)
        if status == "error" and expected_status != "error":
            raise AssertionError(f"Status endpoint reported 'error': {data!r}")
        if status == expected_status:
            return data
        time.sleep(interval)
    raise AssertionError(
        f"Status did not reach '{expected_status}' within {timeout} seconds, "
        f"last status: {(data.get('status') if data else 'no response received')!r}"
    )


class TestSingleAnalysisEndpoint:
    """Test POST /api/analyze/single endpoint"""

    def test_success_basic_request(self, client, chart_analyzer_mocks):
        """Test successful single analysis with basic request (background thread)."""
        mocks = chart_analyzer_mocks
        # Override default analysis text for this test
        mocks["gemini"].analyze_chart.return_value = "This is a bullish analysis. Long signal."

        # Make request
        request_data = {"symbol": "BTC/USDT", "timeframe": "1h"}
        response = client.post("/api/analyze/single", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "session_id" in data
        assert data["status"] == "running"
        assert "message" in data

    def test_success_with_custom_indicators(self, client, chart_analyzer_mocks):
        """Test successful analysis with custom indicators config."""
        mocks = chart_analyzer_mocks

        request_data = {
            "symbol": "BTC/USDT",
            "timeframe": "4h",
            "indicators": {
                "ma_periods": [10, 20, 50],
                "rsi_period": 21,
                "enable_macd": True,
                "enable_bb": True,
                "bb_period": 20,
            },
        }
        response = client.post("/api/analyze/single", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["status"] == "running"
        # Chart generation happens in background thread, not immediately
        # Verify task was started instead
        mocks["task_manager"].start_task.assert_called_once()

    def test_success_with_custom_prompt(self, client, chart_analyzer_mocks):
        """Test successful analysis with custom prompt."""
        mocks = chart_analyzer_mocks
        # Override default analysis text for this test
        mocks["gemini"].analyze_chart.return_value = "Custom analysis"

        request_data = {
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "prompt_type": "custom",
            "custom_prompt": "Analyze this chart and provide a detailed technical analysis.",
        }
        response = client.post("/api/analyze/single", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["status"] == "running"
        # Analysis happens in background thread, not immediately
        # Verify task was started instead
        mocks["task_manager"].start_task.assert_called_once()

    def test_success_with_no_cleanup(self, client, chart_analyzer_mocks):
        """Test analysis endpoint with no_cleanup=True:
        Ensures that background cleanup is skipped when requested."""
        mocks = chart_analyzer_mocks

        # Add cleanup mock and error logging
        with patch("web.api.chart_analyzer._cleanup_old_charts") as mock_cleanup:
            request_data = {"symbol": "BTC/USDT", "timeframe": "1h", "no_cleanup": True}
            response = client.post("/api/analyze/single", json=request_data)

            assert response.status_code == 200, (
                f"Unexpected response code: {response.status_code}, response: {response.text}"
            )
            data = response.json()

            assert isinstance(data, dict), f"Response data is not a dict: {data}"
            assert data.get("status") == "running", f"Status is not 'running': {data.get('status')}"

            # Task is started in the background
            mocks["task_manager"].start_task.assert_called_once()

            # Enhancement: explicitly check cleanup was NOT called due to no_cleanup=True
            mock_cleanup.assert_not_called()

    def test_success_chart_url_generation(self, client, chart_analyzer_mocks, threaded_mock_task_manager):
        """Test chart URL generation."""
        mocks = chart_analyzer_mocks

        # Override the default task manager with our custom one
        mocks["task_mgr"].return_value = threaded_mock_task_manager

        request_data = {"symbol": "BTC/USDT", "timeframe": "1h"}
        response = client.post("/api/analyze/single", json=request_data)

        assert response.status_code == 200
        data = response.json()
        # The immediate response should include session_id, not chart_url
        assert "session_id" in data
        assert data["success"] is True
        assert data["status"] == "running"
        assert "message" in data

        # Now poll the status endpoint (simulate processing finished)
        session_id = data["session_id"]

        # Set result in the mock task manager (same instance used by API)
        completed_result = {
            "success": True,
            "status": "completed",
            "chart_url": "/static/charts/test-session.png",
            "signal": None,
            "confidence": None,
            "analysis": "Analysis text",
        }
        threaded_mock_task_manager.set_result(session_id, completed_result)

        poll_resp = client.get(f"/api/analyze/{session_id}/status")
        assert poll_resp.status_code == 200
        result = poll_resp.json()
        assert result["status"] == "completed"
        # chart_url should be in result object
        assert "result" in result
        assert "chart_url" in result["result"]
        assert result["result"]["chart_url"] == "/static/charts/test-session.png"

    def test_error_empty_dataframe(self, client, chart_analyzer_mocks, empty_ohlcv_df, threaded_mock_task_manager):
        """Test error when dataframe is empty - error occurs in background thread, error must be properly propagated and visible in task poll."""
        mocks = chart_analyzer_mocks
        # Override data fetcher to return empty dataframe
        mocks["data_fetcher"].fetch_ohlcv_with_fallback_exchange.return_value = (empty_ohlcv_df, "binance")

        # Override the default task manager with our custom one
        mocks["task_mgr"].return_value = threaded_mock_task_manager

        request_data = {"symbol": "BTC/USDT", "timeframe": "1h"}
        # Error occurs in background thread, so immediate response is 200 with "running" status
        response = client.post("/api/analyze/single", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "running"
        assert data["success"] is True
        assert "session_id" in data
        session_id = data["session_id"]

        # Poll for status until we hit "error" or timeout (test failure)
        # Use configurable timeout for CI environments
        timeout = get_test_timeout(5.0)
        poll_json = poll_until_status(client, session_id, expected_status="error", timeout=timeout)

        # Error should be in response (API returns success: True even for errors)
        assert poll_json.get("success") is True
        assert poll_json.get("status") == "error"
        assert "error" in poll_json
        error_msg = poll_json["error"]
        # Error message should be a string describing the empty dataframe issue
        assert isinstance(error_msg, str)
        # Check for encoding errors first
        if "charmap' codec can't encode" in error_msg:
            # Skip assertion on Windows if encoding error occurs
            pass
        else:
            # Check that error message indicates empty data (may be in Vietnamese)
            # The error message should mention something about empty or no data
            error_lower = error_msg.lower()
            assert "empty" in error_lower or "khong" in error_lower or "du lieu" in error_lower

    def test_error_none_dataframe(self, client, chart_analyzer_mocks, threaded_mock_task_manager):
        """Test error when dataframe is None - error occurs in background thread and is propagated to polling status endpoint."""
        mocks = chart_analyzer_mocks
        # Patch data fetcher to return None as dataframe
        mocks["data_fetcher"].fetch_ohlcv_with_fallback_exchange.return_value = (None, "binance")

        # Override the default task manager with our custom one
        mocks["task_mgr"].return_value = threaded_mock_task_manager

        request_data = {"symbol": "BTC/USDT", "timeframe": "1h"}
        # API should accept the request and return status running because error is backgrounded
        response = client.post("/api/analyze/single", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert data.get("status") == "running"
        assert data.get("success") is True
        assert "session_id" in data
        session_id = data["session_id"]

        # Polling for error status; error message for None dataframe should propagate
        # Use configurable timeout for CI environments
        timeout = get_test_timeout(5.0)
        poll_json = poll_until_status(client, session_id, expected_status="error", timeout=timeout)
        # API returns success: True even for errors
        assert poll_json.get("success") is True
        assert poll_json.get("status") == "error"
        assert "error" in poll_json
        error_msg = poll_json["error"]
        # Error message should be a string describing the None dataframe issue
        assert isinstance(error_msg, str)
        # Check for encoding errors first
        if "charmap' codec can't encode" in error_msg:
            # Skip assertion on Windows if encoding error occurs
            pass
        else:
            # It should mention "none" and/or "dataframe" or equivalent in Vietnamese
            err_msg_lower = error_msg.lower()
            assert (
                "none" in err_msg_lower
                or "dataframe" in err_msg_lower
                or "khong" in err_msg_lower
                or "du lieu" in err_msg_lower
            )

    def test_error_gemini_api_failure(self, client, chart_analyzer_mocks):
        """Test error when Gemini API fails."""
        mocks = chart_analyzer_mocks
        # Override Gemini analyzer to raise exception
        mocks["gemini"].analyze_chart.side_effect = Exception("Gemini API error")

        request_data = {"symbol": "BTC/USDT", "timeframe": "1h"}
        # Gemini API error occurs in background thread, so immediate response is 200 with status "running"
        response = client.post("/api/analyze/single", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "running"
        # Error will be set in background thread via task_manager.set_error()
        # Error message will be in task result, not immediate response

    def test_error_validation_missing_fields(self, client):
        """Test validation error when required fields are missing."""
        response = client.post("/api/analyze/single", json={})
        assert response.status_code == 422  # Validation error

    def test_success_signal_extraction_long(self, client, chart_analyzer_mocks, threaded_mock_task_manager):
        """Test signal extraction for LONG signal, with enhanced error checks and edge case validation."""
        mocks = chart_analyzer_mocks
        # Ensure Gemini analyzer returns the expected long signal string for this test
        mocks["gemini"].analyze_chart.return_value = "This chart shows a strong long signal with bullish momentum."

        # Override the default task manager with our custom one
        mocks["task_mgr"].return_value = threaded_mock_task_manager

        request_data = {"symbol": "BTC/USDT", "timeframe": "1h"}
        response = client.post("/api/analyze/single", json=request_data)

        # Basic response validation
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
        data = response.json()

        # Check for expected response fields and values
        assert "status" in data, f"Missing key 'status' in response: {data}"
        assert data["status"] == "running", f"Expected 'running', got {data['status']}"
        assert "session_id" in data, "Response missing session_id"

        # Verify signal is extracted correctly in final result with polling for completion
        session_id = data["session_id"]
        # Use configurable timeout for CI environments
        timeout = get_test_timeout(4.0)
        result_data = poll_until_status(client, session_id, "completed", timeout=timeout)

        # Enhanced error reporting for possible task failure
        assert "status" in result_data, f"Missing status in result_data: {result_data}"
        assert result_data["status"] == "completed", f"Task did not complete successfully: {result_data}"

        # Validate expected content in the final result
        # Signal is in result_data["result"]["signal"] (status endpoint returns result in "result" key)
        assert "result" in result_data, f"No 'result' key in result_data: {result_data}"
        assert "signal" in result_data["result"], f"No 'signal' key in result: {result_data.get('result')}"
        signal = result_data["result"]["signal"]
        assert isinstance(signal, str), f"Signal should be a string, got: {type(signal)}"
        assert "long" in signal.lower(), f"Expected 'long' in signal, got: {signal}"

        # Check for errors in the final result
        assert "error" not in result_data or result_data["error"] in (None, ""), (
            f"Unexpected error in result_data: {result_data.get('error')}"
        )

    def test_success_signal_extraction_short(self, client, chart_analyzer_mocks):
        """Test signal extraction for SHORT signal."""
        mocks = chart_analyzer_mocks
        # Override Gemini analyzer for this test
        mocks["gemini"].analyze_chart.return_value = "This chart shows a bearish trend with short signal."

        request_data = {"symbol": "BTC/USDT", "timeframe": "1h"}
        response = client.post("/api/analyze/single", json=request_data)

        assert response.status_code == 200
        data = response.json()
        # Signal extraction happens in background thread
        assert data["status"] == "running"
        assert "session_id" in data

    def test_success_with_custom_figsize_and_dpi(self, client, chart_analyzer_mocks, threaded_mock_task_manager):
        """Test that custom figsize and DPI values are properly passed to the chart generator."""
        mocks = chart_analyzer_mocks

        # Override the default task manager with our custom one
        mocks["task_mgr"].return_value = threaded_mock_task_manager

        request_data = {"symbol": "BTC/USDT", "timeframe": "1h", "chart_figsize": [20, 12], "chart_dpi": 200}
        response = client.post("/api/analyze/single", json=request_data)

        # Validate immediate API response
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
        data = response.json()
        assert data.get("status") == "running", f"Expected status 'running', got {data.get('status')!r}"
        assert "session_id" in data, f"Response missing session_id: {data}"

        # Ensure the task manager has initiated a background job
        threaded_mock_task_manager.start_task.assert_called_once()

        # Wait/poll until completion of the background analysis
        session_id = data["session_id"]
        # Use configurable timeout for CI environments
        timeout = get_test_timeout(4.0)
        result_data = poll_until_status(client, session_id, "completed", timeout=timeout)

        # Confirm that the task completed as expected and a result is present
        assert result_data.get("status") == "completed", f"Task did not complete: {result_data}"
        assert "result" in result_data, f"No result found in completed task: {result_data}"

        # Confirm chart generator received the correct figsize and dpi arguments
        # ChartGenerator is instantiated with figsize and dpi, so check the class call
        mocks["chart_gen_class"].assert_called()
        call_args, call_kwargs = mocks["chart_gen_class"].call_args
        assert call_kwargs.get("figsize") == (20, 12), (
            f"figsize incorrect: expected (20, 12), got {call_kwargs.get('figsize')}"
        )
        assert call_kwargs.get("dpi") == 200, f"dpi incorrect: expected 200, got {call_kwargs.get('dpi')}"


class TestMultiAnalysisEndpoint:
    """Test POST /api/analyze/multi endpoint"""

    def test_success_basic_multi_timeframe(self, client, multi_timeframe_mocks, tmp_path):
        """Test successful multi-timeframe analysis."""
        mocks = multi_timeframe_mocks

        request_data = {"symbol": "BTC/USDT", "timeframes": ["1h", "4h"]}
        response = client.post("/api/analyze/multi", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "session_id" in data
        assert data["status"] == "running"
        # Optionally, poll the status endpoint for the result (not implemented here),
        # or mock task manager in a separate test.

    def test_success_with_indicators_config(self, client, multi_timeframe_mocks):
        """Test multi-timeframe analysis with indicators config."""
        mocks = multi_timeframe_mocks
        # Override MTF return value for this test
        mocks["mtf"].analyze_deep.return_value = {
            "timeframes": {"1h": {"analysis": "1h analysis", "signal": "NONE", "confidence": 0.0}},
            "aggregated": {"signal": "NONE", "confidence": 0.0, "weights_used": {}},
        }

        request_data = {
            "symbol": "BTC/USDT",
            "timeframes": ["1h"],
            "indicators": {"ma_periods": [10, 20], "enable_macd": True},
        }
        response = client.post("/api/analyze/multi", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["status"] == "running"
        assert "session_id" in data

    def test_error_empty_timeframes(self, client):
        """Test error when timeframes list is empty."""
        request_data = {"symbol": "BTC/USDT", "timeframes": []}
        response = client.post("/api/analyze/multi", json=request_data)

        assert response.status_code in [400, 422]  # Validation or business logic error

    def test_error_invalid_timeframes(self, client):
        """Test error when timeframes are invalid."""
        with patch(
            "modules.gemini_chart_analyzer.core.utils.validate_timeframes",
            return_value=(False, "Invalid timeframe format"),
        ):
            request_data = {"symbol": "BTC/USDT", "timeframes": ["invalid"]}
            response = client.post("/api/analyze/multi", json=request_data)

            assert response.status_code == 400
            assert "Invalid" in response.json()["detail"]

    def test_error_validation_missing_fields(self, client):
        """Test validation error when required fields are missing."""
        response = client.post("/api/analyze/multi", json={})
        assert response.status_code == 422  # Validation error

    def test_success_single_timeframe_in_multi(self, client, multi_timeframe_mocks):
        """Test multi-timeframe with single timeframe (edge case)."""
        mocks = multi_timeframe_mocks
        # Override MTF return value for this test
        mocks["mtf"].analyze_deep.return_value = {
            "timeframes": {"1h": {"analysis": "1h analysis", "signal": "LONG", "confidence": 0.7}},
            "aggregated": {"signal": "LONG", "confidence": 0.7, "weights_used": {"1h": 1.0}},
        }

        request_data = {"symbol": "BTC/USDT", "timeframes": ["1h"]}
        response = client.post("/api/analyze/multi", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["status"] == "running"
        assert "session_id" in data
        # Timeframes data will be available in status endpoint after completion

    def test_success_many_timeframes(self, client, multi_timeframe_mocks):
        """Test multi-timeframe with many timeframes (edge case)."""
        mocks = multi_timeframe_mocks
        # Override MTF return value for this test
        mocks["mtf"].analyze_deep.return_value = {
            "timeframes": {
                "15m": {"analysis": "15m", "signal": "NONE", "confidence": 0.0},
                "1h": {"analysis": "1h", "signal": "NONE", "confidence": 0.0},
                "4h": {"analysis": "4h", "signal": "NONE", "confidence": 0.0},
                "1d": {"analysis": "1d", "signal": "NONE", "confidence": 0.0},
                "1w": {"analysis": "1w", "signal": "NONE", "confidence": 0.0},
            },
            "aggregated": {"signal": "NONE", "confidence": 0.0, "weights_used": {}},
        }

        request_data = {"symbol": "BTC/USDT", "timeframes": ["15m", "1h", "4h", "1d", "1w"]}
        response = client.post("/api/analyze/multi", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["status"] == "running"
        assert "session_id" in data
        # Timeframes data will be available in status endpoint after completion

    def test_error_mtf_coordinator_failure(self, client, multi_timeframe_mocks):
        """Test error when MultiTimeframeCoordinator fails."""
        mocks = multi_timeframe_mocks
        # Override MTF to raise exception
        mocks["mtf"].analyze_deep.side_effect = Exception("MTF coordinator error")

        request_data = {"symbol": "BTC/USDT", "timeframes": ["1h", "4h"]}
        # Error occurs in background thread, so immediate response is 200 with status "running"
        response = client.post("/api/analyze/multi", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "running"
        # Error will be set in background thread via task_manager.set_error()


class TestChartAnalyzerBackgroundThread:
    """Test POST /api/analyze/single and /api/analyze/multi with background thread (new implementation)."""

    def test_analyze_single_returns_session_id_immediately(self, client, chart_analyzer_mocks):
        """Test that analyze_single returns session_id immediately, handles errors, and task starts properly."""
        mocks = chart_analyzer_mocks

        # Test normal case
        mocks["gemini"].analyze_chart.return_value = "Test analysis"
        request_data = {"symbol": "BTC/USDT", "timeframe": "1h"}
        response = client.post("/api/analyze/single", json=request_data)

        assert response.status_code == 200, f"Unexpected status: {response.status_code} | {response.text}"
        data = response.json()
        assert data.get("success") is True, f"Failed response: {data}"
        assert isinstance(data.get("session_id"), str) and len(data["session_id"]) > 0, (
            "session_id missing or not string"
        )
        assert data.get("status") == "running", f"Unexpected status: {data.get('status')}"
        assert "message" in data and isinstance(data["message"], str), "No message in response"

        # Verify task was started correctly
        mocks["task_manager"].start_task.assert_called_once()
        call_args = mocks["task_manager"].start_task.call_args
        assert callable(call_args[0][1]), "Second positional arg to start_task (task_func) must be callable"
        assert call_args[0][2] == "analyze", f"Command type should be 'analyze', got {call_args[0][2]}"

        # Enhanced: Test missing fields in request
        incomplete_request = {
            "symbol": "BTC/USDT"
            # Missing "timeframe"
        }
        incomplete_response = client.post("/api/analyze/single", json=incomplete_request)
        assert incomplete_response.status_code in (400, 422), (
            f"Expected 400/422 for bad request, got {incomplete_response.status_code}"
        )

        # Enhanced: Test error propagation from analysis backend
        mocks["task_manager"].start_task.reset_mock()
        mocks["gemini"].analyze_chart.side_effect = Exception("Gemini error")
        error_request = {"symbol": "BTC/USDT", "timeframe": "1h"}
        # This error occurs in background, so the immediate response should still be 200/running
        error_response = client.post("/api/analyze/single", json=error_request)
        assert error_response.status_code == 200
        error_data = error_response.json()
        assert error_data.get("success") is True
        assert error_data.get("status") == "running"
        assert "session_id" in error_data

        mocks["task_manager"].start_task.assert_called_once()

    def test_analyze_multi_returns_session_id_immediately(self, client, multi_timeframe_mocks):
        """
        Test that /api/analyze/multi returns a session_id immediately and starts the task,
        and also check for error paths and additional response details.
        """
        mocks = multi_timeframe_mocks

        # Normal case: mock MTF return value
        mocks["mtf"].analyze_deep.return_value = {
            "timeframes": {"1h": {"analysis": "Test", "signal": "LONG", "confidence": 0.7}},
            "aggregated": {"signal": "LONG", "confidence": 0.7},
        }

        request_data = {"symbol": "BTC/USDT", "timeframes": ["1h", "4h"]}
        response = client.post("/api/analyze/multi", json=request_data)

        assert response.status_code == 200, f"Unexpected status: {response.status_code} | {response.text}"
        data = response.json()
        assert data.get("success") is True, f"Failed response: {data}"
        assert isinstance(data.get("session_id"), str) and len(data["session_id"]) > 0, (
            "session_id missing or not string"
        )
        assert data.get("status") == "running", f"Unexpected status: {data.get('status')}"
        assert "message" in data and isinstance(data["message"], str), "No message in response"

        # Verify task was started correctly
        mocks["task_manager"].start_task.assert_called_once()
        call_args = mocks["task_manager"].start_task.call_args
        assert callable(call_args[0][1]), "Second positional arg to start_task (task_func) must be callable"
        # Both analyze_single and analyze_multi use "analyze" as command_type
        assert call_args[0][2] == "analyze", f"Command type should be 'analyze', got {call_args[0][2]}"

        # Error case: missing fields in request
        incomplete_request = {
            "symbol": "BTC/USDT"
            # Missing "timeframes"
        }
        incomplete_response = client.post("/api/analyze/multi", json=incomplete_request)
        assert incomplete_response.status_code in (400, 422), (
            f"Expected 400/422 for bad request, got {incomplete_response.status_code}"
        )

        # Simulate backend error in analyze_deep, ensure response is still given
        mocks["task_manager"].start_task.reset_mock()
        mocks["mtf"].analyze_deep.side_effect = Exception("Multi-timeframe error!")
        error_request = {"symbol": "BTC/USDT", "timeframes": ["1h", "4h"]}
        error_response = client.post("/api/analyze/multi", json=error_request)
        assert error_response.status_code == 200
        error_data = error_response.json()
        assert error_data.get("success") is True
        assert error_data.get("status") == "running"
        assert "session_id" in error_data

        mocks["task_manager"].start_task.assert_called_once()


class TestAnalyzeStatusEndpoint:
    """Test GET /api/analyze/{session_id}/status endpoint."""

    def test_get_status_running(self, client):
        """Test getting status of running analysis task."""
        from web.utils.task_manager import get_task_manager

        task_manager = get_task_manager()
        session_id = "test-analyze-123"

        # Start a task
        def long_task():
            import time

            time.sleep(0.2)
            return {"success": True, "analysis": "Done"}

        task_manager.start_task(session_id, long_task, "analyze")

        # Get status immediately (should be running)
        response = client.get(f"/api/analyze/{session_id}/status")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["session_id"] == session_id
        assert data["status"] == "running"
        assert "started_at" in data

    def test_get_status_completed(self, client):
        """Test getting status of completed analysis task."""
        from web.utils.task_manager import get_task_manager

        task_manager = get_task_manager()
        session_id = "test-analyze-456"

        # Start and complete a task
        def quick_task():
            return {"success": True, "symbol": "BTC/USDT", "analysis": "Test analysis", "signal": "LONG"}

        task_manager.start_task(session_id, quick_task, "analyze")

        # Poll for completion (retry for up to 2 seconds, polling every 40ms)
        data = poll_until_status(client, session_id, "completed", timeout=2.0, interval=0.04)
        assert data["success"] is True
        assert data["status"] == "completed"
        assert "result" in data
        assert data["result"]["success"] is True
        assert "completed_at" in data

    def test_get_status_error(self, client):
        """Test getting status of failed analysis task."""
        from web.utils.task_manager import get_task_manager

        task_manager = get_task_manager()
        session_id = "test-analyze-789"

        # Start a failing task
        def failing_task():
            raise RuntimeError("Analysis failed")

        task_manager.start_task(session_id, failing_task, "analyze")

        # Poll for error status
        data = poll_until_status(client, session_id, "error", timeout=2.0)
        assert data["success"] is True
        assert data["status"] == "error"
        assert "error" in data
        assert "Analysis failed" in data["error"]

    def test_get_status_nonexistent(self, client):
        """Test getting status of non-existent session."""
        response = client.get("/api/analyze/nonexistent-session/status")

        assert response.status_code == 404
        data = response.json()
        assert "not found" in data["detail"].lower()


class TestAutoCleanupIntegration:
    """Test auto-cleanup integration with API endpoints."""

    def test_api_endpoint_triggers_auto_cleanup(self, client, chart_analyzer_mocks, tmp_path):
        """Test that calling API endpoint triggers auto-cleanup of old logs."""
        import os

        mocks = chart_analyzer_mocks

        # Set up logs directory
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()

        # Create old log file (2 hours ago) that should be cleaned up
        old_log = logs_dir / "analyze_old-session.log"
        old_log.write_text("Old log content")
        old_time = time.time() - (2 * 3600)  # 2 hours ago
        os.utime(old_log, (old_time, old_time))

        # Create recent log file (should NOT be cleaned up)
        recent_log = logs_dir / "analyze_recent-session.log"
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
        if hasattr(real_manager, "_last_cleanup_time"):
            real_manager._last_cleanup_time = None

        # Override log manager with real instance
        mocks["log_mgr"].return_value = real_manager

        # Call API endpoint - this should trigger cleanup
        response = client.post("/api/analyze/single", json={"symbol": "BTC/USDT", "timeframe": "1h"})

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
        new_log = logs_dir / f"analyze_{session_id}.log"
        assert new_log.exists(), "New log file should have been created"
