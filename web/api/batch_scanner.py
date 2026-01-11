
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional
import json
import os
import uuid

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from modules.common.ui.logging import log_error, log_info, log_warn
from modules.common.utils import normalize_timeframe
from modules.gemini_chart_analyzer.core.scanners.market_batch_scanner import MarketBatchScanner
from web.utils.cli_logger import CLILogger
from web.utils.log_manager import get_log_manager
from web.utils.task_manager import get_task_manager
from web.utils.log_manager import get_log_manager
from web.utils.task_manager import get_task_manager

"""
API routes for Batch Scanner (single and multi-timeframe market scanning).
"""




router = APIRouter()


# Request/Response models
class BatchScanRequest(BaseModel):
    """Request model for batch market scan."""

    timeframe: Optional[str] = Field(default=None, description="Single timeframe (e.g., 1h, 4h, 1d)")
    timeframes: Optional[List[str]] = Field(default=None, description="List of timeframes for multi-timeframe mode")
    max_symbols: Optional[int] = Field(default=None, description="Maximum number of symbols to scan (None = all)", ge=1)
    limit: Optional[int] = Field(default=500, description="Number of candles per symbol", ge=1, le=5000)
    cooldown: Optional[float] = Field(
        default=2.5, description="Cooldown between batch requests in seconds", ge=0.0, le=60.0
    )
    charts_per_batch: Optional[int] = Field(
        default=100, description="Number of charts per batch (single TF mode)", ge=1, le=1000
    )
    quote_currency: Optional[str] = Field(default="USDT", description="Quote currency filter")
    exchange_name: Optional[str] = Field(default="binance", description="Exchange name")


@router.post("/batch/scan")
async def batch_scan(request: BatchScanRequest):
    """
    Scan entire market and return LONG/SHORT signals.

    Supports both single timeframe and multi-timeframe modes.
    Runs in background thread and returns session_id immediately.

    Returns:
        Session ID and status. Use GET /api/batch/scan/{session_id}/status to check progress.
    """
    try:
        # Validate that either timeframe or timeframes is provided
        if not request.timeframe and not request.timeframes:
            raise HTTPException(
                status_code=400, detail="Either 'timeframe' (single) or 'timeframes' (multi) must be provided"
            )

        # Generate session_id immediately
        session_id = str(uuid.uuid4())

        # Normalize timeframe if single mode
        timeframe = None
        if request.timeframe:
            timeframe = normalize_timeframe(request.timeframe)

        # Normalize timeframes if multi mode
        timeframes = None
        if request.timeframes:
            from modules.gemini_chart_analyzer.core.utils import normalize_timeframes, validate_timeframes

            is_valid, error_msg = validate_timeframes(request.timeframes)
            if not is_valid:
                raise HTTPException(status_code=400, detail=error_msg or "Invalid timeframes")

            timeframes = normalize_timeframes(request.timeframes)

        # Create log file
        log_manager = get_log_manager()
        log_manager.create_log_file(session_id, "scan")

        # Get task manager
        task_manager = get_task_manager()

        # Define background task function
        def run_scan():
            """Run scan in background thread with logging."""
            cli_logger = CLILogger(session_id, "scan")

            try:
                with cli_logger.capture_output():
                    # Check if cancelled before starting
                    if task_manager.is_cancelled(session_id):
                        # Status is already set to 'cancelled' by cancel_task(), just return
                        return

                    # Initialize scanner
                    scanner = MarketBatchScanner(
                        charts_per_batch=request.charts_per_batch,
                        cooldown_seconds=request.cooldown,
                        quote_currency=request.quote_currency,
                        exchange_name=request.exchange_name,
                    )

                    # Create cancelled callback
                    def check_cancelled():
                        return task_manager.is_cancelled(session_id)

                    # Run scan with cancellation support
                    results = scanner.scan_market(
                        timeframe=timeframe,
                        timeframes=timeframes,
                        max_symbols=request.max_symbols,
                        limit=request.limit,
                        cancelled_callback=check_cancelled,
                    )

                    # Check if cancelled after scan
                    if task_manager.is_cancelled(session_id):
                        # If cancelled, status is already set by cancel_task(), just return
                        return

                    # Convert results to API response format
                    results_file = results.get("results_file", "")
                    results_url = None
                    if results_file and os.path.exists(results_file):
                        from modules.gemini_chart_analyzer.core.utils.chart_paths import get_analysis_results_dir

                        results_dir = get_analysis_results_dir()
                        try:
                            rel_path = os.path.relpath(results_file, results_dir)
                            rel_path = rel_path.replace("\\", "/")
                            from urllib.parse import quote

                            filename = quote(os.path.basename(rel_path))
                            results_url = f"/static/results/batch_scan/{filename}"
                        except Exception:
                            pass

                    # Prepare response
                    response = {
                        "success": True,
                        "mode": "multi-timeframe" if timeframes else "single-timeframe",
                        "timeframe": timeframe,
                        "timeframes": timeframes,
                        "summary": results.get("summary", {}),
                        "long_symbols": results.get("long_symbols", []),
                        "short_symbols": results.get("short_symbols", []),
                        "long_symbols_with_confidence": results.get("long_symbols_with_confidence", []),
                        "short_symbols_with_confidence": results.get("short_symbols_with_confidence", []),
                        "all_results": results.get("all_results", {}),
                        "results_file": results_file,
                        "results_url": results_url,
                    }

                    # Check if cancelled before saving result
                    if task_manager.is_cancelled(session_id):
                        # Task was cancelled, status is already set by cancel_task(), don't save result
                        return

                    # Save result to task manager
                    # Use set_result which atomically sets both result and status to 'completed'
                    task_manager.set_result(session_id, response)

                    # Log for debugging
                    log_info(f"Batch scan result saved for session {session_id}")

            except Exception as e:
                # Don't set error if task was cancelled
                if not task_manager.is_cancelled(session_id):
                    task_manager.set_error(session_id, str(e))
                log_error(f"Exception in batch scan task (session_id={session_id}): {e}")

            finally:
                # Cleanup scanner resources to free memory
                try:
                    if "scanner" in locals():
                        scanner.cleanup()
                except Exception as cleanup_error:
                    log_warn(f"Error cleaning up scanner resources: {cleanup_error}")

        # Start background task
        task_manager.start_task(session_id, run_scan, "scan")

        # Return immediately with session_id
        return {
            "success": True,
            "session_id": session_id,
            "status": "running",
            "message": "Scan started. Use GET /api/batch/scan/{session_id}/status to check progress.",
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi khởi động scan: {str(e)}")


@router.get("/batch/scan/{session_id}/status")
async def get_batch_scan_status(session_id: str):
    """
    Get status of a batch scan task.

    Args:
        session_id: Session ID from batch_scan endpoint

    Returns:
        Status and results (if completed)
    """
    try:
        task_manager = get_task_manager()
        status = task_manager.get_status(session_id)

        if status is None:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

        response = {
            "success": True,
            "session_id": session_id,
            "status": status["status"],
            "started_at": status.get("started_at").isoformat() if status.get("started_at") else None,
            "completed_at": status.get("completed_at").isoformat() if status.get("completed_at") else None,
        }

        # Include result if completed
        if status["status"] == "completed" and status.get("result"):
            response["result"] = status["result"]

        # Include error if failed
        if status["status"] == "error" and status.get("error"):
            response["error"] = status["error"]

        # Include cancelled status
        if status["status"] == "cancelled":
            response["message"] = "Scan was cancelled by user"

        return response

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi lấy status: {str(e)}")


@router.post("/batch/scan/{session_id}/cancel")
async def cancel_batch_scan(session_id: str):
    """
    Cancel a running batch scan task.

    Args:
        session_id: Session ID from batch_scan endpoint

    Returns:
        Success message and cancelled status
    """
    try:
        task_manager = get_task_manager()

        # Check if task exists
        status = task_manager.get_status(session_id)
        if status is None:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

        # Check if task can be cancelled
        current_status = status.get("status")
        if current_status not in ["running"]:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot cancel task with status '{current_status}'. Only running tasks can be cancelled.",
            )

        # Cancel the task
        cancelled = task_manager.cancel_task(session_id)
        if not cancelled:
            raise HTTPException(status_code=400, detail=f"Failed to cancel task {session_id}")

        return {
            "success": True,
            "session_id": session_id,
            "status": "cancelled",
            "message": "Scan cancelled successfully. The task will stop processing after the current batch.",
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi cancel scan: {str(e)}")


@router.get("/batch/results/{filename:path}")
async def get_batch_results(filename: str):
    """
    Get saved batch scan results by filename.

    Args:
        filename: Name of the results JSON file

    Returns:
        Batch scan results from JSON file
    """
    try:
        from modules.gemini_chart_analyzer.core.utils.chart_paths import get_analysis_results_dir

        results_dir = get_analysis_results_dir()
        batch_scan_dir = Path(results_dir) / "batch_scan"

        # Security: prevent directory traversal
        # First check: reject filenames containing path traversal patterns
        # This check must come BEFORE JSON extension check to return appropriate error message
        if ".." in filename or filename.startswith("/") or (len(filename) > 1 and filename[1] == ":"):
            raise HTTPException(status_code=400, detail="Directory traversal detected")

        # Security: only allow JSON files (check after traversal patterns)
        if not filename.endswith(".json"):
            raise HTTPException(status_code=400, detail="Only JSON files are allowed")

        # Resolve the batch scan directory with specific error handling
        try:
            resolved_dir = batch_scan_dir.resolve(strict=True)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Batch scan directory not found")

        # Resolve the requested file path
        # Note: resolve(strict=True) only raises if a parent directory doesn't exist,
        # not if the final file doesn't exist. So we resolve without strict and check existence separately.
        results_file = (batch_scan_dir / filename).resolve()

        # Ensure the results_file is inside the batch_scan_dir
        # Use try/except with relative_to to check if file is within directory
        # This check must happen BEFORE checking if file exists
        try:
            relative_path = results_file.relative_to(resolved_dir)
        except ValueError:
            # ValueError: path is not relative to base (directory traversal detected)
            # Don't check file existence for traversal attempts
            raise HTTPException(status_code=400, detail="Directory traversal detected")

        if not results_file.exists():
            raise HTTPException(status_code=404, detail=f"Results file not found: {filename}")

        # Read and return JSON
        with open(results_file, "r", encoding="utf-8") as f:
            results = json.load(f)

        return {"success": True, "filename": filename, "results": results}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi đọc kết quả: {str(e)}")


@router.get("/batch/list")
async def list_batch_results(
    skip: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=200), metadata_only: bool = Query(False)
):
    """
    List all available batch scan results.

    Args:
        skip: Number of results to skip (for pagination)
        limit: Maximum number of results to return (1-200)
        metadata_only: If True, skip reading summary from JSON files for better performance

    Returns:
        List of available results files with metadata
    """
    try:
        from modules.gemini_chart_analyzer.core.utils.chart_paths import get_analysis_results_dir

        results_dir = get_analysis_results_dir()
        batch_scan_dir = Path(results_dir) / "batch_scan"

        if not batch_scan_dir.exists():
            return {"success": True, "count": 0, "results": []}

        # Collect file objects first, and sort by modified date (newest first)
        json_files_meta = []
        for json_file in batch_scan_dir.glob("*.json"):
            try:
                stat = json_file.stat()
                json_files_meta.append({"obj": json_file, "size": stat.st_size, "modified": stat.st_mtime})
            except Exception:
                pass

        json_files_meta.sort(key=lambda x: x["modified"], reverse=True)

        paged_files = json_files_meta[skip : skip + limit]

        results_files = []
        for entry in paged_files:
            json_file = entry["obj"]
            try:
                stat = json_file.stat()
            except Exception:
                # Skip files that can't be stat'd
                continue

            summary = {}
            if not metadata_only:
                # Try to read summary, skip file if it's unreadable
                try:
                    with open(json_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        summary = data.get("summary", {})
                except Exception:
                    # Skip unreadable files when metadata_only is False
                    continue

            results_files.append(
                {
                    "filename": json_file.name,
                    "size": stat.st_size,
                    "modified": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
                    .isoformat()
                    .replace("+00:00", "Z"),
                    "summary": summary,
                    "url": f"/api/batch/results/{json_file.name}",
                }
            )

        # Sort by modified time (newest first)
        return {"success": True, "count": len(results_files), "results": results_files}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi liệt kê kết quả: {str(e)}")
