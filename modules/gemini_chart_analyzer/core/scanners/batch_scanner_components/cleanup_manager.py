"""
Cleanup Manager Component

Handles cleanup of old batch scan results and chart files.
"""

import glob
import os

from modules.common.ui.logging import log_info, log_warn
from modules.gemini_chart_analyzer.core.utils.chart_paths import get_analysis_results_dir, get_charts_dir


class CleanupManager:
    """
    Manages cleanup operations for old batch scan files.

    IMPORTANT: Cleanup operations are destructive and delete ALL previous batch scan
    results and charts without applying any retention policy or age threshold.

    WARNING:
        - Historical scan results and charts will be permanently deleted
        - If preservation is needed, backup files before running cleanup
        - This happens automatically at scan start unless disabled with skip_cleanup=True
    """

    def cleanup_old_results(self):
        """
        Cleanup old batch scan results (JSON and HTML files).

        Deletes:
            - batch_scan/batch_scan_*.json
            - batch_scan/batch_scan_*.html

        Errors are logged as warnings but do not stop the cleanup process.
        """
        try:
            results_base_dir = get_analysis_results_dir()
            batch_scan_dir = os.path.join(results_base_dir, "batch_scan")

            if os.path.exists(batch_scan_dir):
                # Clean up JSON files
                self._cleanup_files(batch_scan_dir, "batch_scan_*.json", "batch scan result")

                # Clean up HTML files
                self._cleanup_files(batch_scan_dir, "batch_scan_*.html", "batch scan HTML report")

        except Exception as e:
            log_warn(f"Error cleaning up batch scan results: {e}")

    def cleanup_old_charts(self):
        """
        Cleanup old batch chart files (PNG and HTML).

        Deletes:
            - charts/batch/batch_chart_*.png
            - charts/batch/batch_chart_*.html

        Errors are logged as warnings but do not stop the cleanup process.
        """
        try:
            charts_dir = get_charts_dir()
            batch_charts_dir = os.path.join(str(charts_dir), "batch")

            if os.path.exists(batch_charts_dir):
                # Clean up PNG files
                self._cleanup_files(batch_charts_dir, "batch_chart_*.png", "batch chart")

                # Clean up HTML files
                self._cleanup_files(batch_charts_dir, "batch_chart_*.html", "batch chart HTML")

        except Exception as e:
            log_warn(f"Error cleaning up batch charts: {e}")

    def _cleanup_files(self, directory: str, pattern: str, file_type: str):
        """
        Helper method to cleanup files matching a pattern.

        Args:
            directory: Directory to search in
            pattern: Glob pattern to match files
            file_type: Human-readable description of file type (for logging)
        """
        file_paths = glob.glob(os.path.join(directory, pattern))

        deleted_count = 0
        for file_path in file_paths:
            try:
                os.remove(file_path)
                deleted_count += 1
            except OSError as e:
                log_warn(f"Could not delete file {os.path.basename(file_path)}: {e}")
            except Exception as e:
                log_warn(f"Unexpected error deleting {os.path.basename(file_path)}: {e}")

        if deleted_count > 0:
            log_info(f"Deleted {deleted_count} old {file_type} file(s)")
