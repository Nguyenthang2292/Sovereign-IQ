"""
Cache Monitoring Utilities for Adaptive Trend LTS Mini

Provides tools for monitoring cache performance, viewing statistics,
and generating performance reports.

Usage:
    # As a standalone script
    python modules/adaptive_trend_LTS_mini/utils/cache_monitor.py

    # In code
    from modules.adaptive_trend_LTS_mini.utils.cache_monitor import CacheMonitor
    monitor = CacheMonitor()
    monitor.display_dashboard()
"""

import json
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from modules.adaptive_trend_LTS_mini.utils.cache_manager import CacheManager, get_cache_manager
from modules.common.ui.logging import log_info


class CacheMonitor:
    """
    Monitor and display cache performance metrics.

    Provides real-time monitoring, historical tracking, and performance analytics.
    """

    def __init__(self, history_size: int = 100, cache: Optional[CacheManager] = None):
        """
        Initialize cache monitor.

        Args:
            history_size: Number of historical snapshots to keep
            cache: Optional cache instance (for testing). If None, uses get_cache_manager().
        """
        self.cache = cache if cache is not None else get_cache_manager()
        self.history: List[Dict[str, Any]] = []
        self.history_size = history_size
        self.start_time = time.time()

    def take_snapshot(self) -> Dict[str, Any]:
        """
        Take a snapshot of current cache metrics.

        Returns:
            Dictionary with current metrics and timestamp
        """
        stats = self.cache.get_detailed_metrics()
        snapshot = {
            "timestamp": time.time(),
            "datetime": datetime.now().isoformat(),
            **stats,
        }

        # Add to history (keep only recent entries)
        self.history.append(snapshot)
        if len(self.history) > self.history_size:
            self.history.pop(0)

        return snapshot

    def get_trends(self) -> Dict[str, Any]:
        """
        Analyze historical trends in cache performance.

        Returns:
            Dictionary with trend analysis
        """
        if len(self.history) < 2:
            return {"error": "Insufficient history for trend analysis"}

        # Calculate trends
        hit_rates = [h["hit_rate_percent"] for h in self.history]
        recent_hit_rates = [h["recent_hit_rate_percent"] for h in self.history]
        eviction_counts = [h["evictions"] for h in self.history]

        trends = {
            "hit_rate": {
                "current": hit_rates[-1],
                "min": min(hit_rates),
                "max": max(hit_rates),
                "avg": sum(hit_rates) / len(hit_rates),
                "trend": "increasing" if hit_rates[-1] > hit_rates[0] else "decreasing",
            },
            "recent_hit_rate": {
                "current": recent_hit_rates[-1],
                "min": min(recent_hit_rates),
                "max": max(recent_hit_rates),
                "avg": sum(recent_hit_rates) / len(recent_hit_rates),
            },
            "evictions": {
                "total": eviction_counts[-1],
                "rate": (eviction_counts[-1] - eviction_counts[0]) / len(self.history) if len(self.history) > 1 else 0,
            },
            "samples": len(self.history),
            "duration_seconds": time.time() - self.start_time,
        }

        return trends

    def display_dashboard(self, show_trends: bool = True):
        """
        Display a text-based dashboard of cache metrics.

        Args:
            show_trends: Whether to include trend analysis
        """
        snapshot = self.take_snapshot()

        print("\n" + "=" * 80)
        print("CACHE PERFORMANCE DASHBOARD")
        print("=" * 80)
        print(f"Timestamp: {snapshot['datetime']}")
        print()

        # Cache Size
        print("CACHE SIZE:")
        print(f"  L1 Entries: {snapshot['entries_l1']:,} / {self.cache.max_entries_l1:,}")
        print(f"  L2 Entries: {snapshot['entries_l2']:,} / {self.cache.max_entries_l2:,}")
        print(f"  L2 Size: {snapshot['size_l2_mb']:.2f} MB / {self.cache.max_size_bytes_l2 / (1024*1024):.2f} MB")
        print()

        # Hit/Miss Statistics
        print("HIT/MISS STATISTICS:")
        print(f"  Total Requests: {snapshot['total_requests']:,}")
        print(f"  Total Hits: {snapshot['hits']:,} ({snapshot['hit_rate_percent']:.2f}%)")
        print(f"    - L1 Hits: {snapshot['hits_l1']:,} ({snapshot['hit_rate_l1_percent']:.2f}%)")
        print(f"    - L2 Hits: {snapshot['hits_l2']:,} ({snapshot['hit_rate_l2_percent']:.2f}%)")
        print(f"  Total Misses: {snapshot['misses']:,} ({snapshot['miss_rate_percent']:.2f}%)")
        print()

        # Recent Performance
        print("RECENT PERFORMANCE (Last 60s):")
        print(f"  Recent Hit Rate: {snapshot['recent_hit_rate_percent']:.2f}%")
        print(f"  Recent Hits: {snapshot['recent_hits']:,}")
        print(f"  Recent Misses: {snapshot['recent_misses']:,}")
        print()

        # Cache Operations
        print("CACHE OPERATIONS:")
        print(f"  Evictions: {snapshot['evictions']:,}")
        print(f"  Promotions (L2→L1): {snapshot['promotions']:,}")
        print()

        # Insights
        if snapshot.get("insights"):
            print("PERFORMANCE INSIGHTS:")
            for insight in snapshot["insights"]:
                print(f"  ⚠ {insight}")
            print()

        # Trends
        if show_trends and len(self.history) >= 2:
            trends = self.get_trends()
            print("TRENDS:")
            print(f"  Hit Rate: {trends['hit_rate']['current']:.2f}% (avg: {trends['hit_rate']['avg']:.2f}%, "
                  f"min: {trends['hit_rate']['min']:.2f}%, max: {trends['hit_rate']['max']:.2f}%)")
            print(f"  Trend: {trends['hit_rate']['trend'].upper()}")
            print(f"  Eviction Rate: {trends['evictions']['rate']:.2f} evictions/snapshot")
            print(f"  Monitoring Duration: {trends['duration_seconds']:.1f}s ({trends['samples']} samples)")
            print()

        print("=" * 80)

    def export_metrics(self, filepath: str):
        """
        Export current metrics to JSON file.

        Args:
            filepath: Path to export file
        """
        snapshot = self.take_snapshot()
        with open(filepath, "w") as f:
            json.dump(snapshot, f, indent=2)
        log_info(f"Cache metrics exported to {filepath}")

    def export_history(self, filepath: str):
        """
        Export historical metrics to JSON file.

        Args:
            filepath: Path to export file
        """
        with open(filepath, "w") as f:
            json.dump(self.history, f, indent=2)
        log_info(f"Cache history exported to {filepath}")

    def get_summary_report(self) -> str:
        """
        Generate a summary report of cache performance.

        Returns:
            Formatted string report
        """
        snapshot = self.take_snapshot()
        trends = self.get_trends() if len(self.history) >= 2 else None

        report = []
        report.append("CACHE PERFORMANCE SUMMARY")
        report.append("=" * 50)
        report.append(f"Hit Rate: {snapshot['hit_rate_percent']:.2f}%")
        report.append(f"Recent Hit Rate: {snapshot['recent_hit_rate_percent']:.2f}%")
        report.append(f"Total Requests: {snapshot['total_requests']:,}")
        report.append(f"Cache Entries: {snapshot['entries']:,} (L1: {snapshot['entries_l1']}, L2: {snapshot['entries_l2']})")
        report.append(f"Evictions: {snapshot['evictions']:,}")
        report.append(f"Promotions: {snapshot['promotions']:,}")

        if trends:
            report.append("")
            report.append("TREND: " + trends['hit_rate']['trend'].upper())

        if snapshot.get("insights"):
            report.append("")
            report.append("INSIGHTS:")
            for insight in snapshot["insights"]:
                report.append(f"  - {insight}")

        return "\n".join(report)


def monitor_continuously(interval_seconds: int = 10, duration_seconds: Optional[int] = None):
    """
    Continuously monitor cache performance and display updates.

    Args:
        interval_seconds: Interval between updates
        duration_seconds: Total monitoring duration (None for indefinite)
    """
    monitor = CacheMonitor()
    start_time = time.time()

    print("\nStarting continuous cache monitoring...")
    print(f"Update interval: {interval_seconds}s")
    if duration_seconds:
        print(f"Duration: {duration_seconds}s")
    print("Press Ctrl+C to stop\n")

    try:
        while True:
            monitor.display_dashboard(show_trends=True)

            # Check duration
            if duration_seconds and (time.time() - start_time) >= duration_seconds:
                break

            time.sleep(interval_seconds)

    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")

    # Final report
    print("\n" + monitor.get_summary_report())


def main():
    """Main entry point for standalone script."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Monitor Adaptive Trend LTS Mini cache performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Display current dashboard
  python cache_monitor.py

  # Continuous monitoring with 5s updates
  python cache_monitor.py --continuous --interval 5

  # Export metrics to file
  python cache_monitor.py --export metrics.json

  # Monitor for 60 seconds
  python cache_monitor.py --continuous --duration 60
        """,
    )

    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Continuously monitor and update display",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=10,
        help="Update interval in seconds (default: 10)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        help="Monitoring duration in seconds (default: indefinite)",
    )
    parser.add_argument(
        "--export",
        type=str,
        metavar="FILE",
        help="Export current metrics to JSON file",
    )
    parser.add_argument(
        "--export-history",
        type=str,
        metavar="FILE",
        help="Export historical metrics to JSON file",
    )

    args = parser.parse_args()

    monitor = CacheMonitor()

    if args.continuous:
        monitor_continuously(
            interval_seconds=args.interval,
            duration_seconds=args.duration,
        )
    elif args.export:
        monitor.export_metrics(args.export)
        print(f"Metrics exported to {args.export}")
    elif args.export_history:
        monitor.export_history(args.export_history)
        print(f"History exported to {args.export_history}")
    else:
        # Single snapshot
        monitor.display_dashboard(show_trends=False)
        print("\nTip: Use --continuous for real-time monitoring")


if __name__ == "__main__":
    main()
