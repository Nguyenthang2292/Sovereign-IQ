"""
Cache Metrics API Endpoint

Provides HTTP endpoint for cache metrics that can be integrated with
web dashboards or monitoring tools.

Usage:
    # Standalone server
    python cache_metrics_api.py --port 8080

    # Import and use in FastAPI app
    from cache_metrics_api import create_cache_metrics_router
    app.include_router(create_cache_metrics_router())
"""

import time
from typing import Any, Dict, List, Optional

from modules.adaptive_trend_LTS_mini.utils.cache_manager import CacheManager, get_cache_manager


class CacheMetricsService:
    """
    Service for providing cache metrics via API.

    Thread-safe implementation for use in web servers.
    """

    def __init__(self, cache: Optional[CacheManager] = None):
        """Initialize metrics service.

        Args:
            cache: Optional cache instance (for testing). If None, uses get_cache_manager().
        """
        self.cache = cache if cache is not None else get_cache_manager()
        self.snapshots: List[Dict[str, Any]] = []
        self.max_snapshots = 1000  # Keep last 1000 snapshots

    def get_current_metrics(self) -> Dict[str, Any]:
        """
        Get current cache metrics.

        Returns:
            Dictionary with current metrics
        """
        stats = self.cache.get_detailed_metrics()
        return {
            "timestamp": time.time(),
            "metrics": stats,
        }

    def get_health_status(self) -> Dict[str, Any]:
        """
        Get cache health status.

        Returns:
            Dictionary with health status
        """
        stats = self.cache.get_stats()

        # Determine health based on metrics
        health = "healthy"
        warnings = []

        if stats["total_requests"] > 0:
            if stats["hit_rate_percent"] < 30:
                health = "degraded"
                warnings.append("Low hit rate (< 30%)")

            if stats["hit_rate_percent"] < 10:
                health = "critical"
                warnings.append("Critical hit rate (< 10%)")

            if stats["recent_hit_rate_percent"] < stats["hit_rate_percent"] - 20:
                health = "degraded"
                warnings.append("Hit rate declining rapidly")

        # Check cache utilization
        l1_utilization = (stats["entries_l1"] / self.cache.max_entries_l1 * 100) if self.cache.max_entries_l1 > 0 else 0
        l2_utilization = (stats["entries_l2"] / self.cache.max_entries_l2 * 100) if self.cache.max_entries_l2 > 0 else 0

        if l1_utilization > 95 or l2_utilization > 95:
            if health == "healthy":
                health = "degraded"
            warnings.append("Cache near capacity")

        return {
            "status": health,
            "warnings": warnings,
            "timestamp": time.time(),
        }

    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get performance summary.

        Returns:
            Dictionary with performance metrics
        """
        stats = self.cache.get_stats()

        return {
            "hit_rate": {
                "overall": stats["hit_rate_percent"],
                "l1": stats["hit_rate_l1_percent"],
                "l2": stats["hit_rate_l2_percent"],
                "recent": stats["recent_hit_rate_percent"],
            },
            "requests": {
                "total": stats["total_requests"],
                "hits": stats["hits"],
                "misses": stats["misses"],
            },
            "cache_size": {
                "l1_entries": stats["entries_l1"],
                "l1_max": self.cache.max_entries_l1,
                "l2_entries": stats["entries_l2"],
                "l2_max": self.cache.max_entries_l2,
                "l2_size_mb": stats["size_l2_mb"],
                "l2_max_mb": self.cache.max_size_bytes_l2 / (1024 * 1024),
            },
            "operations": {
                "evictions": stats["evictions"],
                "promotions": stats["promotions"],
            },
            "timestamp": time.time(),
        }

    def record_snapshot(self) -> Dict[str, Any]:
        """
        Record a metrics snapshot for historical tracking.

        Returns:
            The recorded snapshot
        """
        snapshot = self.get_current_metrics()
        self.snapshots.append(snapshot)

        # Keep only recent snapshots
        if len(self.snapshots) > self.max_snapshots:
            self.snapshots.pop(0)

        return snapshot

    def get_snapshots(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get historical snapshots.

        Args:
            limit: Maximum number of snapshots to return (most recent)

        Returns:
            List of snapshots
        """
        if limit:
            return self.snapshots[-limit:]
        return self.snapshots


# Global service instance
_metrics_service: Optional[CacheMetricsService] = None


def get_metrics_service() -> CacheMetricsService:
    """Get global metrics service instance."""
    global _metrics_service
    if _metrics_service is None:
        _metrics_service = CacheMetricsService()
    return _metrics_service


# FastAPI integration
def create_cache_metrics_router():
    """
    Create FastAPI router for cache metrics endpoints.

    Returns:
        APIRouter configured with cache metrics endpoints
    """
    try:
        from fastapi import APIRouter, HTTPException
    except ImportError:
        raise ImportError("FastAPI is required for API endpoints. Install with: pip install fastapi")

    router = APIRouter(prefix="/cache", tags=["cache"])
    service = get_metrics_service()

    @router.get("/metrics")
    async def get_metrics():
        """Get current cache metrics."""
        return service.get_current_metrics()

    @router.get("/health")
    async def get_health():
        """Get cache health status."""
        return service.get_health_status()

    @router.get("/summary")
    async def get_summary():
        """Get performance summary."""
        return service.get_performance_summary()

    @router.get("/snapshots")
    async def get_snapshots(limit: Optional[int] = 100):
        """Get historical snapshots."""
        return {
            "snapshots": service.get_snapshots(limit=limit),
            "count": len(service.snapshots),
        }

    @router.post("/snapshots")
    async def record_snapshot():
        """Record a new metrics snapshot."""
        return service.record_snapshot()

    return router


def main():
    """Main entry point for standalone API server."""
    import argparse

    try:
        import uvicorn
        from fastapi import FastAPI
    except ImportError:
        print("FastAPI and uvicorn are required for the API server.")
        print("Install with: pip install fastapi uvicorn")
        return

    parser = argparse.ArgumentParser(description="Cache Metrics API Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind (default: 8080)")
    args = parser.parse_args()

    # Create FastAPI app
    app = FastAPI(
        title="Adaptive Trend Cache Metrics API",
        description="Real-time cache performance metrics and monitoring",
        version="1.0.0",
    )

    # Add cache metrics router
    app.include_router(create_cache_metrics_router())

    # Root endpoint
    @app.get("/")
    async def root():
        return {
            "service": "Adaptive Trend Cache Metrics API",
            "version": "1.0.0",
            "endpoints": {
                "metrics": "/cache/metrics",
                "health": "/cache/health",
                "summary": "/cache/summary",
                "snapshots": "/cache/snapshots",
                "docs": "/docs",
            },
        }

    print(f"\n🚀 Starting Cache Metrics API Server")
    print(f"   Host: {args.host}")
    print(f"   Port: {args.port}")
    print(f"   Docs: http://{args.host}:{args.port}/docs")
    print(f"   Metrics: http://{args.host}:{args.port}/cache/metrics")
    print()

    # Run server
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
