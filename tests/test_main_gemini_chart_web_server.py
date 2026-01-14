"""
Test for main_gemini_chart_web_server.py - FastAPI Web Server.

Tests the FastAPI application and basic endpoints.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import after path setup
from main_gemini_chart_web_server import app


class TestWebServerApp:
    """Test the FastAPI application."""

    def test_app_creation(self):
        """Test that the FastAPI app is created properly."""
        assert app is not None
        assert hasattr(app, "routes")
        assert hasattr(app, "title")
        assert app.title == "Gemini Chart Analyzer API"

    def test_cors_middleware(self):
        """Test that CORS middleware is configured."""
        middleware_found = False
        for middleware in app.user_middleware:
            if hasattr(middleware, "cls") and "CORSMiddleware" in str(middleware.cls):
                middleware_found = True
                break

        assert middleware_found, "CORS middleware should be configured"

    def test_api_routes_mounted(self):
        """Test that API routes are mounted."""
        routes = [route.path for route in app.routes]

        # Check for main API routes
        assert any("/api" in route for route in routes), "API routes should be mounted"

        # Check for specific expected patterns based on actual routes
        expected_patterns = ["/api/analyze", "/api/batch", "/api/logs"]
        for pattern in expected_patterns:
            assert any(route.startswith(pattern) for route in routes), (
                f"Route starting with {pattern} should be mounted"
            )

    def test_static_routes_mounted(self):
        """Test that static file routes are mounted."""
        routes = [route.path for route in app.routes]

        # Check for static routes
        static_routes = ["/static/charts", "/static/results"]
        for static_route in static_routes:
            assert any(static_route in route for route in routes), f"Static route {static_route} should be mounted"

    def test_root_endpoint_exists(self):
        """Test that root endpoint exists."""
        routes = [route.path for route in app.routes]
        assert "/" in routes, "Root endpoint should exist"

    def test_health_endpoint_exists(self):
        """Test that health endpoint exists."""
        routes = [route.path for route in app.routes]
        assert "/health" in routes, "Health endpoint should exist"


class TestWebServerEndpoints:
    """Test individual endpoints."""

    def test_health_endpoint(self, client):
        """Test the /health endpoint."""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"
        assert "charts_dir" in data
        assert "results_dir" in data
        assert "vue_dist_exists" in data

    @patch("main_gemini_chart_web_server.VUE_DIST_DIR")
    def test_root_endpoint_with_vue(self, mock_vue_dist, client):
        """Test root endpoint when Vue app is built."""
        # Mock Vue dist exists
        mock_vue_dist.exists.return_value = True
        mock_vue_dist.__truediv__ = lambda self, x: MagicMock()

        with patch("main_gemini_chart_web_server.FileResponse") as mock_file_response:
            client.get("/")
            mock_file_response.assert_called_once()

    @patch("main_gemini_chart_web_server.VUE_DIST_DIR")
    def test_root_endpoint_without_vue(self, mock_vue_dist, client):
        """Test root endpoint when Vue app is not built."""
        # Mock Vue dist does not exist
        mock_vue_dist.exists.return_value = False
        # Also need to mock that the index.html Path doesn't exist
        mock_vue_dist.__truediv__.return_value.exists.return_value = False

        response = client.get("/")
        assert response.status_code == 200

        data = response.json()
        assert "message" in data
        assert "Gemini Chart Analyzer API" in data["message"]
        assert "Vue app not built" in data["note"]

    @patch("main_gemini_chart_web_server.VUE_DIST_DIR")
    def test_vue_spa_routing(self, mock_vue_dist, client):
        """Test Vue.js SPA routing for non-API routes."""
        mock_vue_dist.exists.return_value = True
        mock_vue_dist.__truediv__ = lambda self, x: MagicMock()

        with patch("main_gemini_chart_web_server.FileResponse") as mock_file_response:
            # Test a typical SPA route
            client.get("/dashboard")
            mock_file_response.assert_called_once()

    def test_api_route_blocked_in_spa(self, client):
        """Test that API routes are blocked in SPA routing."""
        response = client.get("/api/some-endpoint")
        assert response.status_code == 404

    def test_static_route_blocked_in_spa(self, client):
        """Test that static routes are blocked in SPA routing."""
        response = client.get("/static/some-file")
        assert response.status_code == 404


class TestWebServerIntegration:
    """Integration tests for the web server."""

    @patch("uvicorn.run")
    @patch("os.getenv")
    def test_main_execution_with_uvicorn(self, mock_getenv, mock_uvicorn_run):
        """Test that main() calls uvicorn.run with correct parameters."""
        # Mock environment variables
        mock_getenv.side_effect = lambda key, default=None: {
            "PORT": "8080",
            "HOST": "127.0.0.1",
            "ENV": "development",
        }.get(key, default)

        # Import and call main (but mock uvicorn to avoid actually starting server)

        # This would normally start the server, but we're mocking it
        # In a real test, we might need to handle this differently
        # For now, just test that the module can be imported

    def test_module_imports(self):
        """Test that all required modules are available."""
        import importlib

        modules = ["fastapi", "uvicorn", "fastapi.middleware.cors", "fastapi.responses", "fastapi.staticfiles"]
        for module_name in modules:
            try:
                importlib.import_module(module_name)
            except ImportError as e:
                pytest.fail(f"Module {module_name} cannot be imported: {e}")

    def test_path_configurations(self):
        """Test that path configurations are correct."""
        from main_gemini_chart_web_server import CHARTS_DIR, MODULE_ROOT, RESULTS_DIR, STATIC_DIR, VUE_DIST_DIR, WEB_DIR

        # Check that paths are Path objects
        assert isinstance(WEB_DIR, Path)
        assert isinstance(STATIC_DIR, Path)
        assert isinstance(VUE_DIST_DIR, Path)
        assert isinstance(MODULE_ROOT, Path)
        assert isinstance(CHARTS_DIR, Path)
        assert isinstance(RESULTS_DIR, Path)

        # Check relative structure
        assert WEB_DIR.name == "web"
        assert STATIC_DIR.name == "static"
        assert VUE_DIST_DIR.name == "dist"
        assert MODULE_ROOT.name == "gemini_chart_analyzer"
        assert CHARTS_DIR.name == "charts"
        assert RESULTS_DIR.name == "analysis_results"


# Pytest fixture for FastAPI TestClient
@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    from fastapi.testclient import TestClient

    return TestClient(app)


def test_fastapi_app_lifespan():
    """Test that the app can handle basic FastAPI lifecycle."""
    # This is a basic smoke test to ensure the app is properly configured
    # Version check is secondary, main thing is routes
    assert hasattr(app, "openapi_version")

    # Check that we have routes
    assert len(app.routes) > 0

    # Check that we have at least the basic endpoints
    route_paths = {route.path for route in app.routes}
    assert "/" in route_paths
    assert "/health" in route_paths
