
from pathlib import Path
import os

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from web.api import batch_scanner, chart_analyzer, logs

from web.api import batch_scanner, chart_analyzer, logs

"""
FastAPI server for Gemini Chart Analyzer Web Interface.

Serves REST API endpoints and static files for Vue.js frontend.
"""



# Import API routers
from web.api import batch_scanner, chart_analyzer, logs

# Initialize FastAPI app
app = FastAPI(
    title="Gemini Chart Analyzer API",
    description="REST API for Gemini Chart Analyzer and Batch Scanner",
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:8000").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Mount API routes
app.include_router(chart_analyzer.router, prefix="/api", tags=["Chart Analyzer"])
app.include_router(batch_scanner.router, prefix="/api", tags=["Batch Scanner"])
app.include_router(logs.router, prefix="/api", tags=["Logs"])

# Get paths
BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
VUE_DIST_DIR = STATIC_DIR / "vue" / "dist"

# Get module paths for serving charts and results
MODULE_ROOT = BASE_DIR.parent / "modules" / "gemini_chart_analyzer"
CHARTS_DIR = MODULE_ROOT / "charts"
RESULTS_DIR = MODULE_ROOT / "analysis_results"

# Ensure directories exist
CHARTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Mount static file directories (existence guaranteed by mkdir)
app.mount("/static/charts", StaticFiles(directory=str(CHARTS_DIR)), name="charts")
app.mount("/static/results", StaticFiles(directory=str(RESULTS_DIR)), name="results")


@app.get("/")
async def root():
    """Root endpoint - serve Vue.js app."""
    index_path = VUE_DIST_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    else:
        return {
            "message": "Gemini Chart Analyzer API",
            "status": "running",
            "note": "Vue app not built. Please run 'npm run build' in web/static/vue/",
        }


@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    Returns minimal information in production to avoid path exposure.
    """

    is_production = os.getenv("ENV", "development").lower() == "production"
    base_response = {
        "status": "healthy",
        "charts_dir_exists": CHARTS_DIR.exists(),
        "results_dir_exists": RESULTS_DIR.exists(),
        "vue_dist_exists": VUE_DIST_DIR.exists(),
    }
    if not is_production:
        base_response["charts_dir"] = str(CHARTS_DIR)
        base_response["results_dir"] = str(RESULTS_DIR)
    return base_response


# Serve Vue.js app - catch-all route must be LAST
if VUE_DIST_DIR.exists():
    # Mount Vue dist directory
    app.mount("/static/vue", StaticFiles(directory=str(VUE_DIST_DIR)), name="vue")

    # Serve index.html for all routes (SPA routing)
    # This must be defined AFTER all specific routes like /health

    @app.get("/{full_path:path}")
    async def serve_vue_app(full_path: str, request: Request):
        """
        Serve Vue.js app for all routes.
        This allows Vue Router to handle client-side routing.
        """
        # Check if it's an API route - if so, return 404 (routing handled by API routers)
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="Not found")

        # Check if it's a static file request
        if full_path.startswith("static/"):
            raise HTTPException(status_code=404, detail="Not found")

        # Serve index.html for all other routes
        index_path = VUE_DIST_DIR / "index.html"
        if index_path.exists():
            return FileResponse(str(index_path))
        else:
            raise HTTPException(
                status_code=404, detail="Vue app not built. Please run 'npm run build' in web/static/vue/"
            )


if __name__ == "__main__":
    import uvicorn

    reload = os.getenv("UVICORN_RELOAD", "true").lower() in ("1", "true", "yes")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=reload)
