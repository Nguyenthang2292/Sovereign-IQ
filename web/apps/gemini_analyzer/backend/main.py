import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .config import (
    APP_TITLE,
    APP_DESCRIPTION,
    APP_VERSION,
    BACKEND_PORT,
    CORS_ORIGINS,
    API_PREFIX,
    CHARTS_DIR,
    RESULTS_DIR,
    VUE_DIST_DIR,
    STATIC_CHARTS_ROUTE,
    STATIC_RESULTS_ROUTE,
    STATIC_VUE_ROUTE,
)
from .api import batch_scanner, chart_analyzer, logs

"""
FastAPI server for Gemini Chart Analyzer Web Interface.

Serves REST API endpoints and static files for Vue.js frontend.
"""


# Initialize FastAPI app
app = FastAPI(
    title=APP_TITLE,
    description=APP_DESCRIPTION,
    version=APP_VERSION,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API routes
app.include_router(chart_analyzer.router, prefix=API_PREFIX, tags=["Chart Analyzer"])
app.include_router(batch_scanner.router, prefix=API_PREFIX, tags=["Batch Scanner"])
app.include_router(logs.router, prefix=API_PREFIX, tags=["Logs"])

# Mount static file directories (existence guaranteed by mkdir)
app.mount(STATIC_CHARTS_ROUTE, StaticFiles(directory=str(CHARTS_DIR)), name="charts")
app.mount(STATIC_RESULTS_ROUTE, StaticFiles(directory=str(RESULTS_DIR)), name="results")

# Mount shared assets
project_root = Path(__file__).resolve().parent.parent.parent.parent
shared_dir = project_root / "web" / "shared"
if shared_dir.exists():
    app.mount("/shared", StaticFiles(directory=str(shared_dir)), name="shared")


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
            "note": f"Vue app not built. Please run 'npm run build' in apps/gemini_analyzer/frontend/",
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
    app.mount(STATIC_VUE_ROUTE, StaticFiles(directory=str(VUE_DIST_DIR)), name="vue")

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
                status_code=404,
                detail=f"Vue app not built. Please run 'npm run build' in apps/gemini_analyzer/frontend/",
            )


if __name__ == "__main__":
    import uvicorn

    reload = os.getenv("UVICORN_RELOAD", "true").lower() in ("1", "true", "yes")
    uvicorn.run(app, host="0.0.0.0", port=BACKEND_PORT, reload=reload)
