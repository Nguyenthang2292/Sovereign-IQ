
from pathlib import Path
import os
import sys

"""
Main entry point for Gemini Chart Analyzer Web Server.

FastAPI server for serving REST API endpoints and Vue.js frontend.
Run from project root: python main_gemini_chart_web_server.py
"""


# Add project root to path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# Import API routers from web.api
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
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API routes
app.include_router(chart_analyzer.router, prefix="/api", tags=["Chart Analyzer"])
app.include_router(batch_scanner.router, prefix="/api", tags=["Batch Scanner"])
app.include_router(logs.router, prefix="/api", tags=["Logs"])

# Get paths relative to project root
WEB_DIR = project_root / "web"
STATIC_DIR = WEB_DIR / "static"
VUE_DIST_DIR = STATIC_DIR / "vue" / "dist"

MODULE_ROOT = project_root / "modules" / "gemini_chart_analyzer"
CHARTS_DIR = MODULE_ROOT / "charts"
RESULTS_DIR = MODULE_ROOT / "analysis_results"

# Ensure directories exist
CHARTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Mount static file directories (existence already guaranteed above)
app.mount("/static/charts", StaticFiles(directory=str(CHARTS_DIR)), name="charts")
app.mount("/static/results", StaticFiles(directory=str(RESULTS_DIR)), name="results")

# Serve Vue.js app
if VUE_DIST_DIR.exists():
    # Mount Vue dist directory
    app.mount("/static/vue", StaticFiles(directory=str(VUE_DIST_DIR)), name="vue")


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
    """Health check endpoint."""
    return {
        "status": "healthy",
        "charts_dir": str(CHARTS_DIR),
        "results_dir": str(RESULTS_DIR),
        "vue_dist_exists": VUE_DIST_DIR.exists(),
    }


# Serve Vue.js app for all other routes (SPA routing)
# This must be defined AFTER explicit routes like "/" and "/health"
if VUE_DIST_DIR.exists():

    @app.get("/{full_path:path}")
    async def serve_vue_app(full_path: str):
        """
        Serve Vue.js app for all routes.
        This allows Vue Router to handle client-side routing.
        """
        # Check if it's an API route
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
                status_code=503, detail="Vue app not built. Please run 'npm run build' in web/static/vue/"
            )


if __name__ == "__main__":
    import uvicorn

    # Parse PORT with error handling
    try:
        port = int(os.getenv("PORT", "8000"))
    except ValueError:
        print(f"Invalid PORT value: {os.getenv('PORT')}. Using default 8000.")
        port = 8000

    uvicorn.run(
        "main_gemini_chart_web_server:app",
        host=os.getenv("HOST", "127.0.0.1"),
        port=port,
        reload=os.getenv("ENV", "development") != "production",
    )
