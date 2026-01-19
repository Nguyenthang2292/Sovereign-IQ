"""
ATC Visualizer API - FastAPI REST API for ATC chart visualization.
"""

import sys
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

# Setup path for imports
current_file_path = Path(__file__).resolve()
backend_dir = current_file_path.parent
# Go from: web/apps/atc_visualizer/backend -> web -> project root (contains web, modules, etc.)
web_dir = backend_dir.parent.parent
project_root = web_dir.parent
sys.path.insert(0, str(project_root))

# Import directly from file path when running as script, or use relative imports as module
# Check if we can use relative imports (when run as module)
try:
    # Try relative imports first (when run as module)
    from .config import (
        APP_DESCRIPTION,
        APP_TITLE,
        APP_VERSION,
        BACKEND_PORT,
        CORS_ORIGINS,
        DEFAULT_CUTOUT,
        DEFAULT_DECAY,
        DEFAULT_LAMBDA,
        DEFAULT_LIMIT,
        DEFAULT_MA_LENGTH,
        DEFAULT_ROBUSTNESS,
        DEFAULT_SYMBOL,
        DEFAULT_TIMEFRAME,
        FRONTEND_DEV_PORT,
        MAX_LIMIT,
        MIN_LIMIT,
        TIMEFRAMES,
    )
    from .services.atc_service import ATCService
except ImportError:
    # Running as script - import from file paths directly
    import importlib.util

    config_spec = importlib.util.spec_from_file_location("config", backend_dir / "config.py")
    config_module = importlib.util.module_from_spec(config_spec)
    config_spec.loader.exec_module(config_module)

    atc_service_spec = importlib.util.spec_from_file_location(
        "atc_service", backend_dir / "services" / "atc_service.py"
    )
    atc_service_module = importlib.util.module_from_spec(atc_service_spec)
    atc_service_spec.loader.exec_module(atc_service_module)

    APP_TITLE = config_module.APP_TITLE
    APP_DESCRIPTION = config_module.APP_DESCRIPTION
    APP_VERSION = config_module.APP_VERSION
    BACKEND_PORT = config_module.BACKEND_PORT
    FRONTEND_DEV_PORT = config_module.FRONTEND_DEV_PORT
    CORS_ORIGINS = config_module.CORS_ORIGINS
    DEFAULT_SYMBOL = config_module.DEFAULT_SYMBOL
    DEFAULT_TIMEFRAME = config_module.DEFAULT_TIMEFRAME
    DEFAULT_LIMIT = config_module.DEFAULT_LIMIT
    MIN_LIMIT = config_module.MIN_LIMIT
    MAX_LIMIT = config_module.MAX_LIMIT
    DEFAULT_MA_LENGTH = config_module.DEFAULT_MA_LENGTH
    DEFAULT_ROBUSTNESS = config_module.DEFAULT_ROBUSTNESS
    DEFAULT_LAMBDA = config_module.DEFAULT_LAMBDA
    DEFAULT_DECAY = config_module.DEFAULT_DECAY
    DEFAULT_CUTOUT = config_module.DEFAULT_CUTOUT
    TIMEFRAMES = config_module.TIMEFRAMES
    ATCService = atc_service_module.ATCService
from fastapi.staticfiles import StaticFiles

from modules.adaptive_trend.utils.config import ATCConfig
from web.shared.middleware.cors import setup_cors

# Initialize FastAPI app
app = FastAPI(
    title=APP_TITLE,
    description=APP_DESCRIPTION,
    version=APP_VERSION,
)

# Setup CORS
setup_cors(app, allowed_origins=CORS_ORIGINS)

# Mount shared assets
shared_dir = project_root / "web" / "shared"
if shared_dir.exists():
    app.mount("/shared", StaticFiles(directory=str(shared_dir)), name="shared")


# Initialize ATC service
atc_service = ATCService()


@app.get("/")
async def root():
    """
    Root endpoint - redirects to health check.
    """
    return {
        "message": "ATC Visualizer API",
        "health": "/api/health",
        "docs": "/docs",
        "frontend": "http://localhost:5174",
    }


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str
    service: str


class SymbolsResponse(BaseModel):
    """Symbols list response model."""

    success: bool
    symbols: List[str]


class TimeframeResponse(BaseModel):
    """Timeframe item model."""

    value: str
    label: str


class TimeframesResponse(BaseModel):
    """Timeframes list response model."""

    success: bool
    timeframes: List[TimeframeResponse]


class ErrorResponse(BaseModel):
    """Error response model."""

    success: bool = False
    error: str


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.

    Returns the status of the API service.
    """
    return {"status": "ok", "service": "ATC Visualizer API"}


@app.get("/api/symbols", response_model=SymbolsResponse)
async def get_symbols():
    """
    Get list of available trading symbols from Binance.

    Returns a list of USDT-M futures symbols sorted by volume.
    """
    try:
        symbols = atc_service.list_available_symbols()
        return {"success": True, "symbols": symbols}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ohlcv")
async def get_ohlcv(
    symbol: str = Query(default=DEFAULT_SYMBOL, description="Trading symbol (e.g., BTC/USDT)"),
    timeframe: str = Query(default=DEFAULT_TIMEFRAME, description="Timeframe (e.g., 15m, 1h, 1d)"),
    limit: int = Query(default=DEFAULT_LIMIT, ge=MIN_LIMIT, le=MAX_LIMIT, description="Number of candles to fetch"),
):
    """
    Fetch OHLCV data for a symbol.

    Returns candlestick data formatted for chart visualization.
    """
    try:
        data = atc_service.get_ohlcv_data(symbol, timeframe, limit)

        if data is None:
            raise HTTPException(status_code=404, detail="Failed to fetch OHLCV data")

        return {"success": True, "data": data}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/atc-signals")
async def get_atc_signals(
    symbol: str = Query(default=DEFAULT_SYMBOL, description="Trading symbol"),
    timeframe: str = Query(default=DEFAULT_TIMEFRAME, description="Timeframe"),
    limit: int = Query(default=DEFAULT_LIMIT, ge=MIN_LIMIT, le=MAX_LIMIT, description="Number of candles"),
    ema_len: int = Query(default=DEFAULT_MA_LENGTH, ge=1, description="EMA length"),
    hma_len: int = Query(default=DEFAULT_MA_LENGTH, ge=1, description="HMA length"),
    wma_len: int = Query(default=DEFAULT_MA_LENGTH, ge=1, description="WMA length"),
    dema_len: int = Query(default=DEFAULT_MA_LENGTH, ge=1, description="DEMA length"),
    lsma_len: int = Query(default=DEFAULT_MA_LENGTH, ge=1, description="LSMA length"),
    kama_len: int = Query(default=DEFAULT_MA_LENGTH, ge=1, description="KAMA length"),
    robustness: str = Query(default=DEFAULT_ROBUSTNESS, description="Robustness level (Narrow, Medium, Wide)"),
    lambda_param: float = Query(default=DEFAULT_LAMBDA, ge=0.0, le=1.0, description="Lambda parameter"),
    decay: float = Query(default=DEFAULT_DECAY, ge=0.0, le=1.0, description="Decay parameter"),
    cutout: int = Query(default=DEFAULT_CUTOUT, ge=0, description="Cutout parameter"),
):
    """
    Compute ATC signals for a symbol.

    Returns comprehensive ATC analysis including:
    - OHLCV data
    - All MA signals (EMA_Signal, HMA_Signal, etc.)
    - Average_Signal (combined signal)
    """
    try:
        config = ATCConfig(
            timeframe=timeframe,
            limit=limit,
            ema_len=ema_len,
            hma_len=hma_len,
            wma_len=wma_len,
            dema_len=dema_len,
            lsma_len=lsma_len,
            kama_len=kama_len,
            robustness=robustness,
            lambda_param=lambda_param,
            decay=decay,
            cutout=cutout,
        )

        data = atc_service.compute_atc_signals(symbol, timeframe, config)

        if data is None:
            raise HTTPException(status_code=404, detail="Failed to compute ATC signals")

        return {"success": True, "data": data}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/moving-averages")
async def get_moving_averages(
    symbol: str = Query(default=DEFAULT_SYMBOL, description="Trading symbol"),
    timeframe: str = Query(default=DEFAULT_TIMEFRAME, description="Timeframe"),
    limit: int = Query(default=DEFAULT_LIMIT, ge=MIN_LIMIT, le=MAX_LIMIT, description="Number of candles"),
    ema_len: int = Query(default=DEFAULT_MA_LENGTH, ge=1, description="EMA length"),
    hma_len: int = Query(default=DEFAULT_MA_LENGTH, ge=1, description="HMA length"),
    wma_len: int = Query(default=DEFAULT_MA_LENGTH, ge=1, description="WMA length"),
    dema_len: int = Query(default=DEFAULT_MA_LENGTH, ge=1, description="DEMA length"),
    lsma_len: int = Query(default=DEFAULT_MA_LENGTH, ge=1, description="LSMA length"),
    kama_len: int = Query(default=DEFAULT_MA_LENGTH, ge=1, description="KAMA length"),
    robustness: str = Query(default=DEFAULT_ROBUSTNESS, description="Robustness level"),
):
    """
    Get all Moving Averages for a symbol.

    Returns 9 MAs for each MA type:
    - MA, MA1, MA2, MA3, MA4 (positive offsets)
    - MA_1, MA_2, MA_3, MA_4 (negative offsets)
    """
    try:
        config = ATCConfig(
            timeframe=timeframe,
            limit=limit,
            ema_len=ema_len,
            hma_len=hma_len,
            wma_len=wma_len,
            dema_len=dema_len,
            lsma_len=lsma_len,
            kama_len=kama_len,
            robustness=robustness,
        )

        data = atc_service.get_moving_averages(symbol, timeframe, config)

        if data is None:
            raise HTTPException(status_code=404, detail="Failed to compute moving averages")

        return {"success": True, "data": data}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/timeframes", response_model=TimeframesResponse)
async def get_timeframes():
    """
    Get available timeframes.

    Returns list of supported timeframes for chart visualization.
    """
    return {
        "success": True,
        "timeframes": TIMEFRAMES,
    }


if __name__ == "__main__":
    import uvicorn

    print(f"Starting ATC Visualizer API on http://localhost:{BACKEND_PORT}")
    print(f"API Documentation available at: http://localhost:{BACKEND_PORT}/docs")
    print(f"Frontend should run on: http://localhost:{FRONTEND_DEV_PORT}")
    uvicorn.run(app, host="0.0.0.0", port=BACKEND_PORT)
