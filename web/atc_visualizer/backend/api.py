"""
ATC Visualizer API - FastAPI REST API for ATC chart visualization.
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import sys
import os
from pathlib import Path

current_file_path = Path(__file__).resolve()
backend_dir = current_file_path.parent

project_root = backend_dir.parent.parent.parent
project_root_absolute = project_root.resolve()

sys.path.insert(0, str(project_root_absolute))

from atc_service import ATCService
from modules.adaptive_trend.utils.config import ATCConfig

app = FastAPI(
    title="ATC Visualizer API", description="REST API for Adaptive Trend Classification visualization", version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
        "frontend": "http://localhost:5173",
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
    symbol: str = Query(default="BTC/USDT", description="Trading symbol (e.g., BTC/USDT)"),
    timeframe: str = Query(default="15m", description="Timeframe (e.g., 15m, 1h, 1d)"),
    limit: int = Query(default=1500, ge=100, le=5000, description="Number of candles to fetch"),
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
    symbol: str = Query(default="BTC/USDT", description="Trading symbol"),
    timeframe: str = Query(default="15m", description="Timeframe"),
    limit: int = Query(default=1500, ge=100, le=5000, description="Number of candles"),
    ema_len: int = Query(default=28, ge=1, description="EMA length"),
    hma_len: int = Query(default=28, ge=1, description="HMA length"),
    wma_len: int = Query(default=28, ge=1, description="WMA length"),
    dema_len: int = Query(default=28, ge=1, description="DEMA length"),
    lsma_len: int = Query(default=28, ge=1, description="LSMA length"),
    kama_len: int = Query(default=28, ge=1, description="KAMA length"),
    robustness: str = Query(default="Medium", description="Robustness level (Narrow, Medium, Wide)"),
    lambda_param: float = Query(default=0.02, ge=0.0, le=1.0, description="Lambda parameter"),
    decay: float = Query(default=0.03, ge=0.0, le=1.0, description="Decay parameter"),
    cutout: int = Query(default=0, ge=0, description="Cutout parameter"),
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
    symbol: str = Query(default="BTC/USDT", description="Trading symbol"),
    timeframe: str = Query(default="15m", description="Timeframe"),
    limit: int = Query(default=1500, ge=100, le=5000, description="Number of candles"),
    ema_len: int = Query(default=28, ge=1, description="EMA length"),
    hma_len: int = Query(default=28, ge=1, description="HMA length"),
    wma_len: int = Query(default=28, ge=1, description="WMA length"),
    dema_len: int = Query(default=28, ge=1, description="DEMA length"),
    lsma_len: int = Query(default=28, ge=1, description="LSMA length"),
    kama_len: int = Query(default=28, ge=1, description="KAMA length"),
    robustness: str = Query(default="Medium", description="Robustness level"),
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
        "timeframes": [
            {"value": "1m", "label": "1 Minute"},
            {"value": "5m", "label": "5 Minutes"},
            {"value": "15m", "label": "15 Minutes"},
            {"value": "30m", "label": "30 Minutes"},
            {"value": "1h", "label": "1 Hour"},
            {"value": "4h", "label": "4 Hours"},
            {"value": "1d", "label": "1 Day"},
        ],
    }


if __name__ == "__main__":
    import uvicorn

    print("Starting ATC Visualizer API on http://localhost:5000")
    print("API Documentation available at: http://localhost:5000/docs")
    uvicorn.run(app, host="0.0.0.0", port=5000)
