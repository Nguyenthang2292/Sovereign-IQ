# Thiết Kế Chi Tiết: Adaptive Close Time dựa trên Regime Change Detection

> **Ngày tạo:** 2026-03-09
> **Trạng thái:** Design Approved — Chờ Implementation
> **Tác giả:** Brainstorming Session

---

## Mục Lục

1. [Tổng Quan Bài Toán](#1-tổng-quan-bài-toán)
2. [Kiến Trúc Tổng Thể](#2-kiến-trúc-tổng-thể)
3. [Module `detect_regime_change` — Regime Detection Engine](#3-module-detect_regime_change--regime-detection-engine)
4. [Module `auto_trade` — Adaptive Close Time Consumer](#4-module-auto_trade--adaptive-close-time-consumer)
5. [Data Flow Chi Tiết](#5-data-flow-chi-tiết)
6. [Cấu Trúc Thư Mục](#6-cấu-trúc-thư-mục)
7. [API Contracts giữa 2 Module](#7-api-contracts-giữa-2-module)
8. [Cấu Hình Settings](#8-cấu-hình-settings)
9. [Safety Layers](#9-safety-layers)
10. [Giai Đoạn Triển Khai](#10-giai-đoạn-triển-khai)
11. [Decision Log](#11-decision-log)

---

## 1. Tổng Quan Bài Toán

### 1.1 Hiện trạng

Hệ thống auto-close timer hiện tại trong `auto_trade/execution/auto_close_timer.py`:

- Dùng `max_duration_hours` **cố định** (mặc định 4h) cho tất cả symbol
- Giá trị được config tĩnh trong `settings.yaml` → `auto_close.max_duration_hours: 4.0`
- Hàm `compute_deadline_utc()` tính deadline = `opened_at + max_duration_hours`
- Đã có cơ chế **override per-order** qua field `auto_close_deadline_utc` trong order record

### 1.2 Mục tiêu cải tiến

Thay vì dùng giá trị cố định, hệ thống sẽ:

1. **Tại thời điểm mở order** trên symbol X → chạy phân tích regime lịch sử cho symbol đó
2. Tính **trung bình thời gian mỗi regime** (bao lâu thì symbol đó thay đổi trạng thái)
3. Dùng giá trị đó làm `adaptive_duration_hours` → gắn vào `auto_close_deadline_utc` của order

### 1.3 Nguyên tắc thiết kế

- **Lazy computation** — chỉ tính khi có order mới, không chạy batch ngầm
- **Module separation** — logic detect regime nằm trong `detect_regime_change`, logic close time nằm trong `auto_trade`
- **Defense in depth** — min/max boundary + fallback tĩnh
- **Rust-ready** — Python thuần trước, port PELT sang Rust sau

---

## 2. Kiến Trúc Tổng Thể

```
┌─────────────────────────────────────────────────────────────────┐
│                     ORDER PLACEMENT FLOW                        │
│                                                                 │
│  User/Scanner places order on symbol X                          │
│         │                                                       │
│         ▼                                                       │
│  ┌──────────────────────────────────┐                           │
│  │   auto_trade/execution/          │                           │
│  │   adaptive_close_calculator.py   │  ◄── NEW (Consumer)       │
│  │                                  │                           │
│  │  1. Gọi RegimeDurationAnalyzer   │                           │
│  │  2. Nhận avg_regime_duration     │                           │
│  │  3. Apply clamp(min, max)        │                           │
│  │  4. Set auto_close_deadline_utc  │                           │
│  └──────────┬───────────────────────┘                           │
│             │ calls                                             │
│             ▼                                                   │
│  ┌──────────────────────────────────┐                           │
│  │   detect_regime_change/          │                           │
│  │   regime_duration_analyzer.py    │  ◄── NEW (Engine)         │
│  │                                  │                           │
│  │  1. Fetch OHLCV data (30-90d)    │                           │
│  │  2. Run PELT (change points)     │                           │
│  │  3. Run HMM (state duration)     │                           │
│  │  4. Combine → avg_duration_hrs   │                           │
│  └──────────┬───────────────────────┘                           │
│             │ uses                                              │
│             ▼                                                   │
│  ┌─────────────────┐  ┌──────────────────┐                      │
│  │  PELT Engine    │  │  modules/hmm/    │                      │
│  │  (ruptures lib) │  │  SwingsHMM       │                      │
│  │                 │  │  (existing)      │                      │
│  │  Future: Rust   │  │                  │                      │
│  └─────────────────┘  └──────────────────┘                      │
│                                                                 │
│  ┌──────────────────────────────────┐                           │
│  │   auto_trade/execution/          │                           │
│  │   auto_close_timer.py            │  ◄── EXISTING             │
│  │                                  │                           │
│  │  compute_deadline_utc() picks up │                           │
│  │  auto_close_deadline_utc from    │                           │
│  │  order record (override path)    │                           │
│  └──────────────────────────────────┘                           │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Module `detect_regime_change` — Regime Detection Engine

### 3.1 Trách nhiệm

Module này **CHỈ** chịu trách nhiệm:

- Phân tích dữ liệu giá lịch sử để phát hiện các điểm thay đổi regime
- Tính toán thống kê về độ dài regime (trung bình, median, phân vị)
- Cung cấp API thuần túy — **không biết** về orders, auto-close, hay trading logic

### 3.2 File mới: `regime_duration_analyzer.py`

```
detect_regime_change/
├── docs/
│   ├── market-regime-detection.md          # Existing research
│   └── adaptive-close-time-design.md       # This document
├── papers.md                               # Existing
├── __init__.py                             # NEW
├── regime_duration_analyzer.py             # NEW — Main engine
├── pelt_detector.py                        # NEW — PELT wrapper
├── hmm_regime_bridge.py                    # NEW — Bridge to modules/hmm
├── models.py                              # NEW — Data classes
└── rust_extensions/                        # PHASE 2 — Rust PELT
    ├── Cargo.toml
    └── src/
        └── lib.rs
```

### 3.3 Data Classes — `models.py`

```python
"""
detect_regime_change/models.py
==============================
Data models for regime change detection results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ChangePoint:
    """A single detected change point."""
    index: int                    # Index trong chuỗi dữ liệu
    timestamp: Optional[str]      # ISO timestamp (nếu có)


@dataclass
class RegimeSegment:
    """A single regime segment between two change points."""
    start_index: int
    end_index: int
    duration_seconds: float       # Độ dài regime tính bằng giây
    duration_hours: float         # Độ dài regime tính bằng giờ
    mean_return: Optional[float]  # Lợi nhuận trung bình trong segment
    volatility: Optional[float]   # Biến động trong segment


@dataclass
class RegimeDurationResult:
    """
    Kết quả phân tích regime duration cho một symbol.
    Đây là output chính mà auto_trade module sẽ consume.
    """
    symbol: str
    timeframe: str                          # Timeframe phân tích (ví dụ: "15m")
    
    # === PELT Results ===
    pelt_change_points: List[ChangePoint] = field(default_factory=list)
    pelt_segments: List[RegimeSegment] = field(default_factory=list)
    pelt_avg_duration_hours: Optional[float] = None
    pelt_median_duration_hours: Optional[float] = None
    
    # === HMM Results ===
    hmm_next_state_duration_hours: Optional[float] = None
    hmm_state: Optional[int] = None          # -1=BEARISH, 0=NEUTRAL, 1=BULLISH
    hmm_state_probability: Optional[float] = None
    
    # === Combined Result ===
    recommended_duration_hours: Optional[float] = None  # Giá trị cuối cùng
    
    # === Metadata ===
    data_points_analyzed: int = 0
    analysis_timestamp: Optional[str] = None
    computation_time_ms: Optional[float] = None
    error: Optional[str] = None
    
    @property
    def is_valid(self) -> bool:
        """Check if result has a valid recommendation."""
        return self.recommended_duration_hours is not None and self.error is None
```

### 3.4 PELT Detector — `pelt_detector.py`

```python
"""
detect_regime_change/pelt_detector.py
=====================================
Change Point Detection using PELT algorithm (ruptures library).

Phân tích chuỗi returns/volatility lịch sử để tìm các breakpoints
— mỗi khoảng giữa 2 breakpoints = 1 regime segment.
"""

from __future__ import annotations

import time
from typing import List, Optional, Tuple

import numpy as np

from modules.detect_regime_change.models import ChangePoint, RegimeSegment


def detect_change_points_pelt(
    returns: np.ndarray,
    timestamps: Optional[np.ndarray] = None,
    penalty: Optional[float] = None,
    model: str = "rbf",
    min_segment_length: int = 10,
) -> Tuple[List[ChangePoint], List[RegimeSegment]]:
    """
    Detect regime change points using PELT algorithm.
    
    Args:
        returns: Array of log-returns or price changes
        timestamps: Optional array of timestamps (datetime64)
        penalty: PELT penalty parameter (beta). None = auto BIC.
        model: Cost model — "rbf" (recommended), "l2", "normal"
        min_segment_length: Minimum segment length
    
    Returns:
        Tuple of (change_points, segments)
    """
    import ruptures as rpt

    n = len(returns)
    if n < min_segment_length * 2:
        return [], []

    # Auto-penalty via BIC if not provided
    if penalty is None:
        penalty = np.log(n) * returns.var()

    # Run PELT
    algo = rpt.Pelt(model=model, min_size=min_segment_length).fit(returns)
    breakpoints = algo.predict(pen=penalty)

    # Build change points
    change_points: List[ChangePoint] = []
    for bp in breakpoints[:-1]:  # Last element is always n
        ts = str(timestamps[bp]) if timestamps is not None and bp < len(timestamps) else None
        change_points.append(ChangePoint(index=bp, timestamp=ts))

    # Build regime segments
    segments: List[RegimeSegment] = []
    starts = [0] + breakpoints[:-1]
    ends = breakpoints

    for s, e in zip(starts, ends):
        seg_returns = returns[s:e]
        
        # Duration calculation
        if timestamps is not None and len(timestamps) > e - 1:
            t_start = np.datetime64(timestamps[s], 's')
            t_end = np.datetime64(timestamps[min(e, len(timestamps)) - 1], 's')
            duration_seconds = float((t_end - t_start) / np.timedelta64(1, 's'))
        else:
            # Fallback: estimate from candle count
            duration_seconds = float((e - s) * 900)  # Assume 15m candles = 900s

        segments.append(RegimeSegment(
            start_index=s,
            end_index=e,
            duration_seconds=duration_seconds,
            duration_hours=duration_seconds / 3600.0,
            mean_return=float(np.mean(seg_returns)) if len(seg_returns) > 0 else None,
            volatility=float(np.std(seg_returns)) if len(seg_returns) > 0 else None,
        ))

    return change_points, segments


def calculate_pelt_avg_duration(
    segments: List[RegimeSegment],
    trim_pct: float = 0.1,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Calculate average and median regime duration from PELT segments.
    
    Args:
        segments: List of regime segments
        trim_pct: Percentage of extreme segments to trim (0.1 = 10% each side)
    
    Returns:
        Tuple of (trimmed_mean_hours, median_hours)
    """
    if not segments:
        return None, None

    durations = sorted([s.duration_hours for s in segments])

    # Trimmed mean — loại bỏ các outlier cực đoan
    n = len(durations)
    if n >= 5:
        trim_count = max(1, int(n * trim_pct))
        trimmed = durations[trim_count:-trim_count]
    else:
        trimmed = durations

    avg = float(np.mean(trimmed)) if trimmed else None
    median = float(np.median(durations))

    return avg, median
```

### 3.5 HMM Bridge — `hmm_regime_bridge.py`

```python
"""
detect_regime_change/hmm_regime_bridge.py
==========================================
Bridge to existing modules/hmm for real-time regime state estimation.

Sử dụng SwingsHMM đã có để lấy:
- next_state_duration: thời gian dự kiến của state tiếp theo
- next_state: trạng thái dự đoán (BULLISH/NEUTRAL/BEARISH)
- probability: độ tin cậy
"""

from __future__ import annotations

from typing import Optional, Tuple

import pandas as pd

from modules.common.utils import log_error, log_warn


def estimate_hmm_regime_duration(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
) -> Tuple[Optional[float], Optional[int], Optional[float]]:
    """
    Use existing HMM module to estimate current regime duration.
    
    Args:
        df: DataFrame with OHLCV columns and DatetimeIndex
        train_ratio: Train/test split ratio
    
    Returns:
        Tuple of (duration_hours, state, probability)
        - duration_hours: predicted next state duration in hours
        - state: -1 (BEARISH), 0 (NEUTRAL), 1 (BULLISH) 
        - probability: confidence of prediction
    """
    try:
        from modules.hmm import hmm_swings

        result = hmm_swings(df, train_ratio=train_ratio, eval_mode=False)

        # next_state_duration từ HMM_SWINGS là đơn vị phụ thuộc timeframe
        # Cần convert sang hours
        duration_raw = result.next_state_duration
        state = result.next_state_with_high_order_hmm
        probability = result.next_state_probability

        # Determine timeframe from data
        if isinstance(df.index, pd.DatetimeIndex) and len(df.index) >= 2:
            interval_seconds = (df.index[1] - df.index[0]).total_seconds()
        else:
            interval_seconds = 900  # Default 15m

        # Convert duration to hours
        # duration_raw là số "đơn vị" (candles hoặc swing distance)
        # _calculate_duration đã convert sang hours/minutes tùy interval
        # Nhưng ta cần đảm bảo kết quả luôn là hours
        if interval_seconds >= 3600:
            # Hourly candles → duration_raw đã là hours
            duration_hours = float(duration_raw)
        elif interval_seconds >= 60:
            # Minute candles → duration_raw là minutes
            duration_hours = float(duration_raw) / 60.0
        else:
            duration_hours = float(duration_raw) / 3600.0

        return duration_hours, state, probability

    except Exception as e:
        log_error(f"HMM regime estimation failed: {e}")
        return None, None, None
```

### 3.6 Main Engine — `regime_duration_analyzer.py`

```python
"""
detect_regime_change/regime_duration_analyzer.py
================================================
Main engine: kết hợp PELT + HMM để tính recommended regime duration cho symbol.

Đây là entry point duy nhất mà auto_trade module gọi.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

from modules.common.utils import log_error, log_info, log_warn
from modules.detect_regime_change.hmm_regime_bridge import estimate_hmm_regime_duration
from modules.detect_regime_change.models import RegimeDurationResult
from modules.detect_regime_change.pelt_detector import (
    calculate_pelt_avg_duration,
    detect_change_points_pelt,
)


class RegimeDurationAnalyzer:
    """
    Phân tích regime duration cho một symbol.
    
    Kết hợp 2 phương pháp:
    1. PELT (offline) — tìm change points trong dữ liệu lịch sử
    2. HMM (real-time) — dự đoán duration của state hiện tại/tiếp theo
    
    Công thức kết hợp:
        recommended = w_pelt * pelt_avg + w_hmm * hmm_duration
        
    Trong đó weights phụ thuộc vào confidence của HMM:
    - HMM probability cao (>0.7) → w_hmm = 0.6, w_pelt = 0.4
    - HMM probability trung bình → w_hmm = 0.3, w_pelt = 0.7
    - HMM thất bại → recommended = pelt_avg (100% PELT)
    """

    def __init__(
        self,
        lookback_days: int = 60,
        pelt_model: str = "rbf",
        pelt_penalty: Optional[float] = None,
        pelt_min_segment: int = 10,
        pelt_trim_pct: float = 0.1,
        hmm_train_ratio: float = 0.8,
        hmm_high_confidence_threshold: float = 0.7,
        w_pelt_high_conf: float = 0.4,
        w_hmm_high_conf: float = 0.6,
        w_pelt_low_conf: float = 0.7,
        w_hmm_low_conf: float = 0.3,
    ):
        self.lookback_days = lookback_days
        self.pelt_model = pelt_model
        self.pelt_penalty = pelt_penalty
        self.pelt_min_segment = pelt_min_segment
        self.pelt_trim_pct = pelt_trim_pct
        self.hmm_train_ratio = hmm_train_ratio
        self.hmm_high_confidence_threshold = hmm_high_confidence_threshold
        self.w_pelt_high_conf = w_pelt_high_conf
        self.w_hmm_high_conf = w_hmm_high_conf
        self.w_pelt_low_conf = w_pelt_low_conf
        self.w_hmm_low_conf = w_hmm_low_conf

    def analyze(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str = "15m",
    ) -> RegimeDurationResult:
        """
        Main entry point: phân tích regime duration cho một symbol.
        
        Args:
            df: DataFrame OHLCV với DatetimeIndex (đã fetch sẵn, ít nhất 30 ngày)
            symbol: Tên symbol (ví dụ: "BTC/USDT")
            timeframe: Timeframe của data (ví dụ: "15m", "1h")
        
        Returns:
            RegimeDurationResult với recommended_duration_hours
        """
        start_time = time.time()
        
        result = RegimeDurationResult(
            symbol=symbol,
            timeframe=timeframe,
            data_points_analyzed=len(df),
            analysis_timestamp=datetime.now(timezone.utc).isoformat(),
        )

        try:
            # === 1. Prepare returns ===
            close_prices = df["close"].values.astype(float)
            log_returns = np.diff(np.log(close_prices + 1e-10))
            
            timestamps = None
            if isinstance(df.index, pd.DatetimeIndex):
                timestamps = df.index[1:].values  # Align with returns

            # === 2. PELT Change Point Detection ===
            try:
                change_points, segments = detect_change_points_pelt(
                    returns=log_returns,
                    timestamps=timestamps,
                    penalty=self.pelt_penalty,
                    model=self.pelt_model,
                    min_segment_length=self.pelt_min_segment,
                )
                result.pelt_change_points = change_points
                result.pelt_segments = segments
                
                avg_hrs, median_hrs = calculate_pelt_avg_duration(
                    segments, trim_pct=self.pelt_trim_pct,
                )
                result.pelt_avg_duration_hours = avg_hrs
                result.pelt_median_duration_hours = median_hrs
                
                log_info(
                    f"PELT [{symbol}]: {len(change_points)} change points, "
                    f"avg={avg_hrs:.2f}h, median={median_hrs:.2f}h"
                    if avg_hrs and median_hrs else
                    f"PELT [{symbol}]: {len(change_points)} change points"
                )
            except Exception as pelt_err:
                log_warn(f"PELT analysis failed for {symbol}: {pelt_err}")

            # === 3. HMM Regime Duration ===
            try:
                hmm_duration, hmm_state, hmm_prob = estimate_hmm_regime_duration(
                    df=df,
                    train_ratio=self.hmm_train_ratio,
                )
                result.hmm_next_state_duration_hours = hmm_duration
                result.hmm_state = hmm_state
                result.hmm_state_probability = hmm_prob
                
                if hmm_duration is not None:
                    log_info(
                        f"HMM [{symbol}]: state={hmm_state}, "
                        f"duration={hmm_duration:.2f}h, prob={hmm_prob:.3f}"
                    )
            except Exception as hmm_err:
                log_warn(f"HMM analysis failed for {symbol}: {hmm_err}")

            # === 4. Combine PELT + HMM ===
            result.recommended_duration_hours = self._combine_results(result)
            
            if result.recommended_duration_hours is not None:
                log_info(
                    f"Regime Duration [{symbol}]: "
                    f"recommended={result.recommended_duration_hours:.2f}h"
                )

        except Exception as e:
            result.error = str(e)
            log_error(f"Regime analysis failed for {symbol}: {e}")

        result.computation_time_ms = (time.time() - start_time) * 1000
        return result

    def _combine_results(self, result: RegimeDurationResult) -> Optional[float]:
        """
        Kết hợp PELT và HMM bằng weighted average.
        
        Strategy:
        - Nếu cả 2 có kết quả → weighted average dựa trên HMM confidence
        - Nếu chỉ PELT → dùng PELT avg (hoặc median)
        - Nếu chỉ HMM → dùng HMM duration
        - Nếu cả 2 thất bại → None (sẽ fallback ở tầng auto_trade)
        """
        pelt_val = result.pelt_avg_duration_hours
        hmm_val = result.hmm_next_state_duration_hours
        hmm_prob = result.hmm_state_probability or 0.0

        # Cả 2 có kết quả
        if pelt_val is not None and hmm_val is not None and hmm_val > 0:
            if hmm_prob >= self.hmm_high_confidence_threshold:
                w_pelt = self.w_pelt_high_conf
                w_hmm = self.w_hmm_high_conf
            else:
                w_pelt = self.w_pelt_low_conf
                w_hmm = self.w_hmm_low_conf
            
            return w_pelt * pelt_val + w_hmm * hmm_val

        # Chỉ PELT
        if pelt_val is not None:
            return pelt_val

        # Chỉ HMM
        if hmm_val is not None and hmm_val > 0:
            return hmm_val

        return None
```

---

## 4. Module `auto_trade` — Adaptive Close Time Consumer

### 4.1 Trách nhiệm

Module này **CHỈ** chịu trách nhiệm:

- Gọi `RegimeDurationAnalyzer` để lấy `recommended_duration_hours`
- Apply safety clamp (min/max boundary)
- Fallback về giá trị tĩnh nếu cần
- Set `auto_close_deadline_utc` lên order record

### 4.2 File mới: `adaptive_close_calculator.py`

```
auto_trade/
├── execution/
│   ├── auto_close_timer.py            # EXISTING — không thay đổi
│   ├── auto_close_timer_job.py        # EXISTING — không thay đổi
│   └── adaptive_close_calculator.py   # NEW — Adaptive close logic
```

### 4.3 Implementation — `adaptive_close_calculator.py`

```python
"""
auto_trade/execution/adaptive_close_calculator.py
==================================================
Adaptive Close Time Calculator.

Tại thời điểm mở order, gọi RegimeDurationAnalyzer để tính
adaptive_duration_hours cho symbol đó, rồi set lên order record.

Tích hợp vào flow mở order hiện có.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from modules.common.ui.logging import log_error, log_info, log_warn


# ─── Configuration defaults ───
DEFAULT_MIN_DURATION_HOURS = 1.0
DEFAULT_MAX_DURATION_HOURS = 12.0
DEFAULT_FALLBACK_DURATION_HOURS = 4.0
DEFAULT_LOOKBACK_DAYS = 60


class AdaptiveCloseCalculator:
    """
    Calculator for adaptive close deadlines based on regime change analysis.
    
    Usage:
        calculator = AdaptiveCloseCalculator(settings_manager)
        deadline_utc = calculator.compute_adaptive_deadline(
            symbol="BTC/USDT",
            opened_at=datetime.now(timezone.utc),
        )
        
        if deadline_utc:
            order_updates["auto_close_deadline_utc"] = deadline_utc.isoformat()
    """

    def __init__(self, settings_manager):
        self.settings_manager = settings_manager

    def _get_config(self) -> Dict[str, Any]:
        """Get adaptive close configuration from settings."""
        cfg = self.settings_manager.get("auto_close", {}) or {}
        adaptive = cfg.get("adaptive", {}) or {}
        return {
            "enabled": bool(adaptive.get("enabled", False)),
            "min_duration_hours": float(adaptive.get("min_duration_hours", DEFAULT_MIN_DURATION_HOURS)),
            "max_duration_hours": float(adaptive.get("max_duration_hours", DEFAULT_MAX_DURATION_HOURS)),
            "fallback_duration_hours": float(cfg.get("max_duration_hours", DEFAULT_FALLBACK_DURATION_HOURS)),
            "lookback_days": int(adaptive.get("lookback_days", DEFAULT_LOOKBACK_DAYS)),
            "timeframe": str(adaptive.get("timeframe", "") or cfg.get("timeframe", "15m")),
        }

    def compute_adaptive_deadline(
        self,
        symbol: str,
        opened_at: datetime,
        ohlcv_df=None,
    ) -> Optional[datetime]:
        """
        Compute adaptive close deadline for an order.
        
        Args:
            symbol: Trading symbol (e.g., "BTC/USDT")
            opened_at: Order open timestamp (UTC)
            ohlcv_df: Optional pre-fetched OHLCV DataFrame.
                       If None, will be fetched internally.
        
        Returns:
            datetime: Adaptive close deadline in UTC, or None if disabled/failed.
        """
        cfg = self._get_config()
        
        if not cfg["enabled"]:
            return None

        try:
            # === 1. Fetch data if not provided ===
            if ohlcv_df is None:
                ohlcv_df = self._fetch_ohlcv(
                    symbol=symbol,
                    timeframe=cfg["timeframe"],
                    lookback_days=cfg["lookback_days"],
                )

            if ohlcv_df is None or len(ohlcv_df) < 100:
                log_warn(
                    f"Adaptive close: insufficient data for {symbol} "
                    f"({len(ohlcv_df) if ohlcv_df is not None else 0} candles), "
                    f"falling back to static {cfg['fallback_duration_hours']}h"
                )
                return opened_at + timedelta(hours=cfg["fallback_duration_hours"])

            # === 2. Run regime analysis ===
            from modules.detect_regime_change.regime_duration_analyzer import (
                RegimeDurationAnalyzer,
            )

            analyzer = RegimeDurationAnalyzer(
                lookback_days=cfg["lookback_days"],
            )
            analysis = analyzer.analyze(
                df=ohlcv_df,
                symbol=symbol,
                timeframe=cfg["timeframe"],
            )

            # === 3. Extract and clamp ===
            if analysis.is_valid and analysis.recommended_duration_hours is not None:
                raw_hours = analysis.recommended_duration_hours
                clamped_hours = max(
                    cfg["min_duration_hours"],
                    min(cfg["max_duration_hours"], raw_hours),
                )

                log_info(
                    f"Adaptive close [{symbol}]: "
                    f"raw={raw_hours:.2f}h → clamped={clamped_hours:.2f}h "
                    f"(min={cfg['min_duration_hours']}h, max={cfg['max_duration_hours']}h)"
                )

                return opened_at + timedelta(hours=clamped_hours)

            # === 4. Fallback ===
            log_warn(
                f"Adaptive close [{symbol}]: analysis invalid "
                f"(error={analysis.error}), falling back to {cfg['fallback_duration_hours']}h"
            )
            return opened_at + timedelta(hours=cfg["fallback_duration_hours"])

        except Exception as e:
            log_error(f"Adaptive close calculation failed for {symbol}: {e}")
            return opened_at + timedelta(hours=cfg["fallback_duration_hours"])

    def _fetch_ohlcv(self, symbol: str, timeframe: str, lookback_days: int):
        """
        Fetch historical OHLCV data for regime analysis.
        
        Sử dụng data fetcher đã có trong project.
        """
        try:
            import ccxt
            import pandas as pd

            exchange = ccxt.binance({"enableRateLimit": True})
            since_ms = int(
                (datetime.now(timezone.utc) - timedelta(days=lookback_days)).timestamp() * 1000
            )

            ohlcv = exchange.fetch_ohlcv(
                symbol, timeframe=timeframe, since=since_ms, limit=1000
            )

            if not ohlcv:
                return None

            df = pd.DataFrame(
                ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df.set_index("timestamp", inplace=True)
            return df

        except Exception as e:
            log_error(f"Failed to fetch OHLCV for {symbol}: {e}")
            return None
```

---

## 5. Data Flow Chi Tiết

```
╔══════════════════════════════════════════════════════════════════════╗
║                    ORDER PLACEMENT TRIGGER                          ║
║                                                                     ║
║  1. Scanner/User places order on BTC/USDT                          ║
║     │                                                               ║
║  2. │ auto_trade gọi AdaptiveCloseCalculator                       ║
║     │  .compute_adaptive_deadline("BTC/USDT", opened_at)           ║
║     │                                                               ║
║     ▼                                                               ║
║  3. AdaptiveCloseCalculator:                                        ║
║     ├─ Check config: adaptive.enabled == true?                      ║
║     ├─ Fetch OHLCV 60 ngày (nếu chưa có)                          ║
║     │                                                               ║
║  4. │ Gọi RegimeDurationAnalyzer.analyze(df, "BTC/USDT")          ║
║     │  ┌──────────────────────────────────────────┐                 ║
║     │  │  detect_regime_change module:             │                 ║
║     │  │                                           │                 ║
║     │  │  a) PELT: log-returns → ruptures.Pelt()  │                 ║
║     │  │     → 15 change points                    │                 ║
║     │  │     → avg segment = 3.2h, median = 2.8h  │                 ║
║     │  │                                           │                 ║
║     │  │  b) HMM: df → SwingsHMM.analyze()        │                 ║
║     │  │     → next_state_duration = 4.1h          │                 ║
║     │  │     → state = BULLISH, prob = 0.82        │                 ║
║     │  │                                           │                 ║
║     │  │  c) Combine: prob > 0.7 (high conf)      │                 ║
║     │  │     → 0.4 * 3.2 + 0.6 * 4.1 = 3.74h     │                 ║
║     │  └──────────────────────────────────────────┘                 ║
║     │                                                               ║
║  5. │ Apply clamp: max(1.0, min(12.0, 3.74)) = 3.74h              ║
║     │                                                               ║
║  6. │ Return: opened_at + 3h44m = deadline_utc                     ║
║     │                                                               ║
║     ▼                                                               ║
║  7. Order record updated:                                           ║
║     { "auto_close_deadline_utc": "2026-03-09T05:00:00Z" }         ║
║                                                                     ║
║  8. auto_close_timer_job.run() picks up this override               ║
║     via compute_deadline_utc() → existing path, NO CHANGES          ║
╚══════════════════════════════════════════════════════════════════════╝
```

---

## 6. Cấu Trúc Thư Mục

### 6.1 `detect_regime_change/` (NEW files)

```
modules/detect_regime_change/
├── docs/
│   ├── market-regime-detection.md           # Existing — research doc
│   └── adaptive-close-time-design.md        # THIS DOCUMENT
├── papers.md                                # Existing
│
├── __init__.py                              # NEW — exports RegimeDurationAnalyzer
├── models.py                               # NEW — ChangePoint, RegimeSegment, RegimeDurationResult
├── pelt_detector.py                         # NEW — PELT wrapper (ruptures)
├── hmm_regime_bridge.py                     # NEW — Bridge to modules/hmm
├── regime_duration_analyzer.py              # NEW — Main engine (combines PELT + HMM)
│
└── rust_extensions/                         # PHASE 2 — Rust PELT implementation
    ├── Cargo.toml
    └── src/
        └── lib.rs                           # pyo3 bindings for PELT
```

### 6.2 `auto_trade/` (NEW files)

```
modules/auto_trade/
├── execution/
│   ├── auto_close_timer.py                  # EXISTING — unchanged
│   ├── auto_close_timer_job.py              # EXISTING — unchanged
│   └── adaptive_close_calculator.py         # NEW — AdaptiveCloseCalculator
├── settings.yaml                            # MODIFIED — add adaptive section
```

---

## 7. API Contracts giữa 2 Module

### 7.1 Interface: `detect_regime_change` → `auto_trade`

```python
# auto_trade gọi detect_regime_change qua entry point DUY NHẤT:

from modules.detect_regime_change.regime_duration_analyzer import RegimeDurationAnalyzer
from modules.detect_regime_change.models import RegimeDurationResult

analyzer = RegimeDurationAnalyzer(lookback_days=60)
result: RegimeDurationResult = analyzer.analyze(df, symbol="BTC/USDT", timeframe="15m")

# Consumer chỉ cần check:
if result.is_valid:
    hours = result.recommended_duration_hours  # float
```

### 7.2 Dependency Direction

```
auto_trade  ──depends on──►  detect_regime_change  ──depends on──►  modules/hmm
                                                    ──depends on──►  ruptures (pip)

detect_regime_change  ──KHÔNG depends on──►  auto_trade
modules/hmm           ──KHÔNG depends on──►  detect_regime_change
```

### 7.3 Điểm tích hợp vào flow mở order hiện tại

Khi mở order, cần thêm 1 bước gọi `AdaptiveCloseCalculator` **trước** khi lưu order vào DB:

```python
# Trong execution flow hiện tại (pseudo-code):

from modules.auto_trade.execution.adaptive_close_calculator import AdaptiveCloseCalculator

# Sau khi order được place thành công trên exchange:
calculator = AdaptiveCloseCalculator(settings_manager)
adaptive_deadline = calculator.compute_adaptive_deadline(
    symbol=order["symbol"],
    opened_at=order_opened_at,
)

if adaptive_deadline is not None:
    order["auto_close_deadline_utc"] = adaptive_deadline.isoformat()

# Lưu order vào DB → auto_close_timer_job sẽ tự pick up
```

---

## 8. Cấu Hình Settings

### 8.1 Mở rộng `settings.yaml`

```yaml
auto_close:
  enabled: true
  max_duration_enabled: true
  max_duration_hours: 4.0          # Fallback tĩnh (giữ nguyên)
  daily_close_enabled: false
  daily_close_time: '22:00'
  daily_close_days: '1234567'
  grace_period_minutes: 5
  tp_offset_pct: 0.05

  # ─── NEW: Adaptive Close Time ───
  adaptive:
    enabled: false                  # Mặc định tắt, bật khi sẵn sàng
    min_duration_hours: 1.0         # Floor — không close sớm hơn 1h
    max_duration_hours: 12.0        # Ceiling — không hold quá 12h
    lookback_days: 60               # Số ngày data lịch sử để phân tích
    timeframe: '15m'                # Timeframe cho regime analysis
```

### 8.2 Logic precedence

```
1. adaptive.enabled = true?
   ├─ YES → Tính adaptive_deadline → set auto_close_deadline_utc trên order
   │        ├─ Thành công → dùng adaptive deadline
   │        └─ Thất bại → fallback về max_duration_hours (4h)
   │
   └─ NO  → Luồng cũ: dùng max_duration_hours cố định (4h)

2. auto_close_timer_job.run() chạy bình thường:
   - compute_deadline_utc() check auto_close_deadline_utc override → dùng nếu có
   - Nếu không có override → tính từ max_duration_hours
   → KHÔNG CẦN SỬA auto_close_timer.py hay auto_close_timer_job.py
```

---

## 9. Safety Layers

### 9.1 Layer 1 — Clamp Boundary

```python
clamped = max(min_duration_hours, min(max_duration_hours, raw_hours))
# Ví dụ: min=1.0, max=12.0
# raw=0.3h → clamped=1.0h
# raw=48h  → clamped=12.0h
# raw=3.7h → clamped=3.7h
```

### 9.2 Layer 2 — Fallback tĩnh

```python
# Nếu RegimeDurationAnalyzer trả error:
if not analysis.is_valid:
    return opened_at + timedelta(hours=fallback_duration_hours)  # 4h
```

### 9.3 Layer 3 — Data sufficiency check

```python
if ohlcv_df is None or len(ohlcv_df) < 100:
    # Không đủ data → fallback
    return opened_at + timedelta(hours=fallback_duration_hours)
```

### 9.4 Layer 4 — Exception handling

```python
try:
    # ... entire flow
except Exception:
    return opened_at + timedelta(hours=fallback_duration_hours)
# KHÔNG BAO GIỜ return None mà không có fallback
```

---

## 10. Giai Đoạn Triển Khai

### Phase 1 — Python Core (ưu tiên)

| # | Task | Module | Ước lượng |
|---|------|--------|-----------|
| 1.1 | Tạo `detect_regime_change/models.py` | detect_regime_change | 30 min |
| 1.2 | Tạo `detect_regime_change/pelt_detector.py` | detect_regime_change | 1h |
| 1.3 | Tạo `detect_regime_change/hmm_regime_bridge.py` | detect_regime_change | 1h |
| 1.4 | Tạo `detect_regime_change/regime_duration_analyzer.py` | detect_regime_change | 1.5h |
| 1.5 | Tạo `detect_regime_change/__init__.py` | detect_regime_change | 15 min |
| 1.6 | Tạo `auto_trade/execution/adaptive_close_calculator.py` | auto_trade | 1h |
| 1.7 | Cập nhật `settings.yaml` — thêm `adaptive` section | auto_trade | 15 min |
| 1.8 | Cập nhật `settings_manager.py` — parse adaptive config | auto_trade | 30 min |
| 1.9 | Tích hợp vào flow mở order | auto_trade | 1h |
| 1.10 | Thêm `ruptures` vào requirements | root | 5 min |
| 1.11 | Tests (pytest) | cả hai module | 2h |

### Phase 2 — Rust Optimization (sau)

| # | Task | Module | Ước lượng |
|---|------|--------|-----------|
| 2.1 | Setup `rust_extensions/Cargo.toml` với pyo3 | detect_regime_change | 30 min |
| 2.2 | Implement PELT core in Rust | detect_regime_change | 4h |
| 2.3 | PyO3 bindings | detect_regime_change | 1h |
| 2.4 | Fallback: Python nếu Rust module chưa build | detect_regime_change | 30 min |
| 2.5 | Benchmark Python vs Rust | detect_regime_change | 1h |

### Phase 3 — GUI & Polish (sau)

| # | Task | Module | Ước lượng |
|---|------|--------|-----------|
| 3.1 | Thêm tab/section trong GUI cho adaptive config | auto_trade/gui | 2h |
| 3.2 | Hiển thị adaptive deadline trên Scheduled Exits panel | auto_trade/gui | 1h |
| 3.3 | Logging chi tiết trong live log | auto_trade/gui | 30 min |

---

## 11. Decision Log

| # | Quyết định | Phương án đã xét | Lý do chọn |
|---|-----------|------------------|------------|
| D1 | **Hybrid: PELT + HMM** | A) Chỉ PELT, B) Chỉ HMM, C) Kết hợp | PELT cho baseline chính xác offline, HMM cho real-time refinement. Tận dụng module HMM đã có sẵn. |
| D2 | **Lazy per-order** (compute on demand) | A) Batch định kỳ, B) Per-order, C) Both | Tránh overhead batch job chạy ngầm. Tính toán chỉ khi thực sự cần (có order). Kết quả gắn vào order qua override mechanism đã có. |
| D3 | **Module tách biệt** | A) Tất cả trong auto_trade, B) Tách ra | Separation of concerns: detect_regime_change không biết gì về trading, auto_trade chỉ consume kết quả. Dễ test, dễ reuse. |
| D4 | **Clamped + Fallback** | A) Chỉ clamp, B) Chỉ fallback, C) Cả hai | Phòng thủ đa lớp: clamp chặn giá trị cực đoan, fallback chặn lỗi hệ thống. Không bao giờ để order "treo" vô thời hạn. |
| D5 | **Python trước, Rust sau** | A) Python thuần, B) Rust ngay, C) Python → Rust | Kiểm chứng logic trước. Port sang Rust khi xác nhận PELT là bottleneck thực sự. Pattern PyO3 đã có sẵn (atc_rust). |
| D6 | **Dùng `auto_close_deadline_utc` override** | A) Sửa compute_deadline_utc, B) Dùng override field đã có | Zero change cho auto_close_timer.py. Override mechanism đã hoạt động, chỉ cần set giá trị. Backward compatible 100%. |
| D7 | **Weighted average PELT+HMM** dựa trên HMM confidence | A) Simple average, B) Chỉ PELT, C) Weighted | HMM confidence cao → tin HMM hơn (biết state hiện tại). HMM confidence thấp → nghiêng về PELT (số liệu lịch sử ổn định hơn). |

---

*Document generated from brainstorming session — 2026-03-09*
