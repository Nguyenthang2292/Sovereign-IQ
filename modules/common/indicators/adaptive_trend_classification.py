"""Adaptive Trend Classification helpers.

Hiện tại file này cung cấp phiên bản KAMA được tinh chỉnh cho
Adaptive Trend Classification (ATC), tái sử dụng implementation
KAMA chuẩn ở `modules.common.indicators.momentum`.

Mục tiêu là bám sát nhất có thể với KAMA trong Pine Script:

- Pine dùng:
    fast = 0.666
    slow = 0.064
  trực tiếp làm smoothing factors trong công thức:
    smooth = (ratio * (fast - slow) + slow) ^ 2

- Implementation Python trong `momentum.py` dùng dạng chuẩn:
    fast_sc = 2 / (fast + 1)
    slow_sc = 2 / (slow + 1)
    smooth  = (ratio * (fast_sc - slow_sc) + slow_sc) ** 2

Nếu giải ngược:
    fast_sc ≈ 0.666  → fast ≈ 2
    slow_sc ≈ 0.064  → slow ≈ 30

Nghĩa là bộ tham số (fast=2, slow=30) trong Python gần như
trùng với (fast=0.666, slow=0.064) trong Pine về mặt smoothing.
"""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

import pandas as pd
import pandas_ta as ta

from .momentum import calculate_kama_series


def diflen(length: int, robustness: str = "Medium") -> Tuple[int, int, int, int, int, int, int, int]:
    """
    Chuyển đổi logic `diflen(length)` trong Pine Script sang Python.

    Pine:
        - trả về: [L1, L2, L3, L4, L_1, L_2, L_3, L_4]
        - phụ thuộc tham số `robustness` ∈ {"Narrow", "Medium", "Wide"}.
    """
    robustness = robustness or "Medium"

    if robustness == "Narrow":
        L1, L_1 = length + 1, length - 1
        L2, L_2 = length + 2, length - 2
        L3, L_3 = length + 3, length - 3
        L4, L_4 = length + 4, length - 4
    elif robustness == "Medium":
        L1, L_1 = length + 1, length - 1
        L2, L_2 = length + 2, length - 2
        L3, L_3 = length + 4, length - 4
        L4, L_4 = length + 6, length - 6
    else:  # "Wide" hoặc bất kỳ giá trị khác
        L1, L_1 = length + 1, length - 1
        L2, L_2 = length + 3, length - 3
        L3, L_3 = length + 5, length - 5
        L4, L_4 = length + 7, length - 7

    return L1, L2, L3, L4, L_1, L_2, L_3, L_4


def calculate_kama_atc(
    prices: pd.Series,
    length: int = 28,
) -> Optional[pd.Series]:
    """
    KAMA cho Adaptive Trend Classification (ATC).

    - Dùng công thức KAMA từ `momentum.calculate_kama_series`
    - Tham số được chọn để khớp với Pine Script:
        + length  ~ `kama_len` trong Pine (mặc định 28)
        + fast    ~ 2   → fast_sc ≈ 0.666 (Pine: 0.666)
        + slow    ~ 30  → slow_sc ≈ 0.064 (Pine: 0.064)

    Args:
        prices: Chuỗi giá (thường là close) dạng pandas Series.
        length: Window KAMA, tương ứng `kama_len` trong Pine.

    Returns:
        pandas.Series KAMA với cùng index như `prices`, hoặc None nếu không tính được.
    """
    if prices is None or len(prices) == 0:
        return None

    return calculate_kama_series(
        prices=prices,
        period=length,
        fast=2,
        slow=30,
    )


def exp_growth(
    L: float,
    index: Optional[pd.Index] = None,
    *,
    cutout: int = 0,
) -> pd.Series:
    """
    Port của hàm Pine Script:

        e(L) =>
            bars   = bar_index == 0 ? 1 : bar_index
            cuttime = time[cutout]
            x = 1.0
            if not na(cuttime) and time >= cuttime
                x := math.pow(math.e, L * (bar_index - cutout))
            x

    Trong TradingView, `time` và `bar_index` là biến môi trường toàn cục.
    Ở đây ta xấp xỉ bằng cách dùng chỉ số vị trí (0, 1, 2, ...) của Series.

    Args:
        L: Lambda (growth rate).
        index: Index thời gian / bar của dữ liệu. Nếu None sẽ xây dựng
            index RangeIndex mặc định.
        cutout: Số bar bỏ qua đầu chuỗi (tương tự input `cutout`).

    Returns:
        pd.Series hệ số nhân e(L) theo thời gian.
    """
    if index is None:
        index = pd.RangeIndex(0, 0)

    # Sử dụng vị trí 0..n-1 làm tương đương `bar_index`
    bars = pd.Series(range(len(index)), index=index, dtype="float64")
    # Trong Pine: nếu bar_index == 0 thì bars = 1, còn lại = bar_index
    bars = bars.where(bars != 0, 1.0)

    # Điều kiện "đã qua cutout"
    active = bars >= cutout
    x = pd.Series(1.0, index=index, dtype="float64")
    x[active] = (pd.NA,)  # placeholder, sau đó gán giá trị thật
    x.loc[active] = (pd.np.e ** (L * (bars[active] - cutout))).astype("float64")  # type: ignore[attr-defined]
    return x


def ma_calculation(
    source: pd.Series,
    length: int,
    ma_type: str,
) -> Optional[pd.Series]:
    """
    Port của hàm Pine Script:

        ma_calculation(source, length, ma_type) =>
            if ma_type == "EMA"
                ta.ema(source, length)
            else if ma_type == "HMA"
                ta.sma(source, length)
            else if ma_type == "WMA"
                ta.wma(source, length)
            else if ma_type == "DEMA"
                ta.dema(source, length)
            else if ma_type == "LSMA"
                lsma(source,length)
            else if ma_type == "KAMA"
                kama(source, length)
            else
                na

    Ghi chú:
    - HMA trong script gốc map sang SMA (không phải Hull MA cổ điển),
      nên ở đây cũng dùng `ta.sma` để bám sát hành vi đó.
    - LSMA dùng `ta.linreg`, tương đương với `lsma()` trong Pine.
    - KAMA gọi `calculate_kama_atc` đã chuẩn hóa tham số fast/slow.
    """
    if source is None or len(source) == 0:
        return None

    ma = ma_type.upper()

    if ma == "EMA":
        return ta.ema(source, length=length)
    if ma == "HMA":
        # Pine: HMA branch đang dùng ta.sma, không phải Hull MA chuẩn.
        return ta.sma(source, length=length)
    if ma == "WMA":
        return ta.wma(source, length=length)
    if ma == "DEMA":
        return ta.dema(source, length=length)
    if ma == "LSMA":
        # LSMA ~ Linear Regression (Least Squares Moving Average)
        return ta.linreg(source, length=length)
    if ma == "KAMA":
        return calculate_kama_atc(source, length=length)

    return None


def set_of_moving_averages(
    length: int,
    source: pd.Series,
    ma_type: str,
    robustness: str = "Medium",
):
    """
    Port của hàm Pine Script:

        SetOfMovingAverages(length, source, ma_type) =>
            [L1,L2,L3,L4,L_1,L_2,L_3,L_4] = diflen(length)
            MA   = ma_calculation(source,  length, ma_type)
            MA1  = ma_calculation(source,  L1,     ma_type)
            MA2  = ma_calculation(source,  L2,     ma_type)
            MA3  = ma_calculation(source,  L3,     ma_type)
            MA4  = ma_calculation(source,  L4,     ma_type)
            MA_1 = ma_calculation(source, L_1,     ma_type)
            MA_2 = ma_calculation(source, L_2,     ma_type)
            MA_3 = ma_calculation(source, L_3,     ma_type)
            MA_4 = ma_calculation(source, L_4,     ma_type)
            [MA, MA1, MA2, MA3, MA4, MA_1, MA_2, MA_3, MA_4]
    """
    if source is None or len(source) == 0:
        return None

    L1, L2, L3, L4, L_1, L_2, L_3, L_4 = diflen(length, robustness=robustness)

    MA = ma_calculation(source, length, ma_type)
    MA1 = ma_calculation(source, L1, ma_type)
    MA2 = ma_calculation(source, L2, ma_type)
    MA3 = ma_calculation(source, L3, ma_type)
    MA4 = ma_calculation(source, L4, ma_type)
    MA_1 = ma_calculation(source, L_1, ma_type)
    MA_2 = ma_calculation(source, L_2, ma_type)
    MA_3 = ma_calculation(source, L_3, ma_type)
    MA_4 = ma_calculation(source, L_4, ma_type)

    return MA, MA1, MA2, MA3, MA4, MA_1, MA_2, MA_3, MA_4


def crossover(series_a: pd.Series, series_b: pd.Series) -> pd.Series:
    """
    Tương đương `ta.crossover(a, b)` trong Pine:
    - true khi a hiện tại > b hiện tại và a[1] <= b[1]
    """
    prev_a = series_a.shift(1)
    prev_b = series_b.shift(1)
    return (series_a > series_b) & (prev_a <= prev_b)


def crossunder(series_a: pd.Series, series_b: pd.Series) -> pd.Series:
    """
    Tương đương `ta.crossunder(a, b)` trong Pine:
    - true khi a hiện tại < b hiện tại và a[1] >= b[1]
    """
    prev_a = series_a.shift(1)
    prev_b = series_b.shift(1)
    return (series_a < series_b) & (prev_a >= prev_b)


def generate_signal_from_ma(
    price: pd.Series,
    ma: pd.Series,
) -> pd.Series:
    """
    Port của hàm Pine Script:

        signal(ma) =>
            var int sig = 0
            if ta.crossover(close, ma)
                sig := 1
            if ta.crossunder(close, ma)
                sig := -1
            sig

    Ở đây trả về Series các giá trị {-1, 0, 1}.
    """
    sig = pd.Series(0, index=price.index, dtype="int8")
    up = crossover(price, ma)
    down = crossunder(price, ma)
    sig[up] = 1
    sig[down] = -1
    return sig


def equity_series(
    starting_equity: float,
    sig: pd.Series,
    R: pd.Series,
    *,
    L: float,
    De: float,
    cutout: int = 0,
) -> pd.Series:
    """
    Port của hàm Pine Script:

        eq(starting_equity, sig, R) =>
            cuttime = time[cutout]
            if not na(cuttime) and time >= cuttime
                r = R * e(La)
                d = 1 - De
                var float a = 0.0
                if (sig[1] > 0)
                    a := r
                else if (sig[1] < 0)
                    a := -r
                var float e = na
                if na(e[1])
                    e := starting_equity
                else
                    e := (e[1] * d) * (1 + a)
                if (e < 0.25)
                    e := 0.25
                e
            else
                na

    Ở Python, hàm này trả về toàn bộ equity theo thời gian dưới dạng Series.
    """
    if sig is None or R is None or len(sig) == 0:
        return pd.Series(dtype="float64")

    index = sig.index
    # R nhân với e(L) (growth factor)
    growth = exp_growth(L=L, index=index, cutout=cutout)
    r = R * growth
    d = 1.0 - De

    e_values = []
    prev_e: Optional[float] = None

    # Duyệt tuần tự để bám sát logic biến trạng thái trong Pine
    for i, (t, r_i, sig_prev) in enumerate(zip(index, r, sig.shift(1))):
        if i < cutout:
            e_values.append(pd.NA)
            continue

        if pd.isna(sig_prev) or sig_prev == 0:
            a = 0.0
        elif sig_prev > 0:
            a = float(r_i)
        else:  # sig_prev < 0
            a = -float(r_i)

        if prev_e is None:
            e_curr = float(starting_equity)
        else:
            e_curr = (prev_e * d) * (1.0 + a)

        if e_curr < 0.25:
            e_curr = 0.25

        prev_e = e_curr
        e_values.append(e_curr)

    equity = pd.Series(e_values, index=index, dtype="float64")
    return equity


def weighted_signal(
    signals: Iterable[pd.Series],
    weights: Iterable[pd.Series],
) -> pd.Series:
    """
    Port của hàm Pine Script:

        Signal(m1, w1, ..., m9, w9) =>
            n = Σ (mi * wi)
            d = Σ wi
            sig = math.round(n/d, 2)
            sig

    Ở Python, nhận list/iterable 9 series `signals` và 9 series `weights`.
    """
    signals = list(signals)
    weights = list(weights)
    if len(signals) != len(weights):
        raise ValueError("signals và weights phải có cùng độ dài")

    if not signals:
        return pd.Series(dtype="float64")

    num = None
    den = None
    for m, w in zip(signals, weights):
        term = m * w
        num = term if num is None else num + term
        den = w if den is None else den + w

    sig = num / den
    return sig.round(2)


def cut_signal(x: pd.Series, threshold: float = 0.49) -> pd.Series:
    """
    Port của hàm Pine Script:

        Cut(x) =>
            c = x > 0.49 ? 1 : x < -0.49 ? -1 : 0
            c
    """
    c = pd.Series(0, index=x.index, dtype="int8")
    c[x > threshold] = 1
    c[x < -threshold] = -1
    return c


def trend_sign(signal: pd.Series, *, strategy: bool = False) -> pd.Series:
    """
    Phiên bản số hóa (không màu) của hàm Pine Script:

        trendcol(signal) =>
            c = strategy ? (signal[1] > 0 ? colup : coldw)
                         : (signal > 0) ? colup : coldw

    Ở đây:
        - trả về +1 nếu bull, -1 nếu bear, 0 nếu neutral.
        - nếu strategy=True thì dùng signal[1] như trong Pine.
    """
    base = signal.shift(1) if strategy else signal
    result = pd.Series(0, index=signal.index, dtype="int8")
    result[base > 0] = 1
    result[base < 0] = -1
    return result


def rate_of_change(prices: pd.Series) -> pd.Series:
    """
    Tương đương biến toàn cục trong Pine:

        R = (close - close[1]) / close[1]
    """
    return prices.pct_change()


def _layer1_signal_for_ma(
    prices: pd.Series,
    ma_tuple,
    *,
    L: float,
    De: float,
    cutout: int = 0,
):
    """
    Port cụm logic:

        E   = eq(1, signal(MA),   R), sE   = signal(MA)
        E1  = eq(1, signal(MA1),  R), sE1  = signal(MA1)
        ...
        EMA_Signal = Signal(sE, E, sE1, E1, ..., sE_4, E_4)
    """
    (
        MA,
        MA1,
        MA2,
        MA3,
        MA4,
        MA_1,
        MA_2,
        MA_3,
        MA_4,
    ) = ma_tuple

    R = rate_of_change(prices)

    s = generate_signal_from_ma(prices, MA)
    s1 = generate_signal_from_ma(prices, MA1)
    s2 = generate_signal_from_ma(prices, MA2)
    s3 = generate_signal_from_ma(prices, MA3)
    s4 = generate_signal_from_ma(prices, MA4)
    s_1 = generate_signal_from_ma(prices, MA_1)
    s_2 = generate_signal_from_ma(prices, MA_2)
    s_3 = generate_signal_from_ma(prices, MA_3)
    s_4 = generate_signal_from_ma(prices, MA_4)

    E = equity_series(1.0, s, R, L=L, De=De, cutout=cutout)
    E1 = equity_series(1.0, s1, R, L=L, De=De, cutout=cutout)
    E2 = equity_series(1.0, s2, R, L=L, De=De, cutout=cutout)
    E3 = equity_series(1.0, s3, R, L=L, De=De, cutout=cutout)
    E4 = equity_series(1.0, s4, R, L=L, De=De, cutout=cutout)
    E_1 = equity_series(1.0, s_1, R, L=L, De=De, cutout=cutout)
    E_2 = equity_series(1.0, s_2, R, L=L, De=De, cutout=cutout)
    E_3 = equity_series(1.0, s_3, R, L=L, De=De, cutout=cutout)
    E_4 = equity_series(1.0, s_4, R, L=L, De=De, cutout=cutout)

    signal_series = weighted_signal(
        signals=[s, s1, s2, s3, s4, s_1, s_2, s_3, s_4],
        weights=[E, E1, E2, E3, E4, E_1, E_2, E_3, E_4],
    )

    return (
        signal_series,
        (s, s1, s2, s3, s4, s_1, s_2, s_3, s_4),
        (E, E1, E2, E3, E4, E_1, E_2, E_3, E_4),
    )


def compute_atc_signals(
    prices: pd.Series,
    src: Optional[pd.Series] = None,
    *,
    ema_len: int = 28,
    hull_len: int = 28,
    wma_len: int = 28,
    dema_len: int = 28,
    lsma_len: int = 28,
    kama_len: int = 28,
    ema_w: float = 1.0,
    hma_w: float = 1.0,
    wma_w: float = 1.0,
    dema_w: float = 1.0,
    lsma_w: float = 1.0,
    kama_w: float = 1.0,
    robustness: str = "Medium",
    La: float = 0.02,
    De: float = 0.03,
    cutout: int = 0,
):
    """
    Port khối code chính trong Pine:

        - Khai báo EMA/HMA/WMA/DEMA/LSMA/KAMA với nhiều length
        - Tính Layer 1: EMA_Signal, HMA_Signal, ...
        - Tính Layer 2: EMA_S, HMA_S, ..., KAMA_S
        - FINAL: Average_Signal

    Args:
        prices: Series giá (tương đương `close` trong Pine) để tính R và signal.
        src: Series nguồn để tính MA (tương đương input `src` trong Pine).
        *_len: Độ dài các MA.
        *_w: Trọng số khởi tạo cho từng họ MA ở Layer 2.
        robustness: "Narrow" / "Medium" / "Wide".
        La: Lambda dùng trong exp growth (đã scale sẵn, khác với input Pine /1000).
        De: Decay rate (đã scale sẵn, khác với input Pine /100).
        cutout: Số bar bỏ qua đầu chuỗi.

    Returns:
        dict chứa:
            - EMA_Signal, HMA_Signal, ..., KAMA_Signal
            - EMA_S, HMA_S, ..., KAMA_S
            - Average_Signal
    """
    if src is None:
        src = prices

    # DECLARE MOVING AVERAGES (SetOfMovingAverages)
    EMA = set_of_moving_averages(ema_len, src, "EMA", robustness=robustness)
    HMA = set_of_moving_averages(hull_len, src, "HMA", robustness=robustness)
    WMA = set_of_moving_averages(wma_len, src, "WMA", robustness=robustness)
    DEMA = set_of_moving_averages(dema_len, src, "DEMA", robustness=robustness)
    LSMA = set_of_moving_averages(lsma_len, src, "LSMA", robustness=robustness)
    KAMA = set_of_moving_averages(kama_len, src, "KAMA", robustness=robustness)

    # MAIN CALCULATIONS - Adaptability Layer 1
    EMA_Signal, _, _ = _layer1_signal_for_ma(prices, EMA, L=La, De=De, cutout=cutout)
    HMA_Signal, _, _ = _layer1_signal_for_ma(prices, HMA, L=La, De=De, cutout=cutout)
    WMA_Signal, _, _ = _layer1_signal_for_ma(prices, WMA, L=La, De=De, cutout=cutout)
    DEMA_Signal, _, _ = _layer1_signal_for_ma(prices, DEMA, L=La, De=De, cutout=cutout)
    LSMA_Signal, _, _ = _layer1_signal_for_ma(prices, LSMA, L=La, De=De, cutout=cutout)
    KAMA_Signal, _, _ = _layer1_signal_for_ma(prices, KAMA, L=La, De=De, cutout=cutout)

    # Adaptability Layer 2
    R = rate_of_change(prices)
    EMA_S = equity_series(ema_w, EMA_Signal, R, L=La, De=De, cutout=cutout)
    HMA_S = equity_series(hma_w, HMA_Signal, R, L=La, De=De, cutout=cutout)
    WMA_S = equity_series(wma_w, WMA_Signal, R, L=La, De=De, cutout=cutout)
    DEMA_S = equity_series(dema_w, DEMA_Signal, R, L=La, De=De, cutout=cutout)
    LSMA_S = equity_series(lsma_w, LSMA_Signal, R, L=La, De=De, cutout=cutout)
    KAMA_S = equity_series(kama_w, KAMA_Signal, R, L=La, De=De, cutout=cutout)

    # FINAL CALCULATIONS
    nom = (
        cut_signal(EMA_Signal) * EMA_S
        + cut_signal(HMA_Signal) * HMA_S
        + cut_signal(WMA_Signal) * WMA_S
        + cut_signal(DEMA_Signal) * DEMA_S
        + cut_signal(LSMA_Signal) * LSMA_S
        + cut_signal(KAMA_Signal) * KAMA_S
    )
    den = EMA_S + HMA_S + WMA_S + DEMA_S + LSMA_S + KAMA_S
    Average_Signal = nom / den

    return {
        "EMA_Signal": EMA_Signal,
        "HMA_Signal": HMA_Signal,
        "WMA_Signal": WMA_Signal,
        "DEMA_Signal": DEMA_Signal,
        "LSMA_Signal": LSMA_Signal,
        "KAMA_Signal": KAMA_Signal,
        "EMA_S": EMA_S,
        "HMA_S": HMA_S,
        "WMA_S": WMA_S,
        "DEMA_S": DEMA_S,
        "LSMA_S": LSMA_S,
        "KAMA_S": KAMA_S,
        "Average_Signal": Average_Signal,
    }


# ---------------------------------------------------------------------------
# Các tiện ích tương ứng CALIBRATION MODE / VISUALIZATION / ALERTS / TABLES
# ---------------------------------------------------------------------------

def select_calibration_series(
    calfor: str,
    ma_map: dict,
    sig_map: dict,
):
    """
    Port logic CALIBRATION MODE trong Pine:

        if calibrate and calfor == "EMA"
            cal  :=  EMA      ; scal  := sE
            cal1 := EMA1      ; scal1 := sE1
            ...

    Args:
        calfor: "EMA" | "HMA" | "WMA" | "DEMA" | "LSMA" | "KAMA".
        ma_map: dict tên → tuple 9 MA: (MA, MA1, MA2, MA3, MA4, MA_1, MA_2, MA_3, MA_4).
        sig_map: dict tên → tuple 9 signal: (s, s1, s2, s3, s4, s_1, s_2, s_3, s_4).

    Returns:
        dict với các key:
            - cal, scal
            - cal1..cal4, cal_m1..cal_m4
            - scal1..scal4, scal_m1..scal_m4
        Nếu calfor không hợp lệ → tất cả None.
    """
    calfor = (calfor or "").upper()
    if calfor not in ma_map or calfor not in sig_map:
        return {k: None for k in [
            "cal",
            "cal1",
            "cal2",
            "cal3",
            "cal4",
            "cal_m1",
            "cal_m2",
            "cal_m3",
            "cal_m4",
            "scal",
            "scal1",
            "scal2",
            "scal3",
            "scal4",
            "scal_m1",
            "scal_m2",
            "scal_m3",
            "scal_m4",
        ]}

    MA, MA1, MA2, MA3, MA4, MA_1, MA_2, MA_3, MA_4 = ma_map[calfor]
    s, s1, s2, s3, s4, s_1, s_2, s_3, s_4 = sig_map[calfor]

    return {
        "cal": MA,
        "cal1": MA1,
        "cal2": MA2,
        "cal3": MA3,
        "cal4": MA4,
        "cal_m1": MA_1,
        "cal_m2": MA_2,
        "cal_m3": MA_3,
        "cal_m4": MA_4,
        "scal": s,
        "scal1": s1,
        "scal2": s2,
        "scal3": s3,
        "scal4": s4,
        "scal_m1": s_1,
        "scal_m2": s_2,
        "scal_m3": s_3,
        "scal_m4": s_4,
    }


def select_cols_for_visualization(
    colbase: str,
    *,
    average_signal: pd.Series,
    ema_signal: pd.Series,
    hma_signal: pd.Series,
    wma_signal: pd.Series,
    dema_signal: pd.Series,
    lsma_signal: pd.Series,
    kama_signal: pd.Series,
) -> pd.Series:
    """
    Port phần chọn `cols` trong block Visualization:

        if colbase == "Average"
            cols := Average_Signal
        else if colbase == "EMA"
            cols := EMA_Signal
        ...
    """
    base = (colbase or "Average").upper()
    if base == "AVERAGE":
        return average_signal
    if base == "EMA":
        return ema_signal
    if base == "HMA":
        return hma_signal
    if base == "WMA":
        return wma_signal
    if base == "DEMA":
        return dema_signal
    if base == "LSMA":
        return lsma_signal
    if base == "KAMA":
        return kama_signal
    return average_signal


def compute_direction_series(
    cols: pd.Series,
    *,
    long_threshold: float,
    short_threshold: float,
    strategy: bool = False,
) -> pd.Series:
    """
    Port logic:

        var int direction = na
        if ta.crossover(strategy ? cols[1] : cols, Long_threshold)
            direction :=  1
        if ta.crossunder(strategy ? cols[1] : cols, Short_threshold)
            direction := -1
    """
    base = cols.shift(1) if strategy else cols
    dir_series = pd.Series(pd.NA, index=cols.index, dtype="float")

    up = crossover(base, pd.Series(long_threshold, index=cols.index))
    down = crossunder(base, pd.Series(short_threshold, index=cols.index))

    dir_series[up] = 1.0
    dir_series[down] = -1.0
    return dir_series


def select_alert_base(
    alertbase: str,
    *,
    average_signal: pd.Series,
    ema_signal: pd.Series,
    hma_signal: pd.Series,
    wma_signal: pd.Series,
    dema_signal: pd.Series,
    lsma_signal: pd.Series,
    kama_signal: pd.Series,
) -> pd.Series:
    """
    Port phần chọn `alertb` trong Alerts Code.
    """
    base = (alertbase or "Average").upper()
    if base == "AVERAGE":
        return average_signal
    if base == "EMA":
        return ema_signal
    if base == "HMA":
        return hma_signal
    if base == "WMA":
        return wma_signal
    if base == "DEMA":
        return dema_signal
    if base == "LSMA":
        return lsma_signal
    if base == "KAMA":
        return kama_signal
    return average_signal


def compute_alert_series(
    alertb: pd.Series,
    direction: pd.Series,
    *,
    long_threshold: float,
    short_threshold: float,
) -> Tuple[pd.Series, pd.Series]:
    """
    Port logic:

        long  = ta.crossover(alertb,  Long_threshold) and direction[1] == -1
        short = ta.crossunder(alertb, Short_threshold) and direction[1] ==  1
    """
    long_cross = crossover(alertb, pd.Series(long_threshold, index=alertb.index))
    short_cross = crossunder(alertb, pd.Series(short_threshold, index=alertb.index))

    prev_dir = direction.shift(1)
    long_sig = long_cross & (prev_dir == -1)
    short_sig = short_cross & (prev_dir == 1)
    return long_sig, short_sig


def compute_exp_multiplier_and_warnings(
    index: pd.Index,
    *,
    La: float,
    cutout: int = 0,
) -> Tuple[float, str, str, str]:
    """
    Port logic bảng Table 2:

        warning1 = e(La) > 1.25 ? "☠️"  : e(La) > 1.1 ? "❗" : ""
        warning2 = ...
        warning3 = ...

    Trả về:
        (exp_mult, warning1, warning2, warning3)
    """
    e_series = exp_growth(L=La, index=index, cutout=cutout)
    exp_mult = float(e_series.iloc[-1]) if len(e_series) else 1.0

    if exp_mult > 1.25:
        warning1 = "☠️"
        warning2 = "Exp Mult. is way too high, reduce λ"
        warning3 = "(Lambda) or increase CutOut Period,"
    elif exp_mult > 1.1:
        warning1 = "❗"
        warning2 = "Exp Mult. is too high, reduce λ"
        warning3 = "(Lambda) or increase CutOut Period"
    else:
        warning1 = ""
        warning2 = ""
        warning3 = ""

    return exp_mult, warning1, warning2, warning3


__all__ = [
    "diflen",
    "calculate_kama_atc",
    "ma_calculation",
    "set_of_moving_averages",
    "exp_growth",
    "crossover",
    "crossunder",
    "generate_signal_from_ma",
    "equity_series",
    "weighted_signal",
    "cut_signal",
    "trend_sign",
    "rate_of_change",
    "compute_atc_signals",
]

