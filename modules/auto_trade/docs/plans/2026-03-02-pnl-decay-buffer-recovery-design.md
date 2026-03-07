# PnL Decay Buffer Recovery — Design Document

**Date**: 2026-03-02  
**Status**: Validated  

---

## 1. Mục tiêu

Cải thiện cơ chế khởi động Recovery bằng cách thay thế trigger **per-order** (nhìn vào 1 lệnh đơn lẻ vừa lỗ) bằng trigger **decay buffer** — tính Weighted Net PnL của chuỗi N lệnh gần nhất với trọng số giảm dần (lệnh cũ hơn ảnh hưởng ít hơn). Recovery chỉ khởi động khi portfolio thực sự đang lỗ ròng, không kích hoạt nhầm sau một lệnh lỗ đơn lẻ giữa chuỗi lãi.

### Vấn đề hiện tại

`RecoveryManager` hiện tại trigger recovery ngay khi **1 lệnh đóng với PnL < 0**:

```
Order đóng PnL < 0 → _start_new_recovery(loss_from_THIS_order)
```

Điều này dẫn tới:
- Trigger sai: 3 lệnh lãi $30 → 1 lệnh lỗ $15 → recovery khởi động dù portfolio đang lãi ròng $15.
- `initial_loss` bị underestimate hoặc overestimate so với tình trạng thực tế của portfolio.

### Giải pháp

Thêm `PnlDecayBuffer` nằm giữa EventBus và recovery trigger. Buffer giữ N lệnh gần nhất, tính Weighted Net PnL với **trade-count decay** (lệnh cũ hơn có trọng số thấp hơn). Chỉ trigger recovery khi `weighted_net_pnl < -threshold_usdt`.

---

## 2. Kiến trúc tổng thể

```
POSITION_CLOSED event
        ↓
RecoveryManager._on_position_closed(event)
        ↓
PnlDecayBuffer.push(pnl)
        ↓
weighted_net_pnl = Σ(pnl_i × α^i) / Σ(α^i)
        ↓
weighted_net_pnl < -threshold_usdt  AND  trades >= min_trades?
   ├── YES → _start_new_recovery(abs(weighted_net_pnl))
   │          + decay_buffer.reset()
   └── NO  → cập nhật buffer, không làm gì thêm
```

### So sánh với cơ chế cũ

| | Hiện tại | Mới |
|--|---------|-----|
| Trigger source | PnL của 1 lệnh vừa đóng | Weighted Net PnL của N lệnh gần nhất |
| `initial_loss` | Loss của 1 lệnh | `abs(weighted_net_pnl)` khi trigger |
| Lãi trước đó | Không tính | Được tính vào buffer, giảm/triệt tiêu trigger |
| False trigger | Cao | Thấp — cần chuỗi lỗ ròng mới trigger |
| Backward-compatible | — | ✅ Nếu không config buffer → hành vi cũ 100% |

---

## 3. Công thức Decay

### Trade-count Decay

Với N lệnh trong buffer (index 0 = mới nhất):

$$w_i = \alpha^i \quad \text{với } i = 0, 1, 2, \ldots, N-1$$

$$\text{weighted\_net\_pnl} = \frac{\sum_{i=0}^{N-1} pnl_i \times w_i}{\sum_{i=0}^{N-1} w_i}$$

### Ví dụ với `alpha=0.7`, `window=5`, `threshold=$15`

| Lệnh | PnL | Weight (α^i) | Contribution |
|------|-----|-------------|-------------|
| Mới nhất (i=0) | -$30 | 1.000 | -$30.00 |
| i=1 | +$20 | 0.700 | +$14.00 |
| i=2 | -$10 | 0.490 | -$4.90 |
| i=3 | +$15 | 0.343 | +$5.15 |
| i=4 | -$5  | 0.240 | -$1.20 |
| **Tổng** | | **Σw = 2.773** | **-$16.95** |

→ Weighted Net = -$16.95 / 2.773 = **-$6.11** → Không trigger (< threshold $15).

### Tuning guide

| alpha | window | Đặc tính |
|-------|--------|---------|
| 0.8 | 8 | Nhớ dài, nhạy vừa |
| 0.7 | 5 | Cân bằng (recommended) |
| 0.5 | 5 | Nhớ ngắn, phản ứng nhanh với chuỗi lỗ liên tiếp |
| 1.0 | N | Không decay — simple average |

---

## 4. Config Parameters

```python
class PnlDecayBufferConfig(TypedDict, total=False):
    enabled: bool            # True = dùng decay buffer; False = hành vi cũ (per-order)
    window_size: int         # Số lệnh tối đa giữ trong buffer (default: 10)
    decay_alpha: float       # Hệ số decay 0 < α ≤ 1.0 (default: 0.7)
    trigger_threshold: float # Ngưỡng USDT tuyệt đối để trigger recovery (default: 20.0)
    min_trades_required: int # Cần ít nhất N lệnh trong buffer mới xét (default: 2)
```

**Ví dụ config trong `settings.yaml`:**

```yaml
recovery:
  enabled: true
  pnl_decay_buffer:
    enabled: true
    window_size: 10
    decay_alpha: 0.7
    trigger_threshold: 20.0
    min_trades_required: 2
  # ... các config recovery hiện có (không đổi)
  target_profit_per_trade: 5.0
  max_recovery_trades: 20
```

---

## 5. Implementation

### File mới: `modules/auto_trade/strategies/pnl_decay_buffer.py`

```python
from collections import deque
from dataclasses import dataclass
from typing import Deque, List

from modules.common.ui.logging import log_debug, log_info


@dataclass
class DecayBufferState:
    trades_in_buffer: int
    weighted_net_pnl: float
    raw_net_pnl: float          # Không decay, để debug/compare
    oldest_pnl: float
    newest_pnl: float
    should_trigger: bool


class PnlDecayBuffer:
    """
    Trade-count decay buffer cho PnL.

    Giữ N lệnh gần nhất. Lệnh mới nhất có trọng số 1.0,
    lệnh cũ hơn có trọng số α^i (i = vị trí từ mới đến cũ).

    Trigger condition: weighted_net_pnl < -threshold AND trades >= min_trades
    """

    def __init__(
        self,
        window_size: int = 10,
        decay_alpha: float = 0.7,
        trigger_threshold: float = 20.0,
        min_trades_required: int = 2,
    ) -> None:
        self._buffer: Deque[float] = deque(maxlen=window_size)
        self._alpha = decay_alpha
        self._threshold = trigger_threshold
        self._min_trades = min_trades_required

    def push(self, pnl: float) -> DecayBufferState:
        """Thêm PnL mới vào buffer và trả về trạng thái hiện tại."""
        self._buffer.appendleft(pnl)  # index 0 = mới nhất
        return self.evaluate()

    def evaluate(self) -> DecayBufferState:
        """Tính weighted net PnL hiện tại."""
        trades: List[float] = list(self._buffer)
        n = len(trades)

        weights = [self._alpha ** i for i in range(n)]
        weighted_sum = sum(p * w for p, w in zip(trades, weights))
        weight_total = sum(weights)

        weighted_net = weighted_sum / weight_total if weight_total > 0 else 0.0
        raw_net = sum(trades)

        state = DecayBufferState(
            trades_in_buffer=n,
            weighted_net_pnl=weighted_net,
            raw_net_pnl=raw_net,
            oldest_pnl=trades[-1] if trades else 0.0,
            newest_pnl=trades[0] if trades else 0.0,
            should_trigger=(
                n >= self._min_trades
                and weighted_net < -self._threshold
            ),
        )

        log_debug(
            f"[PnlDecayBuffer] trades={n}, weighted_net=${weighted_net:.2f}, "
            f"raw_net=${raw_net:.2f}, trigger={state.should_trigger}"
        )

        return state

    def reset(self) -> None:
        """Reset buffer — gọi sau khi trigger recovery hoặc recovery complete."""
        self._buffer.clear()
        log_info("[PnlDecayBuffer] Buffer reset")
```

### Thay đổi `recovery_manager.py`

**Trong `__init__`** — khởi tạo buffer optional:

```python
from modules.auto_trade.strategies.pnl_decay_buffer import PnlDecayBuffer

# Sau các init hiện có:
decay_cfg = self.config.get("pnl_decay_buffer", {})
if decay_cfg and decay_cfg.get("enabled", False):
    self._decay_buffer: Optional[PnlDecayBuffer] = PnlDecayBuffer(
        window_size=int(decay_cfg.get("window_size", 10)),
        decay_alpha=float(decay_cfg.get("decay_alpha", 0.7)),
        trigger_threshold=float(decay_cfg.get("trigger_threshold", 20.0)),
        min_trades_required=int(decay_cfg.get("min_trades_required", 2)),
    )
    log_info("[RecoveryManager] PnlDecayBuffer enabled")
else:
    self._decay_buffer = None
```

**Trong `_on_position_closed`** — push cả profit lẫn loss vào buffer:

```python
if pnl >= 0:
    self._handle_profit(pnl)
    if self._decay_buffer:
        self._decay_buffer.push(pnl)  # profit làm giảm weighted_net
else:
    self._handle_loss_v2(abs(pnl), pnl)
```

**Method mới `_handle_loss_v2`** — thay thế `_handle_loss` khi buffer enabled:

```python
def _handle_loss_v2(self, loss: float, raw_pnl: float) -> None:
    if not self._enabled:
        return

    # Fallback: buffer không config → hành vi cũ
    if self._decay_buffer is None:
        self._handle_loss(loss)
        return

    state = self._decay_buffer.push(raw_pnl)  # push PnL âm

    log_info(
        f"[DecayBuffer] trades={state.trades_in_buffer}, "
        f"weighted_net=${state.weighted_net_pnl:.2f}, "
        f"raw_net=${state.raw_net_pnl:.2f}, trigger={state.should_trigger}"
    )

    if not self._strategy and state.should_trigger:
        # Trigger recovery với abs(weighted_net_pnl) làm initial_loss
        self._start_new_recovery(abs(state.weighted_net_pnl))
        self._decay_buffer.reset()
    elif self._strategy:
        # Recovery đang chạy → record loss per-order như cũ
        self._strategy.record_loss(loss)
        self._persist_state()
```

**Reset buffer khi recovery complete** — trong `_handle_profit`:

```python
if state.is_complete:
    log_info("Recovery COMPLETE! All losses recovered.")
    self._mark_recovery_complete()
    if self._decay_buffer:
        self._decay_buffer.reset()  # Bắt đầu đếm lại từ đầu
```

---

## 6. Testing Strategy

### Test cases quan trọng

| Scenario | Buffer (newest→oldest) | Weighted Net (α=0.7, threshold=$15) | Expected |
|----------|----------------------|-------------------------------------|----------|
| 3 lệnh lỗ liên tiếp | [-$20, -$15, -$10] | ~-$16.2 | TRIGGER |
| Lỗ sau chuỗi lãi | [-$20, +$30, +$15] | ~-$2.5 | KHÔNG trigger |
| 1 lệnh lỗ lớn, chưa đủ min_trades | [-$50] | -$50 | KHÔNG trigger (min_trades=2) |
| Lỗ nhỏ xen kẽ lãi | [-$5, +$2, -$5, +$2] | ~-$2.1 | KHÔNG trigger |
| Recovery đang chạy + lệnh lỗ mới | Recovery active | N/A | record_loss per-order |
| Buffer đầy → lệnh cũ bị đẩy ra | 11 lệnh push (window=10) | recalculate | deque maxlen tự trim |
| alpha=1.0 | Bất kỳ | Simple average | Hoạt động đúng |
| Không có config buffer | None | N/A | Fallback hành vi cũ |

### Files

```
modules/auto_trade/strategies/
├── pnl_decay_buffer.py                     # [NEW]
└── recovery_manager.py                     # [EDIT]

tests/modules/auto_trade/strategies/
└── test_pnl_decay_buffer.py                # [NEW]
    # - test công thức decay (weighted sum đúng với alpha=0.7)
    # - test trigger condition (đúng threshold)
    # - test min_trades_required
    # - test reset sau trigger
    # - test fallback khi buffer=None
```

---

## 7. Edge Cases & Mitigations

| Edge case | Mitigation |
|-----------|-----------|
| `pnl_decay_buffer` không có trong config | `_decay_buffer = None` → fallback hành vi cũ 100% |
| `alpha=1.0` | Tương đương simple average, hoạt động đúng |
| `min_trades_required=1` | Trigger ngay lệnh đầu tiên — user chủ động chọn |
| Recovery complete | Reset buffer sau khi complete, bắt đầu đếm lại |
| API / EventBus lỗi không fire event | Không ảnh hưởng — buffer chỉ nhận dữ liệu từ event |

---

## 8. Out of Scope (YAGNI)

- ❌ **Time-based decay** — trade-count decay đủ cho trading không đều; thêm time dimension phức tạp hóa không cần thiết.
- ❌ **Per-symbol buffer** — GLOBAL scope nhất quán với `RecoveryManager` hiện tại.
- ❌ **Persistent buffer across restart** — buffer reset khi restart là hành vi an toàn hơn.

---

## 9. Next Steps

1. Tạo `pnl_decay_buffer.py` với class `PnlDecayBuffer` + `DecayBufferState`.
2. Viết `test_pnl_decay_buffer.py` — unit test công thức và trigger logic.
3. Edit `recovery_manager.py` — thêm buffer init, `_handle_loss_v2`, reset on complete.
4. Thêm `pnl_decay_buffer` section vào `settings.yaml` với `enabled: false` (opt-in).
5. Test end-to-end với dry_run mode, quan sát log `[DecayBuffer]`.
