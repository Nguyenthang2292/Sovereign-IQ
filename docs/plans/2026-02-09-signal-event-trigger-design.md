# Signal Event Trigger — Design

**Date:** 2026-02-09  
**Status:** Validated  
**Summary:** Add event-driven signal notification from Scanner to AutoTrade. When a fresh signal is generated, Scanner publishes `SIGNAL_GENERATED` event → AutoTrade receives and triggers immediate execution check (bypassing 60s polling timer). Includes overlap guard to prevent concurrent execution.

---

## 1. Goal and scope

**Goals**
- Reduce signal-to-execution latency from max 60s (polling) to <1s (event-driven)
- Increase reliability: ensure signals are not missed due to polling gaps
- Maintain backward compatibility: timer 60s still runs as fallback

**In scope**
- **Scanner** (`gui/main_window/scanner.py`): publish `SIGNAL_GENERATED` event after saving signal to DB
- **AutoTrade** (`gui/main_window/auto_trade.py`): subscribe to event, spawn thread to run `_auto_trade_cycle()` immediately, add overlap guard
- Use existing `EventSystem` from `main_window.event_bus`

**Out of scope**
- No changes to timer intervals (scanner 5min, auto-trade 60s)
- No changes to open position gate logic
- No changes to risk checks or execution logic

---

## 2. Architecture — Event flow

**Current flow (polling only):**
```
Scanner (every 5min) → Generate Signal → Save DB
                                            ↓
AutoTrade (every 60s) → Poll DB → Check fresh signals → Execute
```

**New flow (push + polling):**
```
Scanner (every 5min) → Generate Signal → Save DB → Publish SIGNAL_GENERATED event
                                                           ↓
                                    AutoTrade subscribes → Receive event → Spawn thread → _auto_trade_cycle()
                                            ↓
AutoTrade (every 60s) → Poll DB (fallback if event missed)
```

**Shared EventBus:**
- Use `self.parent.event_bus` (EventSystem instance in `main_window.py`)
- Scanner: `self.parent.event_bus.publish(EventType.SIGNAL_GENERATED, data)`
- AutoTrade: `self.parent.event_bus.subscribe(EventType.SIGNAL_GENERATED, callback)`

**Thread model:**
- Event callback runs on EventBus thread → must spawn background thread (don't block event bus)
- Background thread runs `_auto_trade_cycle()` (same as timer-triggered cycle)
- Guard with `_trading_running` flag prevents overlap between event-triggered and timer-triggered cycles

---

## 3. Scanner side — Publish event

**Where:**
In `scanner.py`, method `_save_signal_to_gui_db` (line 404-430), after successful `save_signal` and log.

**Code:**
```python
logger.info("Signal saved to GUI database")

# Notify AutoTrade immediately via event
try:
    from modules.auto_trade.monitoring.event_system import EventType
    self.parent.event_bus.publish(
        EventType.SIGNAL_GENERATED,
        {
            "symbol": signal.symbol,
            "signal_type": signal.signal_type,
            "score": signal.score,
            "correlation_id": correlation_id,
        },
        source="scanner"
    )
    logger.info(f"Published SIGNAL_GENERATED event for {signal.symbol}")
except Exception as e:
    logger.warning(f"Failed to publish signal event: {e}")
    # Don't fail the scan if event publish fails
```

**Import:**
- Lazy import `EventType` inside the try block (avoid circular import if any)

**Error handling:**
- Wrap in try/except → if event bus unavailable or publish fails, log warning but continue
- Signal is already saved to DB, so timer 60s will pick it up as fallback

---

## 4. AutoTrade side — Subscribe and guard

**Subscribe in `__init__`:**
In `auto_trade.py`, after creating updaters (line 52-55):

```python
# Subscribe to signal events for immediate execution
from modules.auto_trade.monitoring.event_system import EventType
self.parent.event_bus.subscribe(EventType.SIGNAL_GENERATED, self._on_signal_event)
logger.info("AutoTrade subscribed to SIGNAL_GENERATED events")
```

**Add guard flag in `__init__`:**
```python
self._trading_running = False
self._trading_lock = threading.Lock()
```

**Event handler (new method):**
```python
def _on_signal_event(self, event):
    """Handle SIGNAL_GENERATED event from scanner."""
    import threading
    
    symbol = event.data.get("symbol", "unknown")
    print(f"Signal event received: {symbol}, triggering immediate check")
    
    # Check guard: skip if already running
    with self._trading_lock:
        if self._trading_running:
            print("Auto-trade cycle already running, event skipped")
            return
    
    # Spawn background thread to run cycle
    def run_cycle():
        self._auto_trade_cycle()
    
    thread = threading.Thread(target=run_cycle, daemon=True, name="AutoTradeEvent")
    thread.start()
```

**Guard in `_auto_trade_cycle`:**
At the very top of `_auto_trade_cycle` (before line 162):

```python
# Guard: prevent concurrent execution (event + timer)
with self._trading_lock:
    if self._trading_running:
        print("Auto-trade cycle already running, skipping...")
        return
    self._trading_running = True

try:
    # Existing logic...
finally:
    with self._trading_lock:
        self._trading_running = False
```

**Unsubscribe in `stop()`:**
```python
# Unsubscribe from events
from modules.auto_trade.monitoring.event_system import EventType
self.parent.event_bus.unsubscribe(EventType.SIGNAL_GENERATED, self._on_signal_event)
```

**Unchanged:**
- Open position gate (line 162-177) still works
- Risk checks still work
- Timer 60s still runs independently

---

## 5. Verification

**Unit tests:**
1. **`test_signal_event_triggers_cycle.py`**: Mock event bus, publish `SIGNAL_GENERATED` → verify `_auto_trade_cycle` called
2. **`test_auto_trade_overlap_guard.py`**: Publish 2 events concurrently → only 1 cycle runs, other skipped
3. **`test_event_and_timer_concurrent.py`**: Event triggers cycle while timer also fires → guard prevents overlap

**Integration test:**
- Run GUI with auto-scan enabled
- Wait for signal generation
- Check log: "Signal event received" appears immediately after "Signal saved to GUI database"
- Check log: "Auto-trade selected" appears <1s after signal (not 60s)

**Manual test:**
- Enable auto-scan and auto-trade
- Wait for signal
- Verify execution happens immediately (not waiting for next 60s timer)

**Edge cases:**
- Event fires while cycle from timer is running → skip, log "already running"
- Event bus unavailable → scanner logs warning, timer 60s picks up signal
- Multiple signals in quick succession → each triggers event, guard serializes execution

---

## Summary

Two files changed:
- `scanner.py`: publish event after save
- `auto_trade.py`: subscribe to event, add guard, spawn thread on event

Zero changes to:
- Timers (scanner 5min, auto-trade 60s)
- Business logic (gates, risk checks, execution)
- Database schema or queries

Benefit: Signal → execution latency drops from max 60s to <1s while maintaining all safety checks.
