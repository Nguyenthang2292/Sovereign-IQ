# Qwen Vision Fallback — Design Document

**Date:** 2026-03-05  
**Status:** Approved, ready for implementation  
**Scope:** `gemini_chart_analyzer`, `gemini_gann_square`, `auto_trade`

---

## 1. Problem Statement

Both `gemini_chart_analyzer` and `gemini_gann_square` modules call **Google Gemini** as
the sole vision AI provider. When Gemini encounters quota exhaustion (429), rate limits,
or outages, the entire chart analysis pipeline fails with no recovery path.

**Goal:** Add **Alibaba Qwen VL** (via DashScope API) as a secondary vision provider that
activates automatically when all Gemini models are exhausted.

---

## 2. Design Decision: Provider Chain Architecture

Three approaches were considered:

| Option | Description | Decision |
|--------|-------------|----------|
| A | Add Qwen models inside `_call_model_with_retries()` | Rejected — mixes two different APIs in one class |
| **B** | **New `VisionAnalyzerChain` orchestrating Gemini → Qwen** | **Selected** |
| C | try/except wrap at `GannSignalEngine` only | Rejected — not reusable for `GeminiIntegration` |

**Rationale for B:** Clean provider abstraction enables adding future providers
(Claude Vision, GPT-4o, etc.) without touching existing code. Both `GannSignalEngine`
and `GeminiIntegration` share the same chain with zero duplication.

---

## 3. Architecture Overview

```
                ┌──────────────────────────────────┐
                │        VisionAnalyzerChain       │
                │     (shared orchestrator)        │
                └─────────────┬────────────────────┘
                              │ tries providers in order
              ┌───────────────▼───────────────┐
              │    1. GeminiVisionProvider    │
              │    (wraps GeminiChartAnalyzer)│
              │                               │
              │  Gemini 3.1-pro-preview       │
              │    → Gemini 3-flash           │
              │    → Gemini 2.5-flash         │
              │    → Gemini 2.0-flash         │
              │    → Gemini 1.5-pro (stable)  │
              └───────────────┬───────────────┘
                              │ all Gemini models exhausted
              ┌───────────────▼───────────────┐
              │    2. QwenVisionProvider      │  ← NEW
              │    (DashScope API)            │
              │                               │
              │  models from downloaded list  │
              │    (iterate in list order)    │
              └───────────────────────────────┘
                              │ all Qwen models exhausted
                              ▼
                   VisionChainExhaustedError
                   (caller handles — existing
                    try/except in GannSignalEngine
                    and GeminiIntegration)
```

---

## 4. File Structure

### New files

```
modules/gemini_chart_analyzer/core/analyzers/
  vision_provider_protocol.py      # VisionProvider Protocol (runtime_checkable)
  qwen_vision_provider.py          # Qwen DashScope caller
  vision_analyzer_chain.py         # Orchestrator: Gemini → Qwen
```

### Modified files

```
modules/gemini_chart_analyzer/core/analyzers/
  gemini_chart_analyzer.py         # GeminiChartAnalyzer implements VisionProvider
                                   # (backward-compatible, no interface change)

modules/gemini_gann_square/core/
  gann_signal_engine.py            # line 140: swap GeminiChartAnalyzer → VisionAnalyzerChain

modules/auto_trade/core/
  gemini_integration.py            # line 125: swap GeminiChartAnalyzer → VisionAnalyzerChain
                                   # add qwen_api_key param to __init__

config/
  config_api.py                    # add get_dashscope_api_key()
```

---

## 5. Interface: VisionProvider Protocol

```python
# vision_provider_protocol.py
from typing import Protocol, runtime_checkable

@runtime_checkable
class VisionProvider(Protocol):
    """Common interface for all vision AI providers."""

    provider_name: str  # "gemini", "qwen", etc.

    def analyze_chart(
        self,
        image_path: str,
        symbol: str,
        timeframe: str,
        prompt_type: str = "detailed",
        custom_prompt: str | None = None,
    ) -> str:
        """Call vision API and return raw text response."""
        ...

    def is_available(self) -> bool:
        """Return True if this provider has a valid API key and SDK."""
        ...
```

---

## 6. Qwen Vision Provider

### Supported Models (from downloaded list)

- Model list is loaded from the downloaded source (không hardcode danh sách 3 model cố định).
- `QwenVisionProvider` iterates models in the exact order of that list.
- If a model returns `429`, provider skips to the next model immediately.
- If a model returns `503`, provider retries with exponential backoff (max 3) before moving on.

### API Details

- **Provider:** Alibaba DashScope
- **Endpoint:** `https://dashscope-intl.aliyuncs.com/compatible-mode/v1`
- **SDK:** `openai` Python package (OpenAI-compatible interface)
- **Auth:** `DASHSCOPE_API_KEY` environment variable
- **Image format:** Base64 data URL → `"data:image/png;base64,<b64>"`

### Error handling

Same strategy as Gemini:
- `429 / quota` → skip to next Qwen model immediately (no retry)
- `503 / overloaded` → exponential backoff, up to 3 retries per model
- All Qwen models exhausted → raise `VisionChainExhaustedError`

### Sketch

```python
class QwenVisionProvider:
    provider_name = "qwen"

    def __init__(self, api_key: str | None = None, models: list[str] | None = None):
        self._api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        self._models = models or load_qwen_models_from_downloaded_list()
        if self._api_key:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=self._api_key,
                base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
            )

    def is_available(self) -> bool:
        return bool(self._api_key)

    def analyze_chart(self, image_path, symbol, timeframe,
                      prompt_type="detailed", custom_prompt=None) -> str:
        # 1. Read image → base64
        # 2. Build prompt (reuse existing prompt files from gemini_chart_analyzer)
        # 3. Try each model in self._models with retry/fallback
        # 4. Return raw text response
        ...
```

---

## 7. VisionAnalyzerChain Orchestrator

```python
class VisionAnalyzerChain:
    """Orchestrates multiple vision providers with automatic fallback."""

    def __init__(
        self,
        gemini_api_key: str | None = None,
        qwen_api_key: str | None = None,
        qwen_models: list[str] | None = None,
        skip_unavailable: bool = True,
    ):
        providers = []

        gemini = GeminiVisionProvider(api_key=gemini_api_key)
        if not skip_unavailable or gemini.is_available():
            providers.append(gemini)

        qwen = QwenVisionProvider(api_key=qwen_api_key, models=qwen_models)
        if not skip_unavailable or qwen.is_available():
            providers.append(qwen)

        if not providers:
            raise VisionChainExhaustedError("No vision providers configured.")

        self._providers = providers

    def analyze_chart(self, image_path, symbol, timeframe,
                      prompt_type="detailed", custom_prompt=None) -> str:
        last_error = None
        for provider in self._providers:
            try:
                result = provider.analyze_chart(
                    image_path, symbol, timeframe, prompt_type, custom_prompt
                )
                return result
            except Exception as e:
                log_warn(f"[{provider.provider_name}] failed: {e}. Trying next provider...")
                last_error = e
        raise VisionChainExhaustedError(
            f"All vision providers exhausted. Last error: {last_error}"
        )

    def is_available(self) -> bool:
        return any(p.is_available() for p in self._providers)
```

---

## 8. Consumer Integration Changes

### 8.1 GannSignalEngine

```python
# BEFORE (gann_signal_engine.py line 140)
self.gemini_analyzer = GeminiChartAnalyzer(api_key=gemini_api_key)

# AFTER
from modules.gemini_chart_analyzer.core.analyzers.vision_analyzer_chain import VisionAnalyzerChain

self.gemini_analyzer = VisionAnalyzerChain(
    gemini_api_key=gemini_api_key,
    # qwen_api_key auto-reads DASHSCOPE_API_KEY from env
)
# No other changes needed — .analyze_chart() signature is identical
```

### 8.2 GeminiIntegration (auto_trade)

```python
# BEFORE (__init__ line 125)
self.analyzer = GeminiChartAnalyzer(api_key=self._api_key)

# AFTER
from modules.gemini_chart_analyzer.core.analyzers.vision_analyzer_chain import VisionAnalyzerChain

self._qwen_api_key = qwen_api_key or os.getenv("DASHSCOPE_API_KEY")
self.analyzer = VisionAnalyzerChain(
    gemini_api_key=self._api_key,
    qwen_api_key=self._qwen_api_key,
)
```

`is_available()` in `GeminiIntegration` delegates to chain:
```python
def is_available(self) -> bool:
    return self.analyzer.is_available()  # True if ANY provider has a key
```

---

## 9. Configuration

### Environment variables

```bash
# Already existing
GEMINI_API_KEY=AIza...

# New — Alibaba DashScope
DASHSCOPE_API_KEY=sk-...
```

Get a key at: https://dashscope.aliyuncs.com  
International endpoint: https://dashscope-intl.aliyuncs.com/compatible-mode/v1

### config_api.py additions

```python
def get_dashscope_api_key() -> str | None:
    """Get DashScope (Qwen) API key."""
    return os.getenv("DASHSCOPE_API_KEY")
```

### skip_unavailable=True (default)

| Scenario | Behavior |
|----------|----------|
| Only `GEMINI_API_KEY` set | Chain uses Gemini only |
| Only `DASHSCOPE_API_KEY` set | Chain uses Qwen only |
| Both keys set | Chain uses Gemini first, Qwen on failure |
| Neither key set | `VisionChainExhaustedError` raised at init |

---

## 10. Failure Behavior

When all providers fail, `VisionAnalyzerChain.analyze_chart()` raises
`VisionChainExhaustedError`. Existing callers already handle this gracefully:

- **`GannSignalEngine.analyze()`** — the exception propagates to `GannSquareFilter`
  which catches it and skips the signal (existing behavior).
- **`GeminiIntegration.analyze_candidate()`** — wrapped in `try/except Exception`
  at line 291, returns `None` → pipeline skips Gemini step (existing behavior).

No changes needed to error handling at the caller level.

---

## 11. Logging

```
# Normal Gemini success (unchanged)
[INFO] Trying model: models/gemini-3.1-pro-preview
[INFO] Request succeeded with model: models/gemini-3.1-pro-preview

# Gemini quota exhausted, Qwen activates
[WARN] [gemini] All models exhausted (GeminiQuotaExceededError). Trying next provider...
[WARN] Falling back to Qwen vision provider...
[INFO] [qwen] Trying model from downloaded list: <model-id>
[INFO] [qwen] Request succeeded with model: <model-id>

# Both exhausted
[ERROR] All vision providers exhausted. Last error: QwenAPIError(...)
```

---

## 12. Dependencies

| Package | Purpose | Already installed? |
|---------|---------|-------------------|
| `openai` | Qwen DashScope OpenAI-compatible client | Likely yes (check requirements.txt) |
| `google-genai` | Gemini (existing) | Yes |
| `PIL` / `Pillow` | Image reading for base64 encode | Yes |

If `openai` is not installed: `pip install openai`

---

## 13. Implementation Order

1. `vision_provider_protocol.py` — define `VisionProvider` Protocol
2. `qwen_vision_provider.py` — implement `QwenVisionProvider`
3. `vision_analyzer_chain.py` — implement `VisionAnalyzerChain`
4. `config/config_api.py` — add `get_dashscope_api_key()`
5. `gann_signal_engine.py` — swap to `VisionAnalyzerChain`
6. `gemini_integration.py` — swap to `VisionAnalyzerChain`
7. Tests — `tests/test_vision_analyzer_chain.py`

---

## 14. Out of Scope

- Streaming responses (Qwen supports it, but not needed here)
- Async Qwen calls (existing `analyze_candidate_async` wraps sync via `asyncio.to_thread`)
- Qwen fine-tuning or model hosting
- Cost tracking / provider selection by price
