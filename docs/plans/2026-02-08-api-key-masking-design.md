# API Key Masking – Design

**Date**: 2026-02-08  
**Topic**: Mask API key/secret in Config Panel UI; show read-only masked + "Change credentials" when saved.  
**Status**: Validated – ready for implementation

---

## 1. Scope, states, and UI behavior

**Scope**  
Apply only to the **API Keys tab** of the Config Panel (`config_panel.py`). No change to other screens; credentials elsewhere are used in memory only and are not displayed as text. Log masking (e.g. in `gemini_integration.py`) stays as-is.

**States**  
- **Saved**: Credentials exist for the selected exchange (from `CredentialManager.load_credentials(exchange)`); both `api_key` and `api_secret` are non-empty.  
- **Editing**: User clicked "Change credentials"; entry fields are visible and empty (or pre-filled only with what they type in this session).  
- **Not saved**: No credentials for the exchange; entry fields are visible and empty (first-time setup).

**UI behavior**  
- **When Saved**: Show read-only lines: "Current key: abcd••••••••xyz9" and "Current secret: ••••••••••••••••" (masked via a shared `mask_api_key` / `mask_secret` helper). Show one button: "Change credentials". Do **not** put the real key/secret into any entry widget.  
- **When user clicks "Change credentials"**: Hide the masked lines and "Change credentials" button. Show the two entry fields (API Key and API Secret), plus "Save" and "Cancel". Entries start empty (user types new values).  
- **Cancel**: Hide entries again, show masked display and "Change credentials". No persistence.  
- **Save**: Call existing `CredentialManager.save_credentials(exchange, key_from_entry, secret_from_entry)`, then switch back to Saved state (masked display; entries hidden).  
- **When Not saved**: Same as today: show entry fields and "Test connection" / "Save credentials". No masked block. Optionally show a short hint: "No credentials set. Enter key and secret below."

**Data rule**  
Full API key and secret must **never** be written into the entry widgets when loading or refreshing the panel. They are only set from user input in the current session (or left empty). "Saved" state is determined solely by reading from `CredentialManager.load_credentials(exchange)` and showing only masked strings.

---

## 2. Helpers and layout

**Mask helpers**  
Add a small utility (e.g. in `config_panel.py` at top level or in `gui.utils`) so it can be reused and tested:

- `mask_api_key(key: str) -> str`: if `len(key) <= 8` return `"*" * len(key)`; else `f"{key[:4]}{'*' * (len(key) - 8)}{key[-4:]}"`. Example: `"abcd1234wxyz9"` → `"abcd****wxyz9"`.
- `mask_secret(secret: str) -> str`: same rule as key, or optionally always `"••••••••"` (fixed length) to avoid revealing length. Design choice: use same first4+asterisks+last4 for consistency; if `len(secret) <= 8` use `"*" * 8` so length is not revealed.

**Widget structure in API Keys tab**  
Inside `api_key_frame` (the frame that is shown/hidden by mode):

- **Masked block** (visible when Saved): container frame with two labels (`api_key_masked_label`, `api_secret_masked_label`) and one button "Change credentials". Pack this block when credentials exist and we are not in Editing.
- **Entry block** (visible when Not saved or Editing): the existing `api_key_entry` and `api_secret_entry`, plus "Test connection" and "Save credentials". When in Editing mode, "Save credentials" acts as Save and then switches to Saved state; add a "Cancel" button that switches back to Saved without saving. When Not saved, keep current behavior (no Cancel; "Save credentials" saves and then we show Saved state).

So we have two sub-frames: `credentials_masked_frame` and `credentials_entry_frame`. Only one is packed at a time. On exchange change or mode change (or after Save/Cancel), call a small method e.g. `_refresh_credentials_display()`: if `CredentialManager.has_credentials(exchange)` and not editing, pack masked frame and forget entry frame; else pack entry frame and forget masked frame (when Not saved we don’t show masked frame at all).

**Exchange / mode**  
When user changes exchange (e.g. Binance ↔ Demo), run `_refresh_credentials_display()` so we show Saved vs Not saved for the new exchange. When switching to DRY_RUN, the whole `api_key_frame` is hidden (unchanged). When switching back to DEMO/PRODUCTION, show the frame and run `_refresh_credentials_display()` again.

**apply_settings**  
When `apply_settings(settings)` is called with `"api"` in settings: set `mode_var` and `exchange_var` as now; do **not** call `api_key_entry.insert(0, api.get("api_key", ""))` or similar. Instead, clear both entries and run `_refresh_credentials_display()` so the UI reflects Saved/Not saved from `CredentialManager` for the selected exchange. If the caller passes `api_key`/`api_secret` in settings (e.g. from a backup), we still must not put them into entries; we can ignore them for display and only persist via CredentialManager when user explicitly saves.

---

## 3. Error handling and edge cases

**Mask helpers**  
- If `key` or `secret` is `None` or empty, return `""` or a fixed placeholder (e.g. `"—"`) so labels never show raw `None`.  
- If string is shorter than 8 chars, design already says: key → `"*" * len(key)`; secret → `"*" * 8` to avoid length leak.

**Save / Test connection**  
- When user clicks "Save credentials" in entry block: validate as now (both key and secret non-empty); if empty, show existing warning "Please enter both API Key and API Secret" and do not save. On success, call `CredentialManager.save_credentials`, then run `_refresh_credentials_display()` so we switch to Saved state (masked block).  
- "Test connection" uses values from the entry widgets. When in Saved state, the entries are hidden and empty, so "Test connection" is only visible in the entry block (Not saved or Editing). No change to test logic; it continues to use `api_key_entry.get()` and `api_secret_entry.get()` when the entry block is visible.

**Cancel**  
- Cancel only applies when in Editing state. Hide entry block, show masked block; clear both entries so next time user clicks "Change credentials" they start empty. No call to CredentialManager.

**Exchange change while Editing**  
- If user has clicked "Change credentials" (entries visible, maybe partially filled) and then changes the exchange dropdown: either (a) run `_refresh_credentials_display()` for the new exchange (entries cleared, show masked or entry block for new exchange), or (b) show a short prompt "Discard unsaved changes and switch exchange?" then proceed. Design choice: (a) is simpler—on exchange change always run `_refresh_credentials_display()` and clear entries; any unsaved typing is discarded.

**apply_settings and credentials source**  
- Callers must not rely on `apply_settings` to "restore" full API key/secret into the UI. If the app has a backup/export that includes api_key, we do not re-insert it into entries. Only CredentialManager (e.g. from .env) is the source for "has credentials"; display is masked only. So any code that currently passes `api_key`/`api_secret` into `apply_settings` can stop passing them (or we ignore those keys in apply_settings for the api section).

**Empty / None from CredentialManager**  
- `load_credentials(exchange)` may return `{"api_key": None, "api_secret": None}` or empty strings. Treat as Not saved: show entry block, no masked block.

---

## 4. Testing and acceptance

**Unit tests**  
- `mask_api_key`: empty/None → `""` or `"—"`; `len(key) <= 8` → `"*" * len(key)`; longer key → first 4 + asterisks + last 4; no full key in output.  
- `mask_secret`: same for secret; if design uses `"*" * 8` for short secret, assert that.  
- Place helpers in a small module or at top of config_panel so they can be imported and tested without starting the GUI.

**GUI / integration**  
- With credentials saved for an exchange: open API Keys tab → masked block visible (key and secret masked), "Change credentials" visible; no full key in any widget.  
- Click "Change credentials" → entry block visible, entries empty; Save with valid key/secret → masked block visible again.  
- Cancel → masked block visible, entries cleared.  
- No credentials for exchange → entry block visible, "Save credentials" and "Test connection" as now.  
- apply_settings with "api" section does not insert api_key/api_secret into entries; after apply, display reflects CredentialManager (masked or entry block).

**Acceptance**  
- Full API key and secret are never shown in the Config Panel UI; only masked strings (first 4 + asterisks + last 4, or fixed-length for short/secret).  
- Saved state shows read-only masked lines + "Change credentials"; Editing shows entries + Save/Cancel.  
- Save, Cancel, exchange change, and mode change behave as in sections 1–3.  
- No regressions: Test connection and Save credentials still work when the entry block is visible.

---

**Design complete.** Ready for implementation; update REFACTORING_RECOMMENDATIONS.md § API Key Masking when done.
