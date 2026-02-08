# API Key Masking

## Goal
Implement API key/secret masking in Config Panel: when credentials exist, show read-only masked lines + "Change credentials"; never put full key/secret into entry widgets.

## Tasks
- [x] **1. Add mask helpers** — Add `mask_api_key(key)` and `mask_secret(secret)` in `modules/auto_trade/gui/utils/` (new module e.g. `mask_utils.py` or in existing utils). Rules: None/empty → `""` or `"—"`; key len≤8 → `"*"*len(key)`; key len>8 → first4+`*`*(n-8)+last4; secret len≤8 → `"*"*8`; secret len>8 → same as key.
  → Verify: `from modules.auto_trade.gui.utils.mask_utils import mask_api_key; assert "abcd" not in mask_api_key("abcd1234wxyz9")` and output is `"abcd****wxyz9"`.

- [x] **2. Add masked block and entry block** — In `config_panel.py` `_create_api_keys_tab`, inside `api_key_frame`: create `credentials_masked_frame` with two CTkLabels (`api_key_masked_label`, `api_secret_masked_label`) and button "Change credentials". Wrap existing API Key/Secret entries + Test + Save in `credentials_entry_frame`. Add instance var `_editing_credentials = False`. Pack only one of the two frames at a time (start with entry frame for backward compat).
  → Verify: API Keys tab still shows entries and Test/Save; no crash.

- [x] **3. Add _refresh_credentials_display()** — Get exchange from `exchange_var.get()`, get CredentialManager (same as _save_credentials). If `manager.has_credentials(exchange)` and not `_editing_credentials`: pack `credentials_masked_frame`, update both labels with `mask_api_key(creds["api_key"])` and `mask_secret(creds["api_secret"])` from `manager.load_credentials(exchange)`, then `credentials_entry_frame.pack_forget()`. Else: `credentials_masked_frame.pack_forget()`, clear both entries, pack `credentials_entry_frame`. Never insert real key/secret into entries.
  → Verify: With credentials in .env for exchange, call _refresh_credentials_display() and masked frame is visible with masked text.

- [x] **4. Wire Change credentials and Cancel** — "Change credentials" command: set `_editing_credentials = True`, call `_refresh_credentials_display()`. Add "Cancel" button in entry frame (e.g. next to Save): set `_editing_credentials = False`, clear both entries, `_refresh_credentials_display()`. When showing entry frame in Editing mode, show Cancel; when Not saved, hide Cancel or leave visible (design: show Cancel only when _editing_credentials).
  → Verify: Click "Change credentials" → entries visible and empty; Click Cancel → masked block visible again.

- [x] **5. Wire Save, apply_settings, exchange and mode** — In `_save_credentials` after successful save: set `_editing_credentials = False`, call `_refresh_credentials_display()`. In `apply_settings` when `"api"` in settings: set mode_var and exchange_var; remove any `api_key_entry.insert(0, api.get("api_key", ""))` (and secret); clear both entries, call `_refresh_credentials_display()`. On exchange dropdown change (if it has command): call `_refresh_credentials_display()`. In `_on_mode_change` when showing api_key_frame: call `_refresh_credentials_display()`.
  → Verify: Save credentials → masked block appears; switch exchange → display updates; apply_settings with api section does not fill entries with raw key.

- [x] **6. Unit tests** — Add `tests/auto_trade/gui/utils/test_mask_utils.py` (or under components): test mask_api_key(empty), mask_api_key(short), mask_api_key(long), mask_secret(empty), mask_secret(short→8 asterisks), mask_secret(long).
  → Verify: `pytest tests/auto_trade/gui/utils/test_mask_utils.py -v` (or chosen path) passes.

- [x] **7. Verification** — Run GUI, open Settings → API Keys. With no credentials: entry block and Save/Test visible. Save valid key/secret → masked block and "Change credentials" visible. Click "Change credentials" → entries empty, Save/Cancel visible; Cancel → masked again; Save with new values → masked again. Ensure full key never appears in UI.
  → Verify: Manual check; update REFACTORING_RECOMMENDATIONS.md § API Key Masking as DONE.

## Done When
- [x] Saved credentials show masked only; "Change credentials" reveals empty entries; Save/Cancel and exchange/mode refresh work.
- [x] apply_settings never inserts api_key/api_secret into entries.
- [x] Unit tests for mask helpers pass; REFACTORING_RECOMMENDATIONS updated.

## Notes
- Design: `docs/plans/2026-02-08-api-key-masking-design.md`
- CredentialManager is in `modules/auto_trade/gui/utils/credential_manager.py`; config_panel already uses it for save/test.
