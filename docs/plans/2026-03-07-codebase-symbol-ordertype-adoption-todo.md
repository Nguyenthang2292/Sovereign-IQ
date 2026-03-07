# Codebase Symbol/OrderType Adoption To-Do

## Goal
Chuẩn hoa toan bo luong symbol, order type, va cancel params theo 3 file domain moi: `symbol_types.py`, `symbol_codec.py`, `order_type_codec.py`.

## Tasks
- [x] Task 1: Fix 2 bug critical ve `cancel_order` khong truyen stop params.
  - Files: `modules/auto_trade/execution/binance/order_execution.py`, `modules/auto_trade/monitoring/breakeven_manager.py`
  - Action: Dung `BinanceOrderType.cancel_params(order)` va truyen `params` vao `exchange.cancel_order(...)`.
  - Verify: Log cho thay lenh conditional (STOP/TAKE_PROFIT) duoc cancel thanh cong, khong con silent failure.

- [x] Task 2: Refactor manual order type parsing trong `tp_sl_sync.py` sang `BinanceOrderType`.
  - File: `modules/auto_trade/gui/services/tp_sl_sync.py`
  - Action: Thay `info.get("type")/origType`, string matching (`"TAKE_PROFIT"`, `"STOP"`) bang `resolve()` va `classify()`.
  - Verify: TP/SL classification khop voi order data khi test ca long va short.

- [x] Task 3: Refactor custom TP/SL fill detection trong `websocket_data_service.py`.
  - File: `modules/auto_trade/gui/services/websocket_data_service.py`
  - Action: Bo tuple matching thu cong (`"take_profit"`, `"stop_market"`, `"stop_loss"`), chuyen sang `BinanceOrderType.resolve(...)` + check theo type chuan.
  - Verify: Su kien fill TP/SL van duoc phat hien dung cho STOP_MARKET, TAKE_PROFIT_MARKET, STOP, TAKE_PROFIT.

- [x] Task 4: Thay toan bo deprecated `normalize_symbol()`/`normalize_symbol_key()` bang `SymbolCodec`.
  - Files chinh:
    - `modules/order_book/market_data_fetcher.py`
    - `modules/auto_trade/gui/main_window/risk_manager.py`
    - `modules/auto_trade/execution/order_manager.py`
    - `modules/auto_trade/execution/order_executor.py`
    - `modules/auto_trade/execution/binance/position_management.py`
    - `modules/adaptive_trend_LTS_mini/core/scanner/scan_all_symbols.py`
    - `modules/auto_trade/core/xgboost_serverless_filter.py`
    - `modules/auto_trade/core/atc_serverless_scanner.py`
    - `modules/auto_trade/core/atc_scanner.py`
  - Action: Dung `SymbolCodec().to_db(...)` va `SymbolCodec().to_ccxt(...)` tuy nguu canh.
  - Verify: Khong con import tu `modules.common.domain.symbols` trong code production.

- [x] Task 5: Refactor manual symbol string operations (replace/split) sang `SymbolCodec`.
  - Files chinh:
    - `modules/auto_trade/execution/order_executor.py`
    - `modules/auto_trade/execution/trailing_stop_ws_handler.py`
    - `modules/auto_trade/execution/negative_breakeven_ws_handler.py`
    - `modules/auto_trade/gui/services/data_service.py`
    - `modules/common/core/exchange_manager/__init__.py`
    - `modules/common/core/data_fetcher/binance_futures.py`
    - `modules/lstm/cli/workflow.py`
  - Action: Thay `.replace("/", "")`, `.split(":")[0]`, tu dong build futures symbol bang `codec.to_db()/to_ccxt()/to_futures()`.
  - Verify: Symbol key DB luon o dang `BTCUSDT`; symbol CCXT spot la `BTC/USDT`; futures la `BTC/USDT:USDT`.

- [x] Task 6: Chot API domain package va giam duong migration.
  - Files: `modules/common/domain/__init__.py`, `modules/common/domain/symbols.py`
  - Action:
    - Bo export deprecated functions khoi `__all__` (neu da migrate xong).
    - Giu wrapper tam thoi neu can de tranh break runtime, kem warning ro rang.
  - Verify: Import graph clean, pyright khong canh bao import deprecated path.

- [x] Task 7: Bo sung type annotations voi `DbSymbol`, `CcxtSymbol`, `FuturesSymbol` tai cac boundary quan trong.
  - Targets uu tien: execution path, GUI sync services, DB reconciliation path.
  - Action: Annotate input/output cua helper chuyen doi symbol va ham xu ly order quan trong.
  - Verify: Type checker phat hien sai lech format symbol som hon runtime.

## Final Verification (always last)
- [x] Search 1: `rg "normalize_symbol\(|normalize_symbol_key\(" modules` tra ve 0 ket qua trong production modules.
- [x] Search 2: `rg "replace\("/", ""\)|split\(":"\)\[0\]" modules/auto_trade modules/common/core` chi con ket qua khong lien quan conversion symbol.
- [x] Search 3: `rg "info\.get\("type"\)|origType|params=\{"stop": True\}" modules/auto_trade` cho thay cancel logic thong nhat qua `BinanceOrderType`.
- [x] Tests: Chay test target cho execution + gui sync + domain codec tests; tat ca pass.
- [ ] Runtime smoke: Chay 1 chu ky dat va huy TP/SL de xac nhan khong con orphan orders.

## Done When
- [ ] Khong con manual symbol conversion va deprecated symbol helpers trong production flow.
- [ ] Khong con cancel conditional order ma thieu `params={'stop': True}`.
- [ ] TP/SL classification theo `BinanceOrderType` thong nhat o moi luong.
- [ ] Type annotations cho 3 symbol kinds duoc ap dung tai cac diem giao tiep chinh.