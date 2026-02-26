"""Settings, scanner, and recovery handlers for Auto Trade Dashboard."""

from typing import TYPE_CHECKING, Any, Callable

from modules.auto_trade.gui.utils.websocket_data_service import WebSocketDataService
from modules.common.ui.logging import log_debug, log_error, log_info, log_warn


class SettingsRecoveryMixin:
    """Provide settings/scanner/recovery event handlers."""

    # --- Attribute stubs: real values injected by the concrete subclass (MainWindow).
    settings_manager: Any
    settings_handler: Any
    scanner_manager: Any
    data_service: Any
    ws_data_service: Any
    websocket_handler: Any
    recovery_manager: Any
    event_bus: Any
    config_panel: Any
    auto_trade_control: Any
    status_label: Any

    if TYPE_CHECKING:
        mode: str  # defined by AutoTradeDashboard.__init__; str so no conflict

        # Method stubs: satisfied by tkinter.Tk / the concrete subclass.
        def after(self, ms: int, func: Callable) -> str: ...
        def update_idletasks(self) -> None: ...
        def refresh_positions(self) -> None: ...
        def refresh_account(self) -> None: ...

    def on_settings_change(self, setting_type: str, value=None):
        """Handle settings change from ConfigPanel."""
        self.settings_handler.handle_settings_change(setting_type, value)

    def _restart_websocket_service(self):
        """Restart WebSocket service with updated credentials/mode."""
        try:
            if hasattr(self, "ws_data_service") and self.ws_data_service:
                log_info("Stopping existing WebSocket service...")
                self.ws_data_service.stop()

            self.settings_manager.load()

            log_info(f"Creating new WebSocket service (mode={self.mode})...")
            self.ws_data_service = WebSocketDataService(
                mode=self.mode,
                settings_manager=self.settings_manager,
                event_bus=self.event_bus if hasattr(self, "event_bus") else None,
                tk_root=self,
            )

            if self.mode != "DRY_RUN":
                self.websocket_handler.register_callbacks()
                self.ws_data_service.start()
                log_info("WebSocket service restarted successfully")
                if hasattr(self.data_service, "_reload_credentials"):
                    self.data_service._reload_credentials()
                self.after(500, self.refresh_account)
            else:
                log_info("DRY_RUN mode - WebSocket not started")

        except Exception as e:
            log_error(f"Error restarting WebSocket service: {e}")
            import traceback

            traceback.print_exc()

    def _refresh_theme_colors(self):
        """Refresh all component colors when theme changes."""
        self.settings_handler.refresh_theme_colors()

    def _get_current_status(self):
        """Build status dict for Current Settings (database, api_mode, api_connection)."""
        status = {"api_mode": getattr(self, "mode", "DRY_RUN")}
        try:
            from modules.auto_trade.database.repository.context import RepositoryContext

            RepositoryContext.from_env()
            status["database"] = "OK"
        except Exception:
            status["database"] = "Error"
        ws = getattr(self, "ws_data_service", None)
        if ws is None or status["api_mode"] == "DRY_RUN":
            status["api_connection"] = "N/A" if status["api_mode"] == "DRY_RUN" else "—"
        elif getattr(ws, "is_connected", False):
            status["api_connection"] = "Connected"
        else:
            status["api_connection"] = "Disconnected"
        return status

    def on_scan_toggle(self, action):
        """Handle scanner start/stop from ScannerControl."""
        self.scanner_manager.handle_scan_toggle(action)

    def on_risk_limits_toggle(self, enabled: bool):
        """Handle Risk Limits toggle from Trading tab and persist setting."""
        try:
            self.settings_manager.set("risk.limits_enabled", bool(enabled))
            self.settings_manager.save()

            if hasattr(self, "config_panel") and hasattr(self.config_panel, "risk_limits_enabled_var"):
                self.config_panel.risk_limits_enabled_var.set(bool(enabled))

            if hasattr(self, "status_label"):
                state_text = "enabled" if enabled else "disabled"
                self.status_label.configure(text=f"Risk limits {state_text}.")

            log_info(f"Risk limits toggled: {'enabled' if enabled else 'disabled'}")
        except Exception as e:
            log_warn(f"Failed to toggle risk limits: {e}")

    def on_scanner_config_change(self, config: dict):
        """Handle scanner configuration change."""
        self.scanner_manager.handle_config_change(config)

    def on_apply_settings(self):
        """Overwrite settings_manager from form and apply runtime updates."""
        try:
            if not hasattr(self, "config_panel"):
                return
            current = self.config_panel.get_settings()
            if hasattr(self.config_panel, "default_leverage_var"):
                current.setdefault("risk", {})["default_leverage"] = self.config_panel.default_leverage_var.get()
            for key in ("risk", "tp_sl", "api"):
                if key in current:
                    self.settings_manager.settings[key] = current[key]
            if "filters" in current:
                existing = self.settings_manager.settings.get("filters", {})
                self.settings_manager.settings["filters"] = {**existing, **current["filters"]}
            if hasattr(self.config_panel, "recovery_panel"):
                raw = self.config_panel.recovery_panel.get_config()
                try:
                    eb = raw.get("enable_streak_bonus", False)
                    enabled = raw.get("enabled", False)
                    self.settings_manager.settings["recovery"] = {
                        "enabled": enabled
                        if isinstance(enabled, bool)
                        else str(enabled).lower() in ("true", "1", "yes"),
                        "initial_loss": float(raw.get("initial_loss", 500)),
                        "target_profit_per_trade": float(raw.get("target_profit_per_trade", 5)),
                        "max_recovery_trades": int(raw.get("max_recovery_trades", 20)),
                        "margin_scaling_mode": str(raw.get("margin_scaling_mode", "fixed")),
                        "leverage_scaling_mode": str(raw.get("leverage_scaling_mode", "fixed")),
                        "min_leverage": int(raw.get("min_leverage", 2)),
                        "max_leverage": int(raw.get("max_leverage", 10)),
                        "enable_streak_bonus": (
                            eb if isinstance(eb, bool) else str(eb).lower() in ("true", "1", "yes")
                        ),
                    }
                except (TypeError, ValueError):
                    pass
            self.settings_manager.save()

            if hasattr(self, "auto_trade_control") and hasattr(self.auto_trade_control, "update_from_settings"):
                try:
                    self.auto_trade_control.update_from_settings(
                        self.settings_manager.settings, status=self._get_current_status()
                    )
                    self.auto_trade_control.update_idletasks()
                    self.update_idletasks()
                except Exception as refresh_err:
                    log_warn(f"Trading tab Current Settings refresh: {refresh_err}")

            if hasattr(self, "scanner_manager"):
                self.scanner_manager._pipeline_initialized = False
                self.scanner_manager.pipeline = None

            if hasattr(self, "status_label"):
                self.status_label.configure(text="Settings applied (Scanner, Trading, Gradual Recovery default).")
            log_info("Settings applied: Scanner, Trading, Gradual Recovery default (settings_manager overwritten)")

            if self.mode != "DRY_RUN":
                import threading

                t = threading.Thread(target=self._reapply_tp_sl_to_open_positions, daemon=True, name="reapply-tpsl")
                t.start()

        except Exception as e:
            log_error(f"Error applying settings: {e}")
            if hasattr(self, "status_label"):
                self.status_label.configure(text=f"Apply failed: {e}")

    def _reapply_tp_sl_to_open_positions(self) -> None:
        """Background thread: re-apply TP/SL from current settings to every open Binance position."""
        try:
            log_info("[ReapplyTPSL] Starting TP/SL re-apply for open positions...")

            tp_sl_cfg = self.settings_manager.settings.get("tp_sl", {})
            default_tp_pct = float(tp_sl_cfg.get("default_tp", 5.0))
            default_sl_pct = float(tp_sl_cfg.get("default_sl", 2.5))

            if default_tp_pct <= 0 or default_sl_pct <= 0:
                log_warn("[ReapplyTPSL] TP% or SL% is 0 — skipping re-apply.")
                return

            client = None
            if hasattr(self, "data_service") and self.data_service:
                self.data_service._reload_credentials()
                client = self.data_service._get_or_create_client()

            if client is None:
                log_warn("[ReapplyTPSL] No BinanceClient available — skipping re-apply.")
                return

            positions: list[dict[str, Any]] = []
            try:
                if self.data_service and self.data_service.data_fetcher:
                    positions = (
                        self.data_service.data_fetcher.fetch_binance_futures_positions(
                            api_key=self.data_service.api_key,
                            api_secret=self.data_service.api_secret,
                            testnet=self.data_service.testnet,
                        )
                        or []
                    )
            except Exception as fetch_err:
                log_error(f"[ReapplyTPSL] Could not fetch positions: {fetch_err}")
                return

            active_positions = [p for p in positions if float(p.get("contracts", 0)) != 0]
            if not active_positions:
                log_info("[ReapplyTPSL] No open positions — nothing to re-apply.")
                return

            log_info(f"[ReapplyTPSL] Found {len(active_positions)} open position(s). Re-applying TP/SL...")

            from modules.auto_trade.gui.utils.tp_sl_sync import TPSLSyncService

            results: list[str] = []

            for pos in active_positions:
                symbol: str = pos.get("symbol", "")
                side: str = pos.get("direction", "LONG").upper()
                entry_price: float = float(pos.get("entry_price", 0))

                if not symbol or entry_price <= 0:
                    log_warn(f"[ReapplyTPSL] Skipping position with missing symbol/entry_price: {pos}")
                    continue

                try:
                    actual_leverage: float = 1.0
                    try:
                        raw_lev = pos.get("leverage")
                        if raw_lev is None:
                            live_pos = client.get_position(symbol)
                            if live_pos:
                                raw_lev = live_pos.get("leverage") or (live_pos.get("info") or {}).get("leverage")
                        if raw_lev is not None:
                            actual_leverage = max(1.0, float(raw_lev))
                    except Exception as lev_err:
                        log_debug(f"[ReapplyTPSL] Leverage fetch failed for {symbol}: {lev_err}")

                    tp_price_pct = default_tp_pct / actual_leverage
                    sl_price_pct = default_sl_pct / actual_leverage

                    if side == "LONG":
                        new_tp_price = entry_price * (1.0 + tp_price_pct / 100.0)
                        new_sl_price = entry_price * (1.0 - sl_price_pct / 100.0)
                    else:
                        new_tp_price = entry_price * (1.0 - tp_price_pct / 100.0)
                        new_sl_price = entry_price * (1.0 + sl_price_pct / 100.0)

                    mark_price = TPSLSyncService._get_mark_price(client, symbol)
                    sl_buffer = TPSLSyncService._SL_MARK_BUFFER_PCT
                    if mark_price and mark_price > 0:
                        if side == "LONG" and new_sl_price >= mark_price:
                            new_sl_price = mark_price * (1.0 - sl_buffer)
                            log_info(
                                f"[ReapplyTPSL] {symbol}: SL clamped below mark "
                                f"${mark_price:.4f} → new SL ${new_sl_price:.4f}"
                            )
                        elif side == "SHORT" and new_sl_price <= mark_price:
                            new_sl_price = mark_price * (1.0 + sl_buffer)
                            log_info(
                                f"[ReapplyTPSL] {symbol}: SL clamped above mark "
                                f"${mark_price:.4f} → new SL ${new_sl_price:.4f}"
                            )

                    log_info(
                        f"[ReapplyTPSL] {symbol} {side} lev={actual_leverage}x "
                        f"entry=${entry_price} → TP=${new_tp_price:.4f} SL=${new_sl_price:.4f} "
                        f"(TP {default_tp_pct}% ROI, SL {default_sl_pct}% ROI)"
                    )

                    try:
                        cancel_result = client.cancel_open_orders(symbol)
                        log_info(f"[ReapplyTPSL] {symbol}: cancelled existing conditional orders → {cancel_result}")
                    except Exception as cancel_err:
                        log_debug(f"[ReapplyTPSL] {symbol}: cancel orders (non-fatal): {cancel_err}")

                    tp_ok = False
                    try:
                        tp_res = client.modify_take_profit(symbol, None, new_tp_price)
                        if tp_res and (tp_res.get("id") or tp_res.get("dry_run")):
                            tp_ok = True
                            log_info(f"[ReapplyTPSL] ✅ {symbol}: new TP placed @ ${new_tp_price:.4f}")
                        else:
                            log_warn(f"[ReapplyTPSL] ⚠️ {symbol}: TP order response unexpected: {tp_res}")
                    except Exception as tp_err:
                        log_error(f"[ReapplyTPSL] ❌ {symbol}: failed to place TP: {tp_err}")

                    sl_ok = False
                    try:
                        sl_res = client.modify_stop_loss(symbol, None, new_sl_price)
                        if sl_res and (sl_res.get("id") or sl_res.get("dry_run")):
                            sl_ok = True
                            log_info(f"[ReapplyTPSL] ✅ {symbol}: new SL placed @ ${new_sl_price:.4f}")
                        else:
                            log_warn(f"[ReapplyTPSL] ⚠️ {symbol}: SL order response unexpected: {sl_res}")
                    except Exception as sl_err:
                        log_error(f"[ReapplyTPSL] ❌ {symbol}: failed to place SL: {sl_err}")

                    if hasattr(self, "data_service") and self.data_service:
                        self.data_service._tpsl_cache.pop(symbol, None)
                        self.data_service._tpsl_cache_time.pop(symbol, None)

                    status = "✅" if (tp_ok and sl_ok) else ("⚠️" if (tp_ok or sl_ok) else "❌")
                    results.append(f"{status} {symbol}: TP={new_tp_price:.2f}, SL={new_sl_price:.2f}")

                except Exception as pos_err:
                    log_error(f"[ReapplyTPSL] Error processing {symbol}: {pos_err}", exc_info=True)
                    results.append(f"❌ {symbol}: error — {pos_err}")

            summary = " | ".join(results) if results else "no positions updated"
            log_info(f"[ReapplyTPSL] Done. {summary}")

            def _update_status():
                try:
                    if hasattr(self, "status_label") and self.status_label.winfo_exists():
                        self.status_label.configure(
                            text=f"Settings applied + TP/SL synced to Binance ({len(results)} position(s))."
                        )
                except Exception:
                    pass

            self.after(0, _update_status)
            self.after(500, self.refresh_positions)

        except Exception as outer_err:
            log_error(f"[ReapplyTPSL] Unexpected error: {outer_err}", exc_info=True)

    def reload_current_settings(self):
        """Force reload Trading tab Current Settings."""
        try:
            settings_to_show = None
            if hasattr(self, "config_panel"):
                current = self.config_panel.get_settings()
                if hasattr(self.config_panel, "default_leverage_var"):
                    current.setdefault("risk", {})["default_leverage"] = self.config_panel.default_leverage_var.get()
                existing_filters = self.settings_manager.settings.get("filters", {})
                settings_to_show = {
                    "risk": current.get("risk", {}),
                    "filters": {**existing_filters, **current.get("filters", {})},
                    "tp_sl": current.get("tp_sl", {}),
                    "api": current.get("api", {}),
                    "recovery": self.settings_manager.settings.get("recovery", {}),
                }
                if hasattr(self.config_panel, "recovery_panel"):
                    raw = self.config_panel.recovery_panel.get_config()
                    try:
                        eb = raw.get("enable_streak_bonus", False)
                        enabled = raw.get("enabled", False)
                        settings_to_show["recovery"] = {
                            "enabled": (
                                enabled if isinstance(enabled, bool) else str(enabled).lower() in ("true", "1", "yes")
                            ),
                            "initial_loss": float(raw.get("initial_loss", 500)),
                            "target_profit_per_trade": float(raw.get("target_profit_per_trade", 5)),
                            "max_recovery_trades": int(raw.get("max_recovery_trades", 20)),
                            "margin_scaling_mode": str(raw.get("margin_scaling_mode", "fixed")),
                            "leverage_scaling_mode": str(raw.get("leverage_scaling_mode", "fixed")),
                            "min_leverage": int(raw.get("min_leverage", 2)),
                            "max_leverage": int(raw.get("max_leverage", 10)),
                            "enable_streak_bonus": (
                                eb if isinstance(eb, bool) else str(eb).lower() in ("true", "1", "yes")
                            ),
                        }
                    except (TypeError, ValueError):
                        pass
            if settings_to_show is None:
                self.settings_manager.load()
                settings_to_show = self.settings_manager.settings

            if hasattr(self, "auto_trade_control") and hasattr(self.auto_trade_control, "update_from_settings"):
                self.auto_trade_control.update_from_settings(settings_to_show, status=self._get_current_status())
                self.auto_trade_control.update_idletasks()
                self.update_idletasks()
            if hasattr(self, "status_label"):
                self.status_label.configure(text="Current Settings reloaded (from Settings tab form).")
            log_info("Current Settings force-reloaded (Trading tab)")
        except Exception as e:
            log_warn(f"Force reload Current Settings: {e}")
            if hasattr(self, "status_label"):
                self.status_label.configure(text=f"Reload failed: {e}")

    def on_recovery_config_change(self, event_type: str, data):
        """Handle recovery configuration change."""
        try:
            log_info(f"Recovery {event_type}: {data}")

            if event_type == "recovery_started":
                self.settings_manager.set("recovery.enabled", True)
                self.settings_manager.set("recovery.config", data)
                self.settings_manager.save()

                if hasattr(self, "recovery_manager"):
                    self.recovery_manager.set_enabled(True)
                    self.recovery_manager.update_config(data)

            elif event_type == "recovery_reset":
                self.settings_manager.set("recovery.enabled", False)
                self.settings_manager.save()

                if hasattr(self, "recovery_manager"):
                    self.recovery_manager.reset()

            elif event_type == "recovery_alert":
                if hasattr(self, "status_label"):
                    self.status_label.configure(text=f"Recovery: {data}")

            elif event_type == "recovery_enabled_changed":
                enabled = data.get("enabled", False)
                self.settings_manager.set("recovery.enabled", enabled)
                self.settings_manager.save()

                if hasattr(self, "recovery_manager"):
                    self.recovery_manager.set_enabled(enabled)
                    log_info(f"RecoveryManager enabled={enabled}")

        except Exception as e:
            log_error(f"Error handling recovery config change: {e}")
