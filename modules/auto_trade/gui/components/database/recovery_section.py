"""Recovery Section Component for Database Panel."""

from modules.common.ui.logging import log_info, log_error, log_warn, log_debug, log_success, log_system
import tkinter.messagebox as messagebox
from typing import Callable, cast

import customtkinter as ctk

from modules.auto_trade.database.repository.context import RepositoryContext
from modules.auto_trade.gui.config.database_panel_config import DatabasePanelConfig



class RecoverySection:
    """Recovery testing section component."""

    def __init__(self, parent: ctk.CTkFrame, log_callback: Callable, data_viewer: ctk.CTkTextbox):
        self.parent = parent
        self.log_callback = log_callback
        self.data_viewer = data_viewer
        self._create_ui()

    def _create_ui(self):
        """Create the recovery section UI."""
        frame = ctk.CTkFrame(self.parent)
        frame.pack(fill="x", padx=5, pady=5)

        ctk.CTkLabel(frame, text="🔄 Recovery Testing", font=DatabasePanelConfig.TITLE_FONT).pack(
            anchor="w",
            padx=DatabasePanelConfig.PADX_MEDIUM,
            pady=(DatabasePanelConfig.PADX_MEDIUM, DatabasePanelConfig.PADY_SMALL),
        )

        input_frame = ctk.CTkFrame(frame, fg_color="transparent")
        input_frame.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(input_frame, text="Initial Loss $:").pack(side="left", padx=(0, 5))
        self.recovery_initial_loss = ctk.CTkEntry(input_frame, width=80)
        self.recovery_initial_loss.pack(side="left", padx=(0, 10))
        self.recovery_initial_loss.insert(0, "100")

        ctk.CTkLabel(input_frame, text="Mode:").pack(side="left", padx=(0, 5))
        self.recovery_mode = ctk.CTkOptionMenu(input_frame, values=["fixed", "progressive", "adaptive"], width=100)
        self.recovery_mode.pack(side="left", padx=(0, 10))

        ctk.CTkLabel(input_frame, text="Sequence:").pack(side="left", padx=(0, 5))
        self.recovery_sequence = ctk.CTkOptionMenu(input_frame, values=["win_streak", "mixed", "loss_heavy"], width=100)
        self.recovery_sequence.pack(side="left")

        btn_frame = ctk.CTkFrame(frame, fg_color="transparent")
        btn_frame.pack(fill="x", padx=10, pady=5)

        ctk.CTkButton(
            btn_frame,
            text="🧪 Run Test Sequence",
            fg_color="#4488ff",
            hover_color="#2266cc",
            command=self._run_recovery_test_sequence,
        ).pack(side="left", padx=(0, 5), fill="x", expand=True)

        ctk.CTkButton(
            btn_frame,
            text="📊 View Recovery Stats",
            command=self._view_recovery_stats,
        ).pack(side="left", padx=5, fill="x", expand=True)

        ctk.CTkButton(
            btn_frame,
            text="🗑️ Clear Recovery Data",
            fg_color="#ff6644",
            hover_color="#cc4422",
            command=self._clear_recovery_data,
        ).pack(side="left", padx=(5, 0), fill="x", expand=True)

    def _run_recovery_test_sequence(self):
        """Run a test sequence for recovery strategy (pure in-memory, no DB)."""
        try:
            initial_loss = float(self.recovery_initial_loss.get())
            mode = self.recovery_mode.get()
            sequence_type = self.recovery_sequence.get()

            from modules.auto_trade.strategies.gradual_recovery import (
                GradualRecoveryStrategy,
                RecoveryConfig,
            )

            sequences = {
                "win_streak": [10.0, 12.0, 15.0, 18.0, 20.0],
                "mixed": [10.0, -5.0, 15.0, -3.0, 20.0],
                "loss_heavy": [-10.0, 5.0, -8.0, 3.0, -12.0, 25.0],
            }

            sequence = sequences.get(sequence_type, sequences["mixed"])
            config = {
                "target_profit_per_trade": 5.0,
                "max_recovery_trades": 20,
                "margin_scaling_mode": mode,
                "leverage_scaling_mode": mode,
                "min_leverage": 2,
                "max_leverage": 10,
                "enable_streak_bonus": mode == "adaptive",
            }

            strategy = GradualRecoveryStrategy(initial_loss=initial_loss, config=cast(RecoveryConfig, config))

            output = f"Recovery Test - Mode: {mode}, Sequence: {sequence_type}\n"
            output += f"Initial Loss: ${initial_loss:.2f}\n"
            output += "=" * 60 + "\n\n"

            for i, amount in enumerate(sequence):
                margin = strategy.calculate_next_position_size()
                leverage = strategy.calculate_next_leverage()
                state_before = strategy.get_state()

                if amount >= 0:
                    strategy.record_profit(amount)
                    trade_type = "PROFIT"
                else:
                    strategy.record_loss(abs(amount))
                    trade_type = "LOSS"

                state_after = strategy.get_state()

                output += f"Trade {i + 1}: {trade_type} ${abs(amount):.2f}\n"
                output += f"  Margin: ${margin:.2f}, Leverage: {leverage}x\n"
                output += (
                    f"  Progress: {state_before.recovery_percentage:.1f}% -> {state_after.recovery_percentage:.1f}%\n"
                )
                output += f"  Remaining: ${state_after.remaining_loss:.2f}\n"
                if trade_type == "PROFIT":
                    output += f"  Win Streak: {state_after.win_streak}\n"
                output += "\n"

                if state_after.is_complete:
                    output += "*** RECOVERY COMPLETE! ***\n"
                    break
                if strategy.should_stop():
                    output += "*** LIMIT REACHED - RECOVERY STOPPED ***\n"
                    break

            final_state = strategy.get_state()
            output += "=" * 60 + "\n"
            output += "FINAL SUMMARY:\n"
            output += f"  Total Trades: {final_state.trades_count}\n"
            output += f"  Recovery Progress: {final_state.recovery_percentage:.1f}%\n"
            output += f"  Remaining Loss: ${final_state.remaining_loss:.2f}\n"
            output += f"  Status: {'COMPLETE' if final_state.is_complete else 'ACTIVE'}\n"

            self.data_viewer.delete("1.0", "end")
            self.data_viewer.insert("1.0", output)
            self.log_callback(f"Recovery test completed: {sequence_type} with {mode} mode", "SUCCESS")

        except Exception as e:
            self.log_callback(f"Recovery test failed: {e}", "ERROR")

    def _view_recovery_stats(self):
        """View recovery statistics via RepositoryContext (backend-agnostic)."""
        try:
            ctx = RepositoryContext.from_env()
            recoveries = ctx.gradual_recovery.get_all_gradual_recoveries(limit=20)

            output = "Recovery Sessions:\n"
            output += "=" * 70 + "\n"

            if not recoveries:
                output += "No recovery sessions found.\n"
            else:
                for r in recoveries:
                    rid = str(r.get("recovery_id", ""))[:8]
                    status = r.get("status", "")
                    initial_loss = float(r.get("initial_loss", 0) or 0)
                    remaining = float(r.get("remaining_loss", 0) or 0)
                    pct = float(r.get("recovery_percentage", 0) or 0)
                    trades = r.get("trades_count", 0)
                    created = r.get("created_at", "")

                    output += f"ID: {rid}... | Status: {status}\n"
                    output += f"  Initial: ${initial_loss:.2f} | Remaining: ${remaining:.2f}\n"
                    output += f"  Progress: {pct:.1f}% | Trades: {trades}\n"
                    output += f"  Created: {created}\n"
                    output += "-" * 50 + "\n"

            self.data_viewer.delete("1.0", "end")
            self.data_viewer.insert("1.0", output)
            self.log_callback("Retrieved recovery stats", "INFO")

        except Exception as e:
            self.log_callback(f"Failed to view recovery stats: {e}", "ERROR")
            self.data_viewer.delete("1.0", "end")
            self.data_viewer.insert("1.0", f"Error retrieving recovery data: {e}")

    def _clear_recovery_data(self):
        """Clear recovery test data via RepositoryContext."""
        if not messagebox.askyesno("Confirm Clear", "Clear all recovery test data?"):
            return

        try:
            ctx = RepositoryContext.from_env()
            recoveries = ctx.gradual_recovery.get_all_gradual_recoveries(limit=99999)
            cancelled = 0
            for r in recoveries:
                rid = r.get("recovery_id")
                try:
                    ctx.gradual_recovery.cancel_gradual_recovery(rid)
                    cancelled += 1
                except Exception as inner:
                    log_warn(f"Could not cancel recovery {rid}: {inner}")

            self.log_callback(f"Cancelled {cancelled} recovery sessions", "SUCCESS")

        except Exception as e:
            self.log_callback(f"Failed to clear recovery data: {e}", "ERROR")
