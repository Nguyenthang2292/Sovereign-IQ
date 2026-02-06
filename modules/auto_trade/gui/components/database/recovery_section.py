"""Recovery Section Component for Database Panel."""

import customtkinter as ctk
import tkinter.messagebox as messagebox
import logging
from typing import Callable, Optional

from modules.auto_trade.database import session_scope

logger = logging.getLogger(__name__)


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

        ctk.CTkLabel(frame, text="🔄 Recovery Testing", font=("Roboto", 14, "bold")).pack(
            anchor="w", padx=10, pady=(10, 5)
        )

        # Input Frame
        input_frame = ctk.CTkFrame(frame, fg_color="transparent")
        input_frame.pack(fill="x", padx=10, pady=5)

        # Initial Loss
        ctk.CTkLabel(input_frame, text="Initial Loss $:").pack(side="left", padx=(0, 5))
        self.recovery_initial_loss = ctk.CTkEntry(input_frame, width=80)
        self.recovery_initial_loss.pack(side="left", padx=(0, 10))
        self.recovery_initial_loss.insert(0, "100")

        # Mode Selector
        ctk.CTkLabel(input_frame, text="Mode:").pack(side="left", padx=(0, 5))
        self.recovery_mode = ctk.CTkOptionMenu(input_frame, values=["fixed", "progressive", "adaptive"], width=100)
        self.recovery_mode.pack(side="left", padx=(0, 10))

        # Sequence Type
        ctk.CTkLabel(input_frame, text="Sequence:").pack(side="left", padx=(0, 5))
        self.recovery_sequence = ctk.CTkOptionMenu(input_frame, values=["win_streak", "mixed", "loss_heavy"], width=100)
        self.recovery_sequence.pack(side="left")

        # Button Frame
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
        """Run a test sequence for recovery strategy."""
        try:
            initial_loss = float(self.recovery_initial_loss.get())
            mode = self.recovery_mode.get()
            sequence_type = self.recovery_sequence.get()

            # Import the strategy
            from modules.auto_trade.strategies.gradual_recovery import (
                GradualRecoveryStrategy,
                RecoveryConfig,
            )

            # Define test sequences
            sequences = {
                "win_streak": [10.0, 12.0, 15.0, 18.0, 20.0],
                "mixed": [10.0, -5.0, 15.0, -3.0, 20.0],
                "loss_heavy": [-10.0, 5.0, -8.0, 3.0, -12.0, 25.0],
            }

            sequence = sequences.get(sequence_type, sequences["mixed"])

            # Create strategy with selected mode
            config = {
                "target_profit_per_trade": 5.0,
                "max_recovery_trades": 20,
                "margin_scaling_mode": mode,
                "leverage_scaling_mode": mode,
                "min_leverage": 2,
                "max_leverage": 10,
                "enable_streak_bonus": mode == "adaptive",
            }

            strategy = GradualRecoveryStrategy(initial_loss=initial_loss, config=config)

            # Run sequence and collect results
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

                output += f"Trade {i+1}: {trade_type} ${abs(amount):.2f}\n"
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

            # Final summary
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
        """View recovery statistics from database."""
        try:
            from modules.auto_trade.database.models import RecoverySession

            with session_scope() as session:
                # Query recovery sessions
                recoveries = session.query(RecoverySession).order_by(RecoverySession.created_at.desc()).limit(20).all()

                output = "Recovery Sessions:\n"
                output += "=" * 70 + "\n"

                if not recoveries:
                    output += "No recovery sessions found.\n"
                else:
                    for r in recoveries:
                        output += f"ID: {r.recovery_id[:8]}... | Status: {r.status}\n"
                        output += f"  Initial: ${r.initial_loss:.2f} | Remaining: ${r.remaining_loss:.2f}\n"
                        output += f"  Progress: {r.recovery_percentage:.1f}% | Trades: {r.trades_count}\n"
                        output += f"  Created: {r.created_at}\n"
                        output += "-" * 50 + "\n"

                self.data_viewer.delete("1.0", "end")
                self.data_viewer.insert("1.0", output)
                self.log_callback("Retrieved recovery stats", "INFO")

        except ImportError:
            self.log_callback("RecoverySession model not found in database", "WARNING")
            self.data_viewer.delete("1.0", "end")
            self.data_viewer.insert(
                "1.0", "Recovery model not available in database.\nUse the Recovery tab for live testing."
            )
        except Exception as e:
            self.log_callback(f"Failed to view recovery stats: {e}", "ERROR")

    def _clear_recovery_data(self):
        """Clear recovery test data."""
        if not messagebox.askyesno("Confirm Clear", "Clear all recovery test data?"):
            return

        try:
            from modules.auto_trade.database.models import RecoverySession

            with session_scope() as session:
                deleted = session.query(RecoverySession).delete()
                self.log_callback(f"Cleared {deleted} recovery sessions", "SUCCESS")

        except ImportError:
            self.log_callback("RecoverySession model not found - nothing to clear", "INFO")
        except Exception as e:
            self.log_callback(f"Failed to clear recovery data: {e}", "ERROR")
