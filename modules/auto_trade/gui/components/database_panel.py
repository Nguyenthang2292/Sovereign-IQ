import os
import customtkinter as ctk
from typing import Optional, List, Dict, Any
from pathlib import Path
from datetime import datetime, timedelta
import uuid
import logging
import tkinter.messagebox as messagebox
import tkinter.filedialog as filedialog
import threading

# Database imports
from modules.auto_trade.database import (
    session_scope,
    create_order,
    save_signal,
    create_audit_log,
    get_open_positions,
    get_overall_stats,
    get_daily_stats,
    get_recent_signals,
    get_signal_performance_stats,
    get_active_martingale_chains,
    get_recent_audit_logs,
    create_database_backup,
    get_migration_manager,
    reconcile_orders_with_binance,
)
from modules.auto_trade.database.models import Order, Signal, MartingaleChain, AuditLog
from modules.auto_trade.database.config import DEFAULT_DB_PATH, DEFAULT_SCHEMA_PATH

# Set up logging
logger = logging.getLogger(__name__)


class DatabasePanel(ctk.CTkFrame):
    def __init__(self, parent, settings_manager):
        super().__init__(parent)
        self.settings_manager = settings_manager

        # Initialize state variables
        self.stats_labels: Dict[str, ctk.CTkLabel] = {}
        self.current_page = 1
        self.total_pages = 1
        self.page_size = 20
        self.current_table = "Orders"

        # Initialize database
        self.db_manager = self._init_database()

        # Create layout
        self._create_layout()

        # Load initial stats
        self._load_initial_stats()

    def _init_database(self):
        """Initialize database connection"""
        try:
            # Import here to avoid circular imports if any
            from modules.auto_trade.database import DatabaseManager

            # Get database path from settings or use default
            # Assuming settings_manager has a method or property for this,
            # otherwise falling back to default location
            db_path = "crypto_trading.db"
            if hasattr(self.settings_manager, "get_setting"):
                path_setting = self.settings_manager.get_setting("database.path")
                if path_setting:
                    db_path = path_setting

            # Initialize connection logic here if needed, or just return the manager class/instance
            # For now, we'll return the DatabaseManager class or instance as appropriate
            # Based on typical patterns, we might not need to instantiate a manager if using session_scope directly,
            # but following the task:
            return DatabaseManager
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            return None

    def _create_layout(self):
        """Create the main layout structure"""
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=3)  # Left panel (60%)
        self.grid_columnconfigure(1, weight=2)  # Right panel (40%)

        # Left panel (scrollable)
        self.left_panel = ctk.CTkScrollableFrame(self)
        self.left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 5))

        # Right panel (fixed)
        self.right_panel = ctk.CTkFrame(self)
        self.right_panel.grid(row=0, column=1, sticky="nsew", padx=(5, 0))

        # Create sections in left panel
        self._create_orders_section(self.left_panel)
        self._create_signals_section(self.left_panel)
        self._create_martingale_section(self.left_panel)
        self._create_recovery_section(self.left_panel)
        self._create_data_viewer_section(self.left_panel)

        # Create sections in right panel
        self._create_stats_section(self.right_panel)
        self._create_actions_section(self.right_panel)
        self._create_logs_section(self.right_panel)

    def _create_orders_section(self, parent):
        # Frame
        frame = ctk.CTkFrame(parent)
        frame.pack(fill="x", padx=5, pady=5)

        # Title
        ctk.CTkLabel(frame, text="📋 Orders Testing", font=("Roboto", 14, "bold")).pack(
            anchor="w", padx=10, pady=(10, 5)
        )

        # Inputs Frame
        input_frame = ctk.CTkFrame(frame, fg_color="transparent")
        input_frame.pack(fill="x", padx=10, pady=5)

        # Symbol
        ctk.CTkLabel(input_frame, text="Symbol:").pack(side="left", padx=(0, 5))
        self.order_symbol = ctk.CTkEntry(input_frame, width=100)
        self.order_symbol.pack(side="left", padx=(0, 10))
        self.order_symbol.insert(0, "BTCUSDT")

        # Side
        ctk.CTkLabel(input_frame, text="Side:").pack(side="left", padx=(0, 5))
        self.order_side = ctk.CTkOptionMenu(input_frame, values=["LONG", "SHORT"], width=100)
        self.order_side.pack(side="left", padx=(0, 10))

        # Create Button
        ctk.CTkButton(input_frame, text="Create Test Order", command=self._create_test_order).pack(side="right")

        # Query Buttons Frame
        btn_frame = ctk.CTkFrame(frame, fg_color="transparent")
        btn_frame.pack(fill="x", padx=10, pady=5)

        ctk.CTkButton(btn_frame, text="📊 Query Open Positions", command=self._query_open_positions).pack(
            side="left", padx=(0, 5), fill="x", expand=True
        )
        ctk.CTkButton(btn_frame, text="📈 Get Overall Stats", command=self._get_overall_stats).pack(
            side="left", padx=5, fill="x", expand=True
        )
        ctk.CTkButton(btn_frame, text="📅 Get Daily Stats (30d)", command=self._get_daily_stats).pack(
            side="left", padx=(5, 0), fill="x", expand=True
        )

    def _create_signals_section(self, parent):
        frame = ctk.CTkFrame(parent)
        frame.pack(fill="x", padx=5, pady=5)

        ctk.CTkLabel(frame, text="🎯 Signals Testing", font=("Roboto", 14, "bold")).pack(
            anchor="w", padx=10, pady=(10, 5)
        )

        input_frame = ctk.CTkFrame(frame, fg_color="transparent")
        input_frame.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(input_frame, text="Symbol:").pack(side="left", padx=(0, 5))
        self.signal_symbol = ctk.CTkEntry(input_frame, width=100)
        self.signal_symbol.pack(side="left", padx=(0, 10))
        self.signal_symbol.insert(0, "BTCUSDT")

        ctk.CTkLabel(input_frame, text="Confidence:").pack(side="left", padx=(0, 5))
        self.signal_confidence = ctk.CTkEntry(input_frame, width=100)
        self.signal_confidence.pack(side="left", padx=(0, 10))
        self.signal_confidence.insert(0, "0.85")

        ctk.CTkButton(input_frame, text="Create Test Signal", command=self._create_test_signal).pack(side="right")

        btn_frame = ctk.CTkFrame(frame, fg_color="transparent")
        btn_frame.pack(fill="x", padx=10, pady=5)

        ctk.CTkButton(btn_frame, text="📊 Get Recent Signals", command=self._get_recent_signals).pack(
            side="left", padx=(0, 5), fill="x", expand=True
        )
        ctk.CTkButton(btn_frame, text="📈 Signal Performance Stats", command=self._get_signal_stats).pack(
            side="left", padx=(5, 0), fill="x", expand=True
        )

    def _create_martingale_section(self, parent):
        frame = ctk.CTkFrame(parent)
        frame.pack(fill="x", padx=5, pady=5)

        ctk.CTkLabel(frame, text="🔄 Martingale Testing", font=("Roboto", 14, "bold")).pack(
            anchor="w", padx=10, pady=(10, 5)
        )

        btn_frame = ctk.CTkFrame(frame, fg_color="transparent")
        btn_frame.pack(fill="x", padx=10, pady=5)

        ctk.CTkButton(btn_frame, text="🔗 Get Active Chains", command=self._get_active_chains).pack(
            side="left", padx=(0, 5), fill="x", expand=True
        )
        ctk.CTkButton(btn_frame, text="📊 Chain Statistics", command=self._get_chain_stats).pack(
            side="left", padx=(5, 0), fill="x", expand=True
        )

    def _create_recovery_section(self, parent):
        """Recovery Testing Section for gradual recovery strategy"""
        frame = ctk.CTkFrame(parent)
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
        self.recovery_sequence = ctk.CTkOptionMenu(
            input_frame, values=["win_streak", "mixed", "loss_heavy"], width=100
        )
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
        """Run a test sequence for recovery strategy"""
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
            config: RecoveryConfig = {
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
                output += f"  Progress: {state_before.recovery_percentage:.1f}% -> {state_after.recovery_percentage:.1f}%\n"
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
            self._log(f"Recovery test completed: {sequence_type} with {mode} mode", "SUCCESS")

        except Exception as e:
            self._log(f"Recovery test failed: {e}", "ERROR")

    def _view_recovery_stats(self):
        """View recovery statistics from database"""
        try:
            from modules.auto_trade.database.models import RecoverySession

            with session_scope() as session:
                # Query recovery sessions
                recoveries = (
                    session.query(RecoverySession).order_by(RecoverySession.created_at.desc()).limit(20).all()
                )

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
                self._log("Retrieved recovery stats", "INFO")

        except ImportError:
            self._log("RecoverySession model not found in database", "WARNING")
            self.data_viewer.delete("1.0", "end")
            self.data_viewer.insert("1.0", "Recovery model not available in database.\nUse the Recovery tab for live testing.")
        except Exception as e:
            self._log(f"Failed to view recovery stats: {e}", "ERROR")

    def _clear_recovery_data(self):
        """Clear recovery test data"""
        if not messagebox.askyesno("Confirm Clear", "Clear all recovery test data?"):
            return

        try:
            from modules.auto_trade.database.models import RecoverySession

            with session_scope() as session:
                deleted = session.query(RecoverySession).delete()
                self._log(f"Cleared {deleted} recovery sessions", "SUCCESS")
                self._refresh_stats()

        except ImportError:
            self._log("RecoverySession model not found - nothing to clear", "INFO")
        except Exception as e:
            self._log(f"Failed to clear recovery data: {e}", "ERROR")

    def _create_data_viewer_section(self, parent):
        frame = ctk.CTkFrame(parent)
        frame.pack(fill="both", expand=True, padx=5, pady=5)

        header_frame = ctk.CTkFrame(frame, fg_color="transparent")
        header_frame.pack(fill="x", padx=10, pady=(10, 5))

        ctk.CTkLabel(header_frame, text="📂 Data Viewer", font=("Roboto", 14, "bold")).pack(side="left")

        self.table_selector = ctk.CTkOptionMenu(
            header_frame, values=["Orders", "Signals", "Martingale Chains", "Audit Log"], command=self._on_table_changed
        )
        self.table_selector.pack(side="right")

        self.data_viewer = ctk.CTkTextbox(frame, height=200, font=("Consolas", 12))
        self.data_viewer.pack(fill="both", expand=True, padx=10, pady=5)

        pagination_frame = ctk.CTkFrame(frame, fg_color="transparent")
        pagination_frame.pack(fill="x", padx=10, pady=5)

        self.prev_btn = ctk.CTkButton(pagination_frame, text="< Prev", width=80, command=self._prev_page)
        self.prev_btn.pack(side="left")

        self.page_label = ctk.CTkLabel(pagination_frame, text=f"Page {self.current_page}/{self.total_pages}")
        self.page_label.pack(side="left", fill="x", expand=True)

        self.next_btn = ctk.CTkButton(pagination_frame, text="Next >", width=80, command=self._next_page)
        self.next_btn.pack(side="right")

    def _create_test_order(self):
        symbol = self.order_symbol.get()
        side = self.order_side.get()

        try:
            with session_scope() as session:
                order_data = {
                    "order_id": f"TEST_{uuid.uuid4().hex[:8]}",
                    "client_order_id": f"AT_{int(datetime.now().timestamp())}_{symbol}",
                    "symbol": symbol,
                    "side": side,
                    "entry_price": 50000.0,
                    "amount": 0.01,
                    "leverage": 2,
                    "status": "OPEN",
                    "order_source": "PROGRAMMATIC",
                    "execution_mode": "AUTO",
                }

                create_order(session, order_data)

                self._log(f"Created test order for {symbol} ({side})", "SUCCESS")
                self._refresh_stats()
                if self.current_table == "Orders":
                    self._refresh_data_viewer()

        except Exception as e:
            self._log(f"Failed to create test order: {e}", "ERROR")

    def _query_open_positions(self):
        try:
            with session_scope() as session:
                positions = get_open_positions(session)

                output = "Open Positions:\n"
                output += "-" * 50 + "\n"
                for pos in positions:
                    output += f"ID: {pos.order_id} | {pos.symbol} | {pos.side} | Entry: {pos.entry_price}\n"

                self.data_viewer.delete("1.0", "end")
                self.data_viewer.insert("1.0", output)
                self._log(f"Queried {len(positions)} open positions", "INFO")

        except Exception as e:
            self._log(f"Failed to query open positions: {e}", "ERROR")

    def _get_overall_stats(self):
        try:
            with session_scope() as session:
                stats = get_overall_stats(session)

                output = "Overall Trading Statistics:\n"
                output += "=" * 30 + "\n"
                for key, value in stats.items():
                    output += f"{key.replace('_', ' ').title()}: {value}\n"

                self.data_viewer.delete("1.0", "end")
                self.data_viewer.insert("1.0", output)
                self._log("Retrieved overall stats", "INFO")

        except Exception as e:
            self._log(f"Failed to get overall stats: {e}", "ERROR")

    def _get_daily_stats(self):
        try:
            with session_scope() as session:
                stats = get_daily_stats(session, days=30)

                output = "Daily Statistics (Last 30 Days):\n"
                output += f"{'Date':<12} | {'Orders':<8} | {'PnL':<10}\n"
                output += "-" * 35 + "\n"

                for day in stats:
                    date_str = day.get("date", "N/A")
                    orders = day.get("total_orders", 0)
                    pnl = day.get("realized_pnl", 0.0)
                    output += f"{str(date_str):<12} | {orders:<8} | {pnl:<10.2f}\n"

                self.data_viewer.delete("1.0", "end")
                self.data_viewer.insert("1.0", output)
                self._log("Retrieved daily stats", "INFO")

        except Exception as e:
            self._log(f"Failed to get daily stats: {e}", "ERROR")

    def _create_test_signal(self):
        symbol = self.signal_symbol.get()
        try:
            confidence = float(self.signal_confidence.get())
            if not (0 <= confidence <= 1):
                raise ValueError("Confidence must be between 0 and 1")

            with session_scope() as session:
                signal_data = {
                    "correlation_id": f"SIG_{uuid.uuid4().hex[:8]}",
                    "symbol": symbol,
                    "signal_type": "LONG",
                    "confidence": confidence,
                    "executed": False,
                }

                save_signal(session, **signal_data)

                self._log(f"Created test signal for {symbol}", "SUCCESS")
                self._refresh_stats()
                if self.current_table == "Signals":
                    self._refresh_data_viewer()

        except ValueError as ve:
            self._log(str(ve), "WARNING")
        except Exception as e:
            self._log(f"Failed to create test signal: {e}", "ERROR")

    def _get_recent_signals(self):
        try:
            with session_scope() as session:
                signals = get_recent_signals(session, limit=50)

                output = "Recent Signals:\n"
                output += "-" * 50 + "\n"
                for sig in signals:
                    output += f"{sig.created_at} | {sig.symbol} | {sig.signal_type} | Conf: {sig.confidence}\n"

                self.data_viewer.delete("1.0", "end")
                self.data_viewer.insert("1.0", output)
                self._log("Retrieved recent signals", "INFO")

        except Exception as e:
            self._log(f"Failed to get recent signals: {e}", "ERROR")

    def _get_signal_stats(self):
        try:
            with session_scope() as session:
                stats = get_signal_performance_stats(session, days=30)

                output = "Signal Performance (30d):\n"
                output += "=" * 30 + "\n"
                for key, value in stats.items():
                    output += f"{key.replace('_', ' ').title()}: {value}\n"

                self.data_viewer.delete("1.0", "end")
                self.data_viewer.insert("1.0", output)
                self._log("Retrieved signal stats", "INFO")

        except Exception as e:
            self._log(f"Failed to get signal stats: {e}", "ERROR")

    def _get_active_chains(self):
        try:
            with session_scope() as session:
                chains = get_active_martingale_chains(session)

                output = "Active Martingale Chains:\n"
                output += "-" * 60 + "\n"
                for chain in chains:
                    output += f"ID: {chain.chain_id} | {chain.symbol} | Step: {chain.current_step}/{chain.max_allowed_steps} | PnL: {chain.total_loss + chain.total_recovery:.2f}\n"

                self.data_viewer.delete("1.0", "end")
                self.data_viewer.insert("1.0", output)
                self._log(f"Retrieved {len(chains)} active chains", "INFO")

        except Exception as e:
            self._log(f"Failed to get active chains: {e}", "ERROR")

    def _get_chain_stats(self):
        try:
            with session_scope() as session:
                total = session.query(MartingaleChain).count()
                active = session.query(MartingaleChain).filter(MartingaleChain.status == "ACTIVE").count()
                completed = session.query(MartingaleChain).filter(MartingaleChain.status == "RECOVERED").count()

                success_rate = (completed / total * 100) if total > 0 else 0

                output = "Martingale Chain Statistics:\n"
                output += "=" * 30 + "\n"
                output += f"Total Chains: {total}\n"
                output += f"Active Chains: {active}\n"
                output += f"Completed (Recovered): {completed}\n"
                output += f"Success Rate: {success_rate:.2f}%\n"

                self.data_viewer.delete("1.0", "end")
                self.data_viewer.insert("1.0", output)
                self._log("Retrieved chain stats", "INFO")

        except Exception as e:
            self._log(f"Failed to get chain stats: {e}", "ERROR")

    def _on_table_changed(self, value):
        self.current_table = value
        self.current_page = 1
        self._refresh_data_viewer()

    def _prev_page(self):
        if self.current_page > 1:
            self.current_page -= 1
            self._refresh_data_viewer()

    def _next_page(self):
        if self.current_page < self.total_pages:
            self.current_page += 1
            self._refresh_data_viewer()

    def _refresh_data_viewer(self):
        try:
            offset = (self.current_page - 1) * self.page_size
            limit = self.page_size
            table_name = self.table_selector.get()

            with session_scope() as session:
                # Get total count
                total_count = self._get_table_count(session, table_name)
                self.total_pages = max(1, (total_count + self.page_size - 1) // self.page_size)
                self.page_label.configure(text=f"Page {self.current_page}/{self.total_pages}")

                # Query data
                data = self._query_table_data(session, table_name, limit, offset)

                # Format output
                if not data:
                    self.data_viewer.delete("1.0", "end")
                    self.data_viewer.insert("1.0", f"No data found in {table_name}")
                    return

                # Create basic table view
                output = f"Table: {table_name} (Total: {total_count})\n"
                output += "-" * 80 + "\n"

                if data:
                    # Get columns from first item if it's a dict, or attributes if object
                    first = data[0]
                    if hasattr(first, "to_dict"):
                        first_dict = first.to_dict()
                    elif hasattr(first, "__dict__"):
                        first_dict = {k: v for k, v in first.__dict__.items() if not k.startswith("_")}
                    elif isinstance(first, dict):
                        first_dict = first
                    else:
                        first_dict = {"value": str(first)}

                    columns = list(first_dict.keys())
                    # Limit columns for display
                    display_cols = columns[:5]  # Show first 5 columns to fit

                    # Header
                    header = " | ".join([f"{col:<15}" for col in display_cols])
                    output += header + "\n"
                    output += "-" * len(header) + "\n"

                    for item in data:
                        if hasattr(item, "to_dict"):
                            item_dict = item.to_dict()
                        elif hasattr(item, "__dict__"):
                            item_dict = {k: v for k, v in item.__dict__.items() if not k.startswith("_")}
                        elif isinstance(item, dict):
                            item_dict = item
                        else:
                            item_dict = {"value": str(item)}

                        row = " | ".join([f"{str(item_dict.get(col, ''))[:15]:<15}" for col in display_cols])
                        output += row + "\n"

                self.data_viewer.delete("1.0", "end")
                self.data_viewer.insert("1.0", output)

        except Exception as e:
            self._log(f"Failed to refresh data viewer: {e}", "ERROR")

    def _get_table_count(self, session, table_name):
        if table_name == "Orders":
            return session.query(Order).count()
        elif table_name == "Signals":
            return session.query(Signal).count()
        elif table_name == "Martingale Chains":
            return session.query(MartingaleChain).count()
        elif table_name == "Audit Log":
            return session.query(AuditLog).count()
        return 0

    def _query_table_data(self, session, table_name, limit, offset):
        if table_name == "Orders":
            return session.query(Order).order_by(Order.created_at.desc()).limit(limit).offset(offset).all()
        elif table_name == "Signals":
            return session.query(Signal).order_by(Signal.created_at.desc()).limit(limit).offset(offset).all()
        elif table_name == "Martingale Chains":
            return (
                session.query(MartingaleChain)
                .order_by(MartingaleChain.created_at.desc())
                .limit(limit)
                .offset(offset)
                .all()
            )
        elif table_name == "Audit Log":
            return session.query(AuditLog).order_by(AuditLog.timestamp.desc()).limit(limit).offset(offset).all()
        return []

    def _refresh_stats(self):
        try:
            with session_scope() as session:
                # Count records
                total_orders = session.query(Order).count()
                open_positions = session.query(Order).filter(Order.status == "OPEN").count()
                total_signals = session.query(Signal).count()
                active_chains = session.query(MartingaleChain).filter(MartingaleChain.status == "ACTIVE").count()
                audit_logs = session.query(AuditLog).count()

                # Update labels
                if "total_orders" in self.stats_labels:
                    self.stats_labels["total_orders"].configure(text=str(total_orders))
                    self.stats_labels["open_positions"].configure(text=str(open_positions))
                    self.stats_labels["total_signals"].configure(text=str(total_signals))
                    self.stats_labels["active_chains"].configure(text=str(active_chains))
                    self.stats_labels["audit_logs"].configure(text=str(audit_logs))

                # Check last backup
                try:
                    from modules.auto_trade.database.config import DEFAULT_BACKUP_DIR

                    backup_dir = Path(DEFAULT_BACKUP_DIR)
                    if backup_dir.exists():
                        backups = sorted(list(backup_dir.glob("*.db")), key=lambda f: f.stat().st_mtime, reverse=True)
                        if backups and "last_backup" in self.stats_labels:
                            last_backup_time = datetime.fromtimestamp(backups[0].stat().st_mtime).strftime(
                                "%Y-%m-%d %H:%M"
                            )
                            self.stats_labels["last_backup"].configure(text=last_backup_time)
                        elif "last_backup" in self.stats_labels:
                            self.stats_labels["last_backup"].configure(text="None")
                    elif "last_backup" in self.stats_labels:
                        self.stats_labels["last_backup"].configure(text="None")
                except Exception:
                    if "last_backup" in self.stats_labels:
                        self.stats_labels["last_backup"].configure(text="Error")

        except Exception as e:
            self._log(f"Failed to refresh stats: {e}", "ERROR")

    def _load_initial_stats(self):
        self._refresh_stats()

    def _create_stats_section(self, parent):
        frame = ctk.CTkFrame(parent)
        frame.pack(fill="x", padx=5, pady=5)

        ctk.CTkLabel(frame, text="📊 Database Stats", font=("Roboto", 14, "bold")).pack(
            anchor="w", padx=10, pady=(10, 5)
        )

        stats_items = [
            ("total_orders", "Total Orders"),
            ("open_positions", "Open Positions"),
            ("total_signals", "Total Signals"),
            ("active_chains", "Active Chains"),
            ("audit_logs", "Audit Logs"),
            ("last_backup", "Last Backup"),
        ]

        for key, label in stats_items:
            row = ctk.CTkFrame(frame, fg_color="transparent")
            row.pack(fill="x", padx=10, pady=2)

            ctk.CTkLabel(row, text=f"{label}:").pack(side="left")
            value_label = ctk.CTkLabel(row, text="...")
            value_label.pack(side="right")

            self.stats_labels[key] = value_label

    def _create_actions_section(self, parent):
        frame = ctk.CTkFrame(parent)
        frame.pack(fill="x", padx=5, pady=5)

        ctk.CTkLabel(frame, text="⚡ Quick Actions", font=("Roboto", 14, "bold")).pack(
            anchor="w", padx=10, pady=(10, 5)
        )

        actions = [
            ("💾 Create Backup", self._create_backup),
            ("🔄 Run Migrations", self._run_migrations),
            ("🔄 Reconcile with Binance", self._reconcile_with_binance),
            ("🧹 Cleanup Old Records", self._cleanup_records),
            ("📤 Export to CSV", self._export_csv),
            ("📋 View Audit Log", self._view_audit_log),
            ("🔍 Check Integrity", self._check_integrity),
        ]

        for text, command in actions:
            ctk.CTkButton(frame, text=text, command=command).pack(fill="x", padx=10, pady=2)

    def _create_logs_section(self, parent):
        frame = ctk.CTkFrame(parent)
        frame.pack(fill="both", expand=True, padx=5, pady=5)

        header = ctk.CTkFrame(frame, fg_color="transparent")
        header.pack(fill="x", padx=10, pady=(10, 5))

        ctk.CTkLabel(header, text="📝 Activity Logs", font=("Roboto", 14, "bold")).pack(side="left")
        ctk.CTkButton(header, text="Clear", width=60, height=24, command=self._clear_logs).pack(side="right")

        self.logs_viewer = ctk.CTkTextbox(frame)
        self.logs_viewer.pack(fill="both", expand=True, padx=10, pady=5)

    def _log(self, message: str, level: str = "INFO"):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.logs_viewer.insert("end", f"[{timestamp}] [{level}] {message}\n")
        self.logs_viewer.see("end")

    def _clear_logs(self):
        self.logs_viewer.delete("1.0", "end")

    def _create_backup(self):
        try:
            backup_path = create_database_backup()
            self._log(f"Backup created at: {backup_path}", "SUCCESS")
            self._refresh_stats()
        except Exception as e:
            self._log(f"Backup failed: {e}", "ERROR")

    def _run_migrations(self):
        try:
            manager = get_migration_manager(DEFAULT_DB_PATH, DEFAULT_SCHEMA_PATH)
            if manager:
                self._log("Migration manager retrieved (Manual trigger not fully implemented)", "INFO")
            else:
                self._log("Migration manager not available", "WARNING")
        except Exception as e:
            self._log(f"Migration run failed: {e}", "ERROR")

    def _cleanup_records(self):
        if not messagebox.askyesno("Confirm Cleanup", "Are you sure you want to delete old records (>90 days)?"):
            return

        try:
            from modules.auto_trade.database.utils import DatabaseCleaner

            with session_scope() as session:
                deleted_orders = DatabaseCleaner.cleanup_old_records(session, Order, days_to_keep=90)
                deleted_signals = DatabaseCleaner.cleanup_old_records(session, Signal, days_to_keep=90)
                deleted_logs = DatabaseCleaner.cleanup_old_records(
                    session, AuditLog, days_to_keep=90, date_column="timestamp"
                )

                msg = f"Cleanup complete. Deleted: {deleted_orders} orders, {deleted_signals} signals, {deleted_logs} logs"
                self._log(msg, "SUCCESS")
                messagebox.showinfo("Cleanup Complete", msg)
                self._refresh_stats()

        except Exception as e:
            self._log(f"Cleanup failed: {e}", "ERROR")

    def _export_csv(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")])
        if not file_path:
            return

        try:
            from modules.auto_trade.database.utils import DataExporter

            table_map = {
                "Orders": Order,
                "Signals": Signal,
                "Martingale Chains": MartingaleChain,
                "Audit Log": AuditLog,
            }

            model_class = table_map.get(self.current_table)
            if not model_class:
                self._log(f"Unknown table selected for export: {self.current_table}", "ERROR")
                return

            with session_scope() as session:
                success = DataExporter.export_to_csv(session, model_class, file_path)

                if success:
                    self._log(f"Exported {self.current_table} to {file_path}", "SUCCESS")
                else:
                    self._log("Export failed (check logs)", "ERROR")

        except Exception as e:
            self._log(f"Export failed: {e}", "ERROR")

    def _view_audit_log(self):
        try:
            with session_scope() as session:
                logs = get_recent_audit_logs(session, limit=100)

                output = "Recent Audit Logs:\n"
                output += "-" * 80 + "\n"
                for log in logs:
                    output += f"[{log.timestamp}] [{log.severity}] {log.event_type}: {log.event_summary}\n"

                self.data_viewer.delete("1.0", "end")
                self.data_viewer.insert("1.0", output)
                self._log("Retrieved audit logs", "INFO")

        except Exception as e:
            self._log(f"Failed to view audit log: {e}", "ERROR")

    def _check_integrity(self):
        try:
            from modules.auto_trade.database import get_db_manager
            from sqlalchemy import text

            manager = get_db_manager()
            with manager.engine.connect() as conn:
                result = conn.execute(text("PRAGMA integrity_check")).fetchone()
                status = result[0] if result else "Unknown"

                self._log(f"Integrity Check: {status}", "INFO" if status == "ok" else "ERROR")
                messagebox.showinfo("Integrity Check", f"Database Integrity: {status}")

        except Exception as e:
            self._log(f"Integrity check failed: {e}", "ERROR")

    def _reconcile_with_binance(self):
        """Fetch AT_* orders from Binance and insert any missing into DB."""
        api_key = os.getenv("BINANCE_API_KEY", "").strip()
        api_secret = os.getenv("BINANCE_API_SECRET", "").strip()
        if not api_key or not api_secret:
            self._log("Reconcile skipped: BINANCE_API_KEY or BINANCE_API_SECRET not set", "WARNING")
            messagebox.showwarning(
                "Reconcile",
                "Set BINANCE_API_KEY and BINANCE_API_SECRET to reconcile with Binance.",
            )
            return
        testnet = bool(self.settings_manager.get("api.testnet", False))
        symbols = self.settings_manager.get("filters.symbol_whitelist") or None
        self._log("Reconciling with Binance (last 24h)...", "INFO")
        try:
            result = reconcile_orders_with_binance(
                api_key=api_key,
                api_secret=api_secret,
                testnet=testnet,
                symbols=symbols,
                since_hours=24,
            )
            inserted = result.get("inserted", 0)
            skipped = result.get("skipped", 0)
            errors = result.get("errors", [])
            self._log(f"Reconcile done: inserted={inserted}, skipped={skipped}", "SUCCESS")
            for err in errors[:5]:
                self._log(err, "ERROR")
            if len(errors) > 5:
                self._log(f"... and {len(errors) - 5} more errors", "ERROR")
            self._refresh_stats()
            messagebox.showinfo(
                "Reconcile",
                f"Inserted: {inserted}, Skipped (already in DB): {skipped}. Errors: {len(errors)}",
            )
        except Exception as e:
            self._log(f"Reconcile failed: {e}", "ERROR")
            messagebox.showerror("Reconcile", str(e))
