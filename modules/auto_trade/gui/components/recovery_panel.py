import random
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, cast

import customtkinter as ctk

from modules.auto_trade.gui.components.empty_state import EmptyState
from modules.auto_trade.gui.utils.svg_icons import get_icon
from modules.auto_trade.strategies.gradual_recovery import (
    GradualRecoveryStrategy,
    RecoveryConfig,
    create_recovery_plan,
)


class RecoveryPanel(ctk.CTkFrame):
    """
    Recovery Panel for monitoring and configuring Gradual Recovery Strategy
    Displays current recovery state, progress, and allows configuration
    """

    def __init__(self, parent, on_config_change=None, mode: str = "DRY_RUN", compact: bool = False):
        super().__init__(parent)

        self.on_config_change = on_config_change
        self.recovery_strategy: Optional[GradualRecoveryStrategy] = None
        self.mode = mode
        self.compact = compact
        self.test_log_entries: List[str] = []

        # Title
        title = ctk.CTkLabel(
            self,
            text="  Gradual Recovery",
            font=("Arial", 16, "bold"),
            image=get_icon("repeat", size=(20, 20)),
            compound="left",
        )
        title.pack(pady=(10, 15))

        # Create tabbed interface
        self.tabview = ctk.CTkTabview(self)
        self.tabview.pack(fill="both", expand=True, padx=10, pady=10)

        # Add tabs
        self._create_status_tab()
        self._create_config_tab()
        if not compact:
            self._create_history_tab()

        # Add Test tab ONLY in DRY_RUN mode
        if mode == "DRY_RUN" and not compact:
            self._create_test_tab()

        # Add Expand button for compact mode
        if compact:
            self._add_expand_button()

    def _add_expand_button(self):
        """Add expand button for compact mode"""
        expand_frame = ctk.CTkFrame(self, fg_color="transparent")
        expand_frame.pack(fill="x", padx=10, pady=(0, 5))

        ctk.CTkButton(
            expand_frame,
            text="  Expand Full View",
            fg_color="#4488ff",
            hover_color="#2266cc",
            command=self._open_expanded_modal,
            height=28,
            image=get_icon("zoom_in", size=(16, 16)),
            compound="left",
        ).pack(fill="x")

    def _open_expanded_modal(self):
        """Open expanded recovery panel in a modal window"""
        modal = ctk.CTkToplevel(self)
        modal.title("Gradual Recovery - Full View")
        modal.geometry("500x700")
        modal.transient(self.winfo_toplevel())
        modal.grab_set()

        # Create a full RecoveryPanel in the modal (non-compact)
        full_panel = RecoveryPanel(
            modal,
            on_config_change=self.on_config_change,
            mode=self.mode,
            compact=False,
        )
        full_panel.pack(fill="both", expand=True, padx=10, pady=10)

        # Sync state from compact panel to full panel
        if self.recovery_strategy:
            full_panel.recovery_strategy = self.recovery_strategy
            full_panel._update_status_display()

        # Close button
        ctk.CTkButton(
            modal,
            text="Close",
            fg_color="#666666",
            hover_color="#444444",
            command=modal.destroy,
        ).pack(fill="x", padx=20, pady=(0, 10))

        # Sync changes back when modal closes
        def on_modal_close():
            if full_panel.recovery_strategy:
                self.recovery_strategy = full_panel.recovery_strategy
                self._update_status_display()
            modal.destroy()

        modal.protocol("WM_DELETE_WINDOW", on_modal_close)

    def _create_status_tab(self):
        """Create Status tab showing current recovery state"""
        tab = self.tabview.add("Status")

        status_frame = ctk.CTkFrame(tab, fg_color="transparent")
        status_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Empty State
        self.empty_state_widget = EmptyState(
            status_frame,
            icon="🔄",
            message="No active recovery",
            hint="Configure and start a recovery session to begin.",
            action_text="Configure Recovery",
            action_callback=lambda: self.tabview.set("Config"),
        )
        self.empty_state_widget.pack(fill="both", expand=True, padx=20, pady=20)

        # Frame to hold active recovery details
        self.active_recovery_frame = ctk.CTkFrame(status_frame, fg_color="transparent")

        # Initial Loss
        self.initial_loss_label = ctk.CTkLabel(
            self.active_recovery_frame, text="Initial Loss: $0.00", font=("Arial", 12)
        )
        self.initial_loss_label.pack(anchor="w", pady=(5, 2))

        # Remaining Loss
        self.remaining_loss_label = ctk.CTkLabel(
            self.active_recovery_frame, text="Remaining Loss: $0.00", font=("Arial", 14, "bold"), text_color="#ff6b6b"
        )
        self.remaining_loss_label.pack(anchor="w", pady=(5, 2))

        # Progress Bar (widget + percentage label on one row, no floating)
        progress_frame = ctk.CTkFrame(self.active_recovery_frame, fg_color="transparent")
        progress_frame.pack(fill="x", pady=(15, 5), expand=False)

        self.progress_bar_widget = ctk.CTkProgressBar(progress_frame, height=10)
        self.progress_bar_widget.set(0)
        self.progress_bar_widget.pack(side="left", fill="x", expand=True, padx=(0, 8))

        self.progress_bar_label = ctk.CTkLabel(
            progress_frame, text="0%", font=("Arial", 11), text_color="#888", width=36
        )
        self.progress_bar_label.pack(side="left", anchor="w")

        # Recovery Percentage
        self.recovery_pct_label = ctk.CTkLabel(self.active_recovery_frame, text="Recovery: 0.0%", font=("Arial", 12))
        self.recovery_pct_label.pack(anchor="w", pady=(5, 2))

        # Separator
        separator = ctk.CTkFrame(self.active_recovery_frame, height=2, fg_color="#444")
        separator.pack(fill="x", pady=(15, 15))

        # Trades Count
        self.trades_count_label = ctk.CTkLabel(self.active_recovery_frame, text="Trades: 0", font=("Arial", 11))
        self.trades_count_label.pack(anchor="w", pady=2)

        # Win Streak
        self.win_streak_label = ctk.CTkLabel(
            self.active_recovery_frame, text="Win Streak: 0", font=("Arial", 11), text_color="#00ff88"
        )
        self.win_streak_label.pack(anchor="w", pady=2)

        # Estimated Trades Remaining
        self.est_trades_label = ctk.CTkLabel(self.active_recovery_frame, text="Est. Remaining: 0", font=("Arial", 11))
        self.est_trades_label.pack(anchor="w", pady=2)

        # Separator
        separator2 = ctk.CTkFrame(self.active_recovery_frame, height=2, fg_color="#444")
        separator2.pack(fill="x", pady=(15, 15))

        # Next Trade Recommendations
        rec_frame = ctk.CTkFrame(self.active_recovery_frame, fg_color="#2a2a2a")
        rec_frame.pack(fill="x", pady=(5, 10))

        rec_title = ctk.CTkLabel(rec_frame, text="Next Trade Recommendations", font=("Arial", 11, "bold"))
        rec_title.pack(anchor="w", pady=(10, 5), padx=10)

        # Margin
        self.margin_label = ctk.CTkLabel(rec_frame, text="Margin: $0.00", font=("Arial", 11))
        self.margin_label.pack(anchor="w", pady=2, padx=10)

        # Leverage
        self.leverage_label = ctk.CTkLabel(rec_frame, text="Leverage: 0x", font=("Arial", 11))
        self.leverage_label.pack(anchor="w", pady=(2, 10), padx=10)

        # Status Message (this will be handled by EmptyState or updated based on recovery_strategy)
        self.status_label = ctk.CTkLabel(
            self.active_recovery_frame, text="No active recovery", font=("Arial", 11), text_color="gray"
        )
        self.status_label.pack(pady=(10, 5))

        # Reset Button
        reset_btn = ctk.CTkButton(
            self.active_recovery_frame,
            text="  Reset Recovery",
            fg_color="#ff6644",
            hover_color="#cc4422",
            command=self._on_reset,
            image=get_icon("repeat", size=(16, 16)),
            compound="left",
        )
        reset_btn.pack(fill="x", pady=(5, 5))

        # Stop Button
        stop_btn = ctk.CTkButton(
            self.active_recovery_frame,
            text="  Stop Recovery",
            fg_color="#444444",
            hover_color="#333333",
            command=self._on_stop_recovery,
            image=get_icon("square", size=(16, 16)),
            compound="left",
        )
        stop_btn.pack(fill="x", pady=(0, 10))

    def _create_config_tab(self):
        """Create Configuration tab for recovery settings"""
        tab = self.tabview.add("Config")

        config_frame = ctk.CTkFrame(tab, fg_color="transparent")
        config_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Configure grid for 2 columns
        config_frame.grid_columnconfigure(0, weight=1)
        config_frame.grid_columnconfigure(1, weight=1)

        # Row 0: Enable Auto-Recovery checkbox (spans both columns)
        self.recovery_enabled_var = ctk.BooleanVar(value=False)
        enabled_checkbox = ctk.CTkCheckBox(
            config_frame,
            text="Enable Auto-Recovery",
            variable=self.recovery_enabled_var,
            command=self._on_enabled_changed,
            font=("Arial", 12, "bold"),
            fg_color="#00ff88",
            hover_color="#00cc66",
        )
        enabled_checkbox.grid(row=0, column=0, columnspan=2, sticky="w", pady=(5, 15))

        # Row 1-2: Initial Loss | Target Profit Per Trade
        # Left: Initial Loss
        label = ctk.CTkLabel(config_frame, text="Initial Loss ($):", font=("Arial", 11))
        label.grid(row=1, column=0, sticky="w", padx=(0, 5), pady=(5, 2))

        self.initial_loss_entry = ctk.CTkEntry(config_frame, placeholder_text="500.00")
        self.initial_loss_entry.grid(row=2, column=0, sticky="ew", padx=(0, 5), pady=(2, 10))
        self.initial_loss_entry.insert(0, "500.00")

        # Right: Target Profit Per Trade
        label = ctk.CTkLabel(config_frame, text="Target Profit Per Trade (%):", font=("Arial", 11))
        label.grid(row=1, column=1, sticky="w", padx=(5, 0), pady=(5, 2))

        self.target_profit_entry = ctk.CTkEntry(config_frame, placeholder_text="5.0")
        self.target_profit_entry.grid(row=2, column=1, sticky="ew", padx=(5, 0), pady=(2, 10))
        self.target_profit_entry.insert(0, "5.0")

        # Row 3-4: Max Recovery Trades
        label = ctk.CTkLabel(config_frame, text="Max Recovery Trades:", font=("Arial", 11))
        label.grid(row=3, column=0, sticky="w", padx=(0, 5), pady=(5, 2))

        self.max_trades_entry = ctk.CTkEntry(config_frame, placeholder_text="20")
        self.max_trades_entry.grid(row=4, column=0, sticky="ew", padx=(0, 5), pady=(2, 10))
        self.max_trades_entry.insert(0, "20")

        # Row 5-6: Margin Scaling Mode | Leverage Scaling Mode
        # Left: Margin Scaling Mode
        label = ctk.CTkLabel(config_frame, text="Margin Scaling Mode:", font=("Arial", 11))
        label.grid(row=5, column=0, sticky="w", padx=(0, 5), pady=(5, 2))

        self.margin_mode_var = ctk.StringVar(value="fixed")
        margin_mode_dropdown = ctk.CTkComboBox(
            config_frame,
            values=["fixed", "progressive", "adaptive"],
            variable=self.margin_mode_var,
        )
        margin_mode_dropdown.grid(row=6, column=0, sticky="ew", padx=(0, 5), pady=(2, 10))

        # Right: Leverage Scaling Mode
        label = ctk.CTkLabel(config_frame, text="Leverage Scaling Mode:", font=("Arial", 11))
        label.grid(row=5, column=1, sticky="w", padx=(5, 0), pady=(5, 2))

        self.leverage_mode_var = ctk.StringVar(value="fixed")
        leverage_mode_dropdown = ctk.CTkComboBox(
            config_frame,
            values=["fixed", "progressive", "adaptive"],
            variable=self.leverage_mode_var,
        )
        leverage_mode_dropdown.grid(row=6, column=1, sticky="ew", padx=(5, 0), pady=(2, 10))

        # Row 7: Leverage Range (Min/Max on same row, spans both columns)
        leverage_frame = ctk.CTkFrame(config_frame, fg_color="transparent")
        leverage_frame.grid(row=7, column=0, columnspan=2, sticky="ew", pady=(5, 10))

        label = ctk.CTkLabel(leverage_frame, text="Min:", font=("Arial", 11), text_color="gray")
        label.pack(side="left")

        self.min_leverage_entry = ctk.CTkEntry(leverage_frame, placeholder_text="2", width=80)
        self.min_leverage_entry.pack(side="left", padx=(5, 15))
        self.min_leverage_entry.insert(0, "2")

        label = ctk.CTkLabel(leverage_frame, text="Max:", font=("Arial", 11), text_color="gray")
        label.pack(side="left")

        self.max_leverage_entry = ctk.CTkEntry(leverage_frame, placeholder_text="10", width=80)
        self.max_leverage_entry.pack(side="left", padx=(5, 0))
        self.max_leverage_entry.insert(0, "10")

        # Row 8: Enable Streak Bonus (spans both columns)
        self.streak_bonus_var = ctk.BooleanVar(value=False)
        streak_bonus_checkbox = ctk.CTkCheckBox(
            config_frame, text="Enable Streak Bonus", variable=self.streak_bonus_var
        )
        streak_bonus_checkbox.grid(row=8, column=0, columnspan=2, sticky="w", pady=(5, 10))

        # Row 9-10: Presets (spans both columns)
        preset_frame = ctk.CTkFrame(config_frame, fg_color="transparent")
        preset_frame.grid(row=9, column=0, columnspan=2, sticky="ew", pady=(15, 5))

        preset_label = ctk.CTkLabel(preset_frame, text="Presets:", font=("Arial", 11, "bold"))
        preset_label.pack(anchor="w", pady=(0, 5))

        button_frame = ctk.CTkFrame(preset_frame, fg_color="transparent")
        button_frame.pack(fill="x")

        conservative_btn = ctk.CTkButton(
            button_frame,
            text="Conservative",
            fg_color="#44aa88",
            hover_color="#338866",
            command=lambda: self._apply_preset("conservative"),
            width=90,
        )
        conservative_btn.pack(side="left", padx=(0, 5))

        moderate_btn = ctk.CTkButton(
            button_frame,
            text="Moderate",
            fg_color="#4488ff",
            hover_color="#2266cc",
            command=lambda: self._apply_preset("moderate"),
            width=90,
        )
        moderate_btn.pack(side="left", padx=5)

        aggressive_btn = ctk.CTkButton(
            button_frame,
            text="Aggressive",
            fg_color="#ff6644",
            hover_color="#cc4422",
            command=lambda: self._apply_preset("aggressive"),
            width=90,
        )
        aggressive_btn.pack(side="left", padx=5)

        # Row 11: Start Recovery Button (spans both columns)
        start_btn = ctk.CTkButton(
            config_frame,
            text="  Start Recovery",
            fg_color="#00ff88",
            hover_color="#00cc66",
            command=self._on_start_recovery,
            image=get_icon("rocket", size=(16, 16)),
            compound="left",
        )
        start_btn.grid(row=11, column=0, columnspan=2, sticky="ew", pady=(15, 10))

        # Row 12: Recovery Plan Preview (spans both columns)
        plan_frame = ctk.CTkFrame(config_frame, fg_color="#2a2a2a")
        plan_frame.grid(row=12, column=0, columnspan=2, sticky="ew", pady=(5, 10))

        plan_title = ctk.CTkLabel(plan_frame, text="Recovery Plan Preview", font=("Arial", 11, "bold"))
        plan_title.pack(anchor="w", pady=(10, 5), padx=10)

        self.plan_estimated_trades_label = ctk.CTkLabel(plan_frame, text="Est. Trades: 0", font=("Arial", 10))
        self.plan_estimated_trades_label.pack(anchor="w", pady=2, padx=10)

        self.plan_risk_label = ctk.CTkLabel(plan_frame, text="Risk: -", font=("Arial", 10))
        self.plan_risk_label.pack(anchor="w", pady=(2, 10), padx=10)

        preview_btn = ctk.CTkButton(
            config_frame,
            text="  Preview Plan",
            fg_color="#888888",
            hover_color="#666666",
            command=self._update_plan_preview,
            image=get_icon("bar_chart_2", size=(16, 16)),
            compound="left",
        )
        preview_btn.grid(row=13, column=0, columnspan=2, sticky="ew", pady=(0, 10))

    def _on_enabled_changed(self):
        """Handle enabled checkbox change."""
        enabled = self.recovery_enabled_var.get()
        if self.on_config_change:
            self.on_config_change("recovery_enabled_changed", {"enabled": enabled})

    def _create_history_tab(self):
        """Create History tab showing recovery progress over time"""
        tab = self.tabview.add("History")

        history_frame = ctk.CTkFrame(tab, fg_color="transparent")
        history_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Placeholder for chart
        chart_placeholder = ctk.CTkLabel(
            history_frame,
            text="📈 Recovery History Chart\n(Coming Soon)",
            font=("Arial", 14),
            text_color="gray",
        )
        chart_placeholder.pack(expand=True)

    def _create_test_tab(self):
        """Create Test tab for DRY_RUN mode only - allows testing recovery scenarios"""
        tab = self.tabview.add("Test")

        # Scrollable frame for all test content
        scroll_frame = ctk.CTkScrollableFrame(tab, fg_color="transparent")
        scroll_frame.pack(fill="both", expand=True, padx=5, pady=5)

        # Mode Info
        mode_label = ctk.CTkLabel(
            scroll_frame,
            text=f"🧪 Test Mode: {self.margin_mode_var.get()}/{self.leverage_mode_var.get()}",
            font=("Arial", 11),
            text_color="#ffaa00",
        )
        mode_label.pack(anchor="w", pady=(5, 10))
        self.test_mode_label = mode_label

        # ========== Manual Entry Section ==========
        manual_frame = ctk.CTkFrame(scroll_frame, fg_color="#2a2a2a", corner_radius=8)
        manual_frame.pack(fill="x", pady=(0, 10))

        ctk.CTkLabel(manual_frame, text="📝 Manual Trade Entry", font=("Arial", 12, "bold")).pack(
            anchor="w", pady=(10, 5), padx=10
        )

        # Amount input
        amount_frame = ctk.CTkFrame(manual_frame, fg_color="transparent")
        amount_frame.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(amount_frame, text="Amount ($):", font=("Arial", 11)).pack(side="left")
        self.test_amount_entry = ctk.CTkEntry(amount_frame, width=100, placeholder_text="10.00")
        self.test_amount_entry.pack(side="left", padx=(10, 0))
        self.test_amount_entry.insert(0, "10.00")

        # Profit/Loss buttons
        btn_frame = ctk.CTkFrame(manual_frame, fg_color="transparent")
        btn_frame.pack(fill="x", padx=10, pady=(5, 10))

        ctk.CTkButton(
            btn_frame,
            text="✅ Record Profit",
            fg_color="#00ff88",
            hover_color="#00cc66",
            command=self._test_record_profit,
            width=120,
        ).pack(side="left", padx=(0, 10))

        ctk.CTkButton(
            btn_frame,
            text="❌ Record Loss",
            fg_color="#ff4444",
            hover_color="#cc0000",
            command=self._test_record_loss,
            width=120,
        ).pack(side="left")

        # ========== Sequence Generator Section ==========
        seq_frame = ctk.CTkFrame(scroll_frame, fg_color="#2a2a2a", corner_radius=8)
        seq_frame.pack(fill="x", pady=(0, 10))

        ctk.CTkLabel(seq_frame, text="🎲 Random Sequence Generator", font=("Arial", 12, "bold")).pack(
            anchor="w", pady=(10, 5), padx=10
        )

        # Sequence parameters grid
        params_frame = ctk.CTkFrame(seq_frame, fg_color="transparent")
        params_frame.pack(fill="x", padx=10, pady=5)

        # Row 1: Number of trades and Win rate
        row1 = ctk.CTkFrame(params_frame, fg_color="transparent")
        row1.pack(fill="x", pady=2)

        ctk.CTkLabel(row1, text="Trades:", font=("Arial", 10)).pack(side="left")
        self.test_num_trades_entry = ctk.CTkEntry(row1, width=50, placeholder_text="10")
        self.test_num_trades_entry.pack(side="left", padx=(5, 15))
        self.test_num_trades_entry.insert(0, "10")

        ctk.CTkLabel(row1, text="Win Rate %:", font=("Arial", 10)).pack(side="left")
        self.test_win_rate_entry = ctk.CTkEntry(row1, width=50, placeholder_text="60")
        self.test_win_rate_entry.pack(side="left", padx=(5, 0))
        self.test_win_rate_entry.insert(0, "60")

        # Row 2: Avg profit and Avg loss
        row2 = ctk.CTkFrame(params_frame, fg_color="transparent")
        row2.pack(fill="x", pady=2)

        ctk.CTkLabel(row2, text="Avg Profit $:", font=("Arial", 10)).pack(side="left")
        self.test_avg_profit_entry = ctk.CTkEntry(row2, width=50, placeholder_text="15")
        self.test_avg_profit_entry.pack(side="left", padx=(5, 15))
        self.test_avg_profit_entry.insert(0, "15")

        ctk.CTkLabel(row2, text="Avg Loss $:", font=("Arial", 10)).pack(side="left")
        self.test_avg_loss_entry = ctk.CTkEntry(row2, width=50, placeholder_text="10")
        self.test_avg_loss_entry.pack(side="left", padx=(5, 0))
        self.test_avg_loss_entry.insert(0, "10")

        ctk.CTkButton(
            seq_frame,
            text="🎲 Generate Random Sequence",
            fg_color="#4488ff",
            hover_color="#2266cc",
            command=self._test_run_random_sequence,
        ).pack(fill="x", padx=10, pady=(5, 10))

        # ========== Preset Scenarios Section ==========
        preset_frame = ctk.CTkFrame(scroll_frame, fg_color="#2a2a2a", corner_radius=8)
        preset_frame.pack(fill="x", pady=(0, 10))

        ctk.CTkLabel(preset_frame, text="📋 Preset Scenarios", font=("Arial", 12, "bold")).pack(
            anchor="w", pady=(10, 5), padx=10
        )

        preset_btn_frame = ctk.CTkFrame(preset_frame, fg_color="transparent")
        preset_btn_frame.pack(fill="x", padx=10, pady=(5, 10))

        ctk.CTkButton(
            preset_btn_frame,
            text="📈 Perfect Recovery",
            fg_color="#00aa66",
            hover_color="#008855",
            command=lambda: self._test_run_preset("perfect"),
            width=100,
        ).pack(side="left", padx=(0, 5))

        ctk.CTkButton(
            preset_btn_frame,
            text="📉 Struggle",
            fg_color="#ff8844",
            hover_color="#cc6622",
            command=lambda: self._test_run_preset("struggle"),
            width=80,
        ).pack(side="left", padx=5)

        ctk.CTkButton(
            preset_btn_frame,
            text="💥 Failed",
            fg_color="#ff4444",
            hover_color="#cc0000",
            command=lambda: self._test_run_preset("failed"),
            width=80,
        ).pack(side="left", padx=5)

        # ========== Mode Comparison Section ==========
        compare_frame = ctk.CTkFrame(scroll_frame, fg_color="#2a2a2a", corner_radius=8)
        compare_frame.pack(fill="x", pady=(0, 10))

        ctk.CTkLabel(compare_frame, text="🔬 Mode Comparison Test", font=("Arial", 12, "bold")).pack(
            anchor="w", pady=(10, 5), padx=10
        )

        ctk.CTkLabel(
            compare_frame,
            text="Run identical sequence across fixed/progressive/adaptive modes",
            font=("Arial", 10),
            text_color="gray",
        ).pack(anchor="w", padx=10, pady=(0, 5))

        ctk.CTkButton(
            compare_frame,
            text="🔬 Run Mode Comparison",
            fg_color="#aa44ff",
            hover_color="#8822cc",
            command=self._test_run_mode_comparison,
        ).pack(fill="x", padx=10, pady=(0, 10))

        # ========== Results Log ==========
        log_frame = ctk.CTkFrame(scroll_frame, fg_color="#1a1a1a", corner_radius=8)
        log_frame.pack(fill="both", expand=True, pady=(0, 5))

        log_header = ctk.CTkFrame(log_frame, fg_color="transparent")
        log_header.pack(fill="x", padx=10, pady=(10, 5))

        ctk.CTkLabel(log_header, text="📋 Test Results Log", font=("Arial", 12, "bold")).pack(side="left")

        ctk.CTkButton(
            log_header,
            text="Clear",
            fg_color="#666666",
            hover_color="#444444",
            command=self._test_clear_log,
            width=60,
            height=24,
        ).pack(side="right")

        self.test_log_viewer = ctk.CTkTextbox(log_frame, height=150, font=("Consolas", 10))
        self.test_log_viewer.pack(fill="both", expand=True, padx=10, pady=(0, 10))

    def _test_log(self, message: str, level: str = "INFO"):
        """Add entry to test log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        level_colors = {
            "INFO": "",
            "SUCCESS": "[+] ",
            "ERROR": "[!] ",
            "WARN": "[~] ",
        }
        prefix = level_colors.get(level, "")
        entry = f"[{timestamp}] {prefix}{message}\n"
        self.test_log_entries.append(entry)
        self.test_log_viewer.insert("end", entry)
        self.test_log_viewer.see("end")

    def _test_clear_log(self):
        """Clear the test log"""
        self.test_log_entries.clear()
        self.test_log_viewer.delete("0.0", "end")

    def _test_record_profit(self):
        """Record a manual profit"""
        if not self.recovery_strategy:
            self._test_log("No active recovery! Start recovery first.", "ERROR")
            return

        try:
            amount = float(self.test_amount_entry.get())
            state_before = self.recovery_strategy.get_state()
            margin_before = self.recovery_strategy.calculate_next_position_size()
            leverage_before = self.recovery_strategy.calculate_next_leverage()

            self.record_trade(amount)

            state_after = self.recovery_strategy.get_state()
            margin_after = self.recovery_strategy.calculate_next_position_size()
            leverage_after = self.recovery_strategy.calculate_next_leverage()

            self._test_log(f"PROFIT +${amount:.2f}", "SUCCESS")
            self._test_log(f"  Remaining: ${state_before.remaining_loss:.2f} -> ${state_after.remaining_loss:.2f}")
            self._test_log(
                f"  Progress: {state_before.recovery_percentage:.1f}% -> {state_after.recovery_percentage:.1f}%"
            )
            self._test_log(f"  Margin: ${margin_before:.2f} -> ${margin_after:.2f}")
            self._test_log(f"  Leverage: {leverage_before}x -> {leverage_after}x")
            self._test_log(f"  Win Streak: {state_after.win_streak}")

            # Update mode label
            self._update_test_mode_label()

        except ValueError:
            self._test_log("Invalid amount!", "ERROR")

    def _test_record_loss(self):
        """Record a manual loss"""
        if not self.recovery_strategy:
            self._test_log("No active recovery! Start recovery first.", "ERROR")
            return

        try:
            amount = float(self.test_amount_entry.get())
            state_before = self.recovery_strategy.get_state()

            self.record_trade(-amount)

            state_after = self.recovery_strategy.get_state()

            self._test_log(f"LOSS -${amount:.2f}", "ERROR")
            self._test_log(f"  Remaining: ${state_before.remaining_loss:.2f} -> ${state_after.remaining_loss:.2f}")
            self._test_log(f"  Win Streak RESET: {state_before.win_streak} -> 0")

            self._update_test_mode_label()

        except ValueError:
            self._test_log("Invalid amount!", "ERROR")

    def _test_run_random_sequence(self):
        """Run a random trade sequence"""
        if not self.recovery_strategy:
            self._test_log("No active recovery! Start recovery first.", "ERROR")
            return

        try:
            num_trades = int(self.test_num_trades_entry.get())
            win_rate = float(self.test_win_rate_entry.get()) / 100
            avg_profit = float(self.test_avg_profit_entry.get())
            avg_loss = float(self.test_avg_loss_entry.get())

            self._test_log(f"=== Random Sequence: {num_trades} trades, {win_rate * 100:.0f}% win rate ===")

            initial_state = self.recovery_strategy.get_state()

            for i in range(num_trades):
                if self.recovery_strategy.get_state().is_complete:
                    self._test_log(f"Recovery COMPLETE at trade {i + 1}!", "SUCCESS")
                    break

                if self.recovery_strategy.should_stop():
                    self._test_log(f"Recovery STOPPED at trade {i + 1} - limit reached", "WARN")
                    break

                is_win = random.random() < win_rate
                if is_win:
                    amount = avg_profit * (0.8 + random.random() * 0.4)  # +/- 20%
                    self.record_trade(amount)
                    self._test_log(f"Trade {i + 1}: WIN +${amount:.2f}")
                else:
                    amount = avg_loss * (0.8 + random.random() * 0.4)
                    self.record_trade(-amount)
                    self._test_log(f"Trade {i + 1}: LOSS -${amount:.2f}")

            final_state = self.recovery_strategy.get_state()
            self._test_log("=== Sequence Complete ===")
            self._test_log(f"  Total Trades: {final_state.trades_count - initial_state.trades_count}")
            self._test_log(
                f"  Progress: {initial_state.recovery_percentage:.1f}% -> {final_state.recovery_percentage:.1f}%"
            )

            self._update_test_mode_label()

        except ValueError as e:
            self._test_log(f"Invalid input: {e}", "ERROR")

    def _test_run_preset(self, preset: str):
        """Run a preset test scenario"""
        if not self.recovery_strategy:
            self._test_log("No active recovery! Start recovery first.", "ERROR")
            return

        self._test_log(f"=== Preset: {preset.upper()} ===")

        if preset == "perfect":
            # All wins until recovery complete
            sequence = [15.0, 18.0, 20.0, 22.0, 25.0, 28.0, 30.0, 35.0, 40.0, 50.0]
            self._test_log("Running perfect win streak scenario...")

        elif preset == "struggle":
            # 40% win rate with volatility
            sequence = [12.0, -8.0, -10.0, 15.0, -6.0, 20.0, -12.0, -5.0, 25.0, 18.0, -7.0, 30.0]
            self._test_log("Running struggle scenario (40% win rate)...")

        elif preset == "failed":
            # Heavy losses leading to limit breach
            sequence = [-15.0, -20.0, 10.0, -25.0, -30.0, 5.0, -35.0, -40.0, -45.0, -50.0]
            self._test_log("Running failed recovery scenario...")

        else:
            self._test_log(f"Unknown preset: {preset}", "ERROR")
            return

        initial_state = self.recovery_strategy.get_state()

        for i, amount in enumerate(sequence):
            if self.recovery_strategy.get_state().is_complete:
                self._test_log(f"Recovery COMPLETE at step {i + 1}!", "SUCCESS")
                break

            if self.recovery_strategy.should_stop():
                self._test_log(f"Recovery STOPPED at step {i + 1} - limit reached", "WARN")
                break

            margin = self.recovery_strategy.calculate_next_position_size()
            leverage = self.recovery_strategy.calculate_next_leverage()

            self.record_trade(amount)
            trade_type = "WIN" if amount > 0 else "LOSS"
            self._test_log(f"Step {i + 1}: {trade_type} ${abs(amount):.2f} (M:${margin:.2f}, L:{leverage}x)")

        final_state = self.recovery_strategy.get_state()
        self._test_log(f"=== {preset.upper()} Complete ===")
        self._test_log(
            f"  Progress: {initial_state.recovery_percentage:.1f}% -> {final_state.recovery_percentage:.1f}%"
        )
        self._test_log(f"  Status: {'COMPLETE' if final_state.is_complete else 'ACTIVE'}")

        self._update_test_mode_label()

    def _test_run_mode_comparison(self) -> None:
        """Compare all three modes with identical sequence"""
        self._test_log("=== MODE COMPARISON TEST ===")
        self._test_log("Testing identical sequence across fixed/progressive/adaptive modes")

        # Get current config
        try:
            initial_loss = float(self.initial_loss_entry.get())
        except ValueError:
            initial_loss = 100.0

        # Test sequence
        test_sequence = [10.0, 12.0, 15.0, -8.0, 18.0, 20.0, -5.0, 25.0]

        results: Dict[str, Dict[str, Any]] = {}

        for mode in ["fixed", "progressive", "adaptive"]:
            config: RecoveryConfig = {
                "target_profit_per_trade": 5.0,
                "max_recovery_trades": 20,
                "margin_scaling_mode": cast(Literal["fixed", "progressive", "adaptive"], mode),
                "leverage_scaling_mode": cast(Literal["fixed", "progressive", "adaptive"], mode),
                "min_leverage": 2,
                "max_leverage": 10,
                "enable_streak_bonus": mode == "adaptive",
            }

            test_strategy = GradualRecoveryStrategy(initial_loss=initial_loss, config=config)

            margin_history = []
            leverage_history = []

            for amount in test_sequence:
                margin_history.append(test_strategy.calculate_next_position_size())
                leverage_history.append(test_strategy.calculate_next_leverage())

                if amount >= 0:
                    test_strategy.record_profit(amount)
                else:
                    test_strategy.record_loss(abs(amount))

            final_state = test_strategy.get_state()
            results[mode] = {
                "final_pct": final_state.recovery_percentage,
                "margin_range": (min(margin_history), max(margin_history)),
                "leverage_range": (min(leverage_history), max(leverage_history)),
            }

        # Log comparison results
        self._test_log("")
        self._test_log("COMPARISON RESULTS:")
        self._test_log("-" * 50)

        for mode, data in results.items():
            self._test_log(f"  {mode.upper()}:")
            self._test_log(f"    Final Progress: {data['final_pct']:.1f}%")
            self._test_log(f"    Margin Range: ${data['margin_range'][0]:.2f} - ${data['margin_range'][1]:.2f}")
            self._test_log(f"    Leverage Range: {data['leverage_range'][0]}x - {data['leverage_range'][1]}x")

        self._test_log("-" * 50)
        self._test_log("Comparison complete!")

    def _update_test_mode_label(self):
        """Update the test mode label with current modes"""
        if hasattr(self, "test_mode_label"):
            margin_mode = self.margin_mode_var.get()
            leverage_mode = self.leverage_mode_var.get()
            self.test_mode_label.configure(text=f"🧪 Test Mode: {margin_mode}/{leverage_mode}")

    def _apply_preset(self, preset: str):
        """Apply configuration preset"""
        if preset == "conservative":
            self.target_profit_entry.delete(0, "end")
            self.target_profit_entry.insert(0, "3.0")
            self.max_trades_entry.delete(0, "end")
            self.max_trades_entry.insert(0, "33")
            self.margin_mode_var.set("fixed")
            self.leverage_mode_var.set("fixed")
            self.min_leverage_entry.delete(0, "end")
            self.min_leverage_entry.insert(0, "2")
            self.max_leverage_entry.delete(0, "end")
            self.max_leverage_entry.insert(0, "3")
            self.streak_bonus_var.set(False)

        elif preset == "moderate":
            self.target_profit_entry.delete(0, "end")
            self.target_profit_entry.insert(0, "5.0")
            self.max_trades_entry.delete(0, "end")
            self.max_trades_entry.insert(0, "20")
            self.margin_mode_var.set("progressive")
            self.leverage_mode_var.set("progressive")
            self.min_leverage_entry.delete(0, "end")
            self.min_leverage_entry.insert(0, "3")
            self.max_leverage_entry.delete(0, "end")
            self.max_leverage_entry.insert(0, "7")
            self.streak_bonus_var.set(False)

        elif preset == "aggressive":
            self.target_profit_entry.delete(0, "end")
            self.target_profit_entry.insert(0, "8.0")
            self.max_trades_entry.delete(0, "end")
            self.max_trades_entry.insert(0, "13")
            self.margin_mode_var.set("adaptive")
            self.leverage_mode_var.set("adaptive")
            self.min_leverage_entry.delete(0, "end")
            self.min_leverage_entry.insert(0, "5")
            self.max_leverage_entry.delete(0, "end")
            self.max_leverage_entry.insert(0, "15")
            self.streak_bonus_var.set(True)

        self._update_plan_preview()

    def _on_start_recovery(self) -> None:
        """Start new recovery with current config"""
        try:
            initial_loss = float(self.initial_loss_entry.get())

            config: RecoveryConfig = {
                "target_profit_per_trade": float(self.target_profit_entry.get()),
                "max_recovery_trades": int(self.max_trades_entry.get()),
                "margin_scaling_mode": cast(Literal["fixed", "progressive", "adaptive"], self.margin_mode_var.get()),
                "leverage_scaling_mode": cast(
                    Literal["fixed", "progressive", "adaptive"], self.leverage_mode_var.get()
                ),
                "min_leverage": int(self.min_leverage_entry.get()),
                "max_leverage": int(self.max_leverage_entry.get()),
                "enable_streak_bonus": self.streak_bonus_var.get(),
            }

            self.recovery_strategy = GradualRecoveryStrategy(
                initial_loss=initial_loss,
                config=config,
            )

            self._update_status_display()

            if self.on_config_change:
                self.on_config_change("recovery_started", config)

        except ValueError as e:
            print(f"Invalid input: {e}")

    def _on_reset(self):
        """Reset current recovery"""
        if self.recovery_strategy:
            self.recovery_strategy.reset()
            self._update_status_display()

            if self.on_config_change:
                self.on_config_change("recovery_reset", None)

    def _on_stop_recovery(self):
        """Stop current recovery session"""
        self.recovery_strategy = None
        self._update_status_display()
        if self.on_config_change:
            self.on_config_change("recovery_stopped", None)

    def _update_status_display(self):
        """Update status tab with current recovery state"""
        if not self.recovery_strategy:
            # Show empty state, hide active details
            self.active_recovery_frame.pack_forget()
            self.empty_state_widget.pack(fill="both", expand=True, padx=20, pady=20)
            return

        # Hide empty state, show active details
        self.empty_state_widget.pack_forget()
        self.active_recovery_frame.pack(fill="both", expand=True, padx=10, pady=10)

        state = self.recovery_strategy.get_state()

        # Update labels
        self.initial_loss_label.configure(text=f"Initial Loss: ${state.initial_loss:.2f}")
        self.remaining_loss_label.configure(text=f"Remaining Loss: ${state.remaining_loss:.2f}")
        pct = state.recovery_percentage / 100.0
        self.progress_bar_widget.set(min(1.0, max(0.0, pct)))
        self.progress_bar_label.configure(
            text=f"{state.recovery_percentage:.0f}%",
            text_color="#00ff88" if state.recovery_percentage >= 50 else "#ffaa00",
        )
        self.recovery_pct_label.configure(text=f"Recovery: {state.recovery_percentage:.1f}%")
        self.trades_count_label.configure(text=f"Trades: {state.trades_count}")
        self.win_streak_label.configure(text=f"Win Streak: {state.win_streak}")
        self.est_trades_label.configure(text=f"Est. Remaining: {state.estimated_trades_remaining}")

        # Update recommendations
        next_margin = self.recovery_strategy.calculate_next_position_size()
        next_leverage = self.recovery_strategy.calculate_next_leverage()
        self.margin_label.configure(text=f"Margin: ${next_margin:.2f}")
        self.leverage_label.configure(text=f"Leverage: {next_leverage}x")

        # Update status
        if state.is_complete:
            self.status_label.configure(text="✅ Recovery Complete!", text_color="#00ff88")
        elif self.recovery_strategy.should_stop():
            self.status_label.configure(text="⚠️ Limit Reached", text_color="#ff6b6b")
        else:
            self.status_label.configure(text="🔄 In Progress...", text_color="#ffaa00")

    def _update_plan_preview(self) -> None:
        """Update plan preview with current config"""
        try:
            initial_loss = float(self.initial_loss_entry.get())

            config: RecoveryConfig = {
                "target_profit_per_trade": float(self.target_profit_entry.get()),
                "max_recovery_trades": int(self.max_trades_entry.get()),
                "margin_scaling_mode": cast(Literal["fixed", "progressive", "adaptive"], self.margin_mode_var.get()),
                "leverage_scaling_mode": cast(
                    Literal["fixed", "progressive", "adaptive"], self.leverage_mode_var.get()
                ),
                "min_leverage": int(self.min_leverage_entry.get()),
                "max_leverage": int(self.max_leverage_entry.get()),
                "enable_streak_bonus": self.streak_bonus_var.get(),
            }

            plan = create_recovery_plan(initial_loss, config)

            self.plan_estimated_trades_label.configure(text=f"Est. Trades: {plan['estimated_trades_needed']}")
            self.plan_risk_label.configure(text=f"Risk: {plan['risk_assessment']}")

        except ValueError as e:
            print(f"Invalid input: {e}")

    def record_trade(self, profit: float):
        """Record a trade result"""
        if self.recovery_strategy:
            if profit >= 0:
                self.recovery_strategy.record_profit(profit)
            else:
                self.recovery_strategy.record_loss(abs(profit))

            self._update_status_display()

            # Check for milestones
            state = self.recovery_strategy.get_state()
            pct = state.recovery_percentage

            if abs(pct - 25) < 5:
                self._show_alert("📈 25% Recovery Milestone Reached!")
            elif abs(pct - 50) < 5:
                self._show_alert("🎯 50% Recovery Milestone Reached!")
            elif abs(pct - 75) < 5:
                self._show_alert("🏆 75% Recovery Milestone Reached!")
            elif state.is_complete:
                self._show_alert("🎉 Recovery Complete! All losses recovered!")

            if self.recovery_strategy.should_stop() and not state.is_complete:
                self._show_alert("⚠️ Safety limit reached. Consider resetting.")

    def _show_alert(self, message: str):
        """Show alert notification"""
        print(f"ALERT: {message}")
        if self.on_config_change:
            self.on_config_change("recovery_alert", message)

    def get_config(self) -> Dict:
        """Get current configuration"""
        return {
            "enabled": self.recovery_enabled_var.get(),
            "initial_loss": self.initial_loss_entry.get(),
            "target_profit_per_trade": self.target_profit_entry.get(),
            "max_recovery_trades": self.max_trades_entry.get(),
            "margin_scaling_mode": self.margin_mode_var.get(),
            "leverage_scaling_mode": self.leverage_mode_var.get(),
            "min_leverage": self.min_leverage_entry.get(),
            "max_leverage": self.max_leverage_entry.get(),
            "enable_streak_bonus": self.streak_bonus_var.get(),
        }

    def load_config(self, config: Dict):
        """Load configuration from dict (e.g. default from settings for Trading tab)."""
        if not config:
            return
        try:
            if "enabled" in config:
                v = config["enabled"]
                self.recovery_enabled_var.set(v if isinstance(v, bool) else str(v).lower() in ("true", "1", "yes"))
            if "initial_loss" in config:
                self.initial_loss_entry.delete(0, "end")
                self.initial_loss_entry.insert(0, str(config["initial_loss"]))
            if "target_profit_per_trade" in config:
                self.target_profit_entry.delete(0, "end")
                self.target_profit_entry.insert(0, str(config["target_profit_per_trade"]))
            if "max_recovery_trades" in config:
                self.max_trades_entry.delete(0, "end")
                self.max_trades_entry.insert(0, str(config["max_recovery_trades"]))
            if "margin_scaling_mode" in config:
                self.margin_mode_var.set(str(config["margin_scaling_mode"]))
            if "leverage_scaling_mode" in config:
                self.leverage_mode_var.set(str(config["leverage_scaling_mode"]))
            if "min_leverage" in config:
                self.min_leverage_entry.delete(0, "end")
                self.min_leverage_entry.insert(0, str(config["min_leverage"]))
            if "max_leverage" in config:
                self.max_leverage_entry.delete(0, "end")
                self.max_leverage_entry.insert(0, str(config["max_leverage"]))
            if "enable_streak_bonus" in config:
                v = config["enable_streak_bonus"]
                self.streak_bonus_var.set(v if isinstance(v, bool) else str(v).lower() in ("true", "1", "yes"))
        except Exception as e:
            print(f"Error loading recovery config: {e}")
