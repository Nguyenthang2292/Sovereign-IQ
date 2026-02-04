# Gradual Recovery GUI Integration Plan

## Executive Summary

This document outlines the implementation plan for integrating the `gradual_recovery.py` strategy into the auto_trade GUI with three main objectives:

1. **Database Tab Enhancement**: Add test buttons for gradual recovery functionality
2. **Recovery Tab Testing**: Implement fake data generation for testing fixed/adaptive/progressive modes in DRY_RUN
3. **UI Reorganization**: Move recovery configuration into Settings tab for better workflow

---

## Current State Analysis

### Existing Components
- ✅ **RecoveryPanel** (`recovery_panel.py`, 466 lines) - Fully implemented with 3 tabs
  - Status Tab: Progress tracking
  - Config Tab: Strategy configuration with presets
  - History Tab: Placeholder (Coming Soon)
- ✅ **DatabasePanel** (`database_panel.py`, 736 lines) - Comprehensive testing interface
- ✅ **ConfigPanel** (`config_panel.py`, 705 lines) - 5-tab settings interface
- ✅ **GradualRecoveryStrategy** (`gradual_recovery.py`, 260 lines) - Core strategy implementation

### Integration Points
- Main window callback system (`on_config_change`)
- Settings persistence via SettingsManager
- Database operations via `session_scope()`
- Thread-safe UI updates via `_update_queue`

---

## Implementation Plan

### Phase 1: Database Tab Enhancement

#### 1.1 Add Recovery Testing Section

**Location**: `modules/auto_trade/gui/components/database_panel.py`

**New Section** (Insert after Martingale Testing section, ~line 250):

```python
# Recovery Testing Section
recovery_test_frame = self._create_section(
    self.left_scroll,
    "Recovery Testing",
    "Test gradual recovery strategy functionality"
)

# Test buttons in grid layout
btn_create_recovery = ctk.CTkButton(
    recovery_test_frame,
    text="Create Test Recovery",
    command=self._create_test_recovery,
    width=150
)
btn_create_recovery.grid(row=0, column=0, padx=5, pady=5)

btn_simulate_trade = ctk.CTkButton(
    recovery_test_frame,
    text="Simulate Trade Result",
    command=self._simulate_trade_result,
    width=150
)
btn_simulate_trade.grid(row=0, column=1, padx=5, pady=5)

btn_get_recovery_stats = ctk.CTkButton(
    recovery_test_frame,
    text="Get Recovery Stats",
    command=self._get_recovery_stats,
    width=150
)
btn_get_recovery_stats.grid(row=1, column=0, padx=5, pady=5)

btn_reset_recovery = ctk.CTkButton(
    recovery_test_frame,
    text="Reset Recovery",
    command=self._reset_test_recovery,
    width=150,
    fg_color="red"
)
btn_reset_recovery.grid(row=1, column=1, padx=5, pady=5)

# Status label
self.recovery_status_label = ctk.CTkLabel(
    recovery_test_frame,
    text="No active recovery session",
    text_color="gray"
)
self.recovery_status_label.grid(row=2, column=0, columnspan=2, pady=5)
```

#### 1.2 Implement Recovery Test Methods

**New methods in DatabasePanel class**:

```python
def _create_test_recovery(self):
    """Create a test recovery session with random initial loss"""
    try:
        import random
        from modules.auto_trade.strategies.gradual_recovery import (
            GradualRecoveryStrategy,
            RecoveryConfig
        )

        # Random initial loss between $100-$500
        initial_loss = random.uniform(100.0, 500.0)

        config: RecoveryConfig = {
            "target_profit_per_trade": 5.0,
            "max_recovery_trades": 20,
            "max_total_loss": 2.0 * initial_loss,
            "margin_scaling_mode": "fixed",
            "leverage_scaling_mode": "fixed",
            "min_leverage": 2,
            "max_leverage": 10,
            "enable_streak_bonus": False,
        }

        # Store strategy instance
        self.test_recovery = GradualRecoveryStrategy(
            initial_loss=initial_loss,
            config=config,
            database=self.db_manager
        )

        # Update status
        self.recovery_status_label.configure(
            text=f"Recovery created: ${initial_loss:.2f} loss",
            text_color="orange"
        )

        self._add_log(
            f"Created test recovery with ${initial_loss:.2f} initial loss",
            "success"
        )

        # Store in database (optional)
        with self.db_manager.session_scope() as session:
            from modules.auto_trade.database import create_audit_log
            create_audit_log(
                session,
                action="recovery_created",
                description=f"Test recovery: ${initial_loss:.2f}",
                details={"config": config}
            )

    except Exception as e:
        self._add_log(f"Error creating test recovery: {e}", "error")

def _simulate_trade_result(self):
    """Simulate a random trade result (profit or loss)"""
    try:
        import random

        if not hasattr(self, 'test_recovery'):
            self._add_log("No active recovery session. Create one first.", "error")
            return

        # Random: 70% profit, 30% loss
        is_profit = random.random() < 0.7

        if is_profit:
            # Profit: 3-8% of remaining loss
            profit = self.test_recovery._state["remaining_loss"] * random.uniform(0.03, 0.08)
            self.test_recovery.record_profit(profit)
            self._add_log(f"✅ Profit: ${profit:.2f}", "success")
        else:
            # Loss: 2-5% of remaining loss
            loss = self.test_recovery._state["remaining_loss"] * random.uniform(0.02, 0.05)
            self.test_recovery.record_loss(loss)
            self._add_log(f"❌ Loss: ${loss:.2f}", "error")

        # Update status
        state = self.test_recovery.get_state()
        progress = state.recovery_percentage

        if state.is_complete:
            self.recovery_status_label.configure(
                text=f"✅ Recovery Complete! ({progress:.1f}%)",
                text_color="green"
            )
        else:
            self.recovery_status_label.configure(
                text=f"Progress: {progress:.1f}% | Trades: {state.trades_count} | Win Streak: {state.win_streak}",
                text_color="orange"
            )

    except Exception as e:
        self._add_log(f"Error simulating trade: {e}", "error")

def _get_recovery_stats(self):
    """Display current recovery statistics"""
    try:
        if not hasattr(self, 'test_recovery'):
            self._add_log("No active recovery session", "error")
            return

        state = self.test_recovery.get_state()

        stats_text = f"""
Recovery Statistics:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Initial Loss:     ${state.initial_loss:.2f}
Remaining Loss:   ${state.remaining_loss:.2f}
Total Profit:     ${state.total_profit_accumulated:.2f}
Recovery %:       {state.recovery_percentage:.1f}%
Trades Count:     {state.trades_count}
Win Streak:       {state.win_streak}
Est. Remaining:   {state.estimated_trades_remaining} trades
Status:           {'COMPLETE ✅' if state.is_complete else 'ACTIVE 🔄'}

Next Position:    ${self.test_recovery.calculate_next_position_size():.2f}
Next Leverage:    {self.test_recovery.calculate_next_leverage()}x

Progress Bar: {self.test_recovery.progress_bar}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

        self._add_log(stats_text, "info")

    except Exception as e:
        self._add_log(f"Error getting stats: {e}", "error")

def _reset_test_recovery(self):
    """Reset the test recovery session"""
    try:
        if hasattr(self, 'test_recovery'):
            self.test_recovery.reset()
            delattr(self, 'test_recovery')

        self.recovery_status_label.configure(
            text="No active recovery session",
            text_color="gray"
        )

        self._add_log("Test recovery reset", "success")

    except Exception as e:
        self._add_log(f"Error resetting recovery: {e}", "error")
```

#### 1.3 Update Data Viewer

**Modify `_update_data_view()` method** to include Recovery data type:

```python
# Add to data type selector options (line ~400)
data_type_options = ["Orders", "Signals", "Martingale", "Recovery", "Audit Log"]

# Add recovery query case (line ~450)
elif selected_type == "Recovery":
    # Query recovery history from database
    with self.db_manager.session_scope() as session:
        query = session.query(RecoveryHistory).order_by(
            RecoveryHistory.created_at.desc()
        ).limit(page_size).offset(offset)

        headers = ["ID", "Initial Loss", "Final Loss", "Profit", "Trades", "Status", "Date"]
        for record in query:
            row = [
                record.id,
                f"${record.initial_loss:.2f}",
                f"${record.remaining_loss:.2f}",
                f"${record.total_profit:.2f}",
                record.trades_count,
                "Complete" if record.is_complete else "Active",
                record.created_at.strftime("%Y-%m-%d %H:%M")
            ]
            data.append(row)
```

---

### Phase 2: Recovery Tab - Fake Data Testing

#### 2.1 Add Test Mode Toggle

**Location**: `modules/auto_trade/gui/components/recovery_panel.py`

**Modify Config Tab** (insert after line ~150):

```python
# Test Mode Section (DRY_RUN only)
if self.mode == "DRY_RUN":
    test_frame = self._create_section(
        config_container,
        "🧪 Test Mode (DRY_RUN Only)",
        "Generate fake profit/loss sequences for testing"
    )

    # Test mode toggle
    self.test_mode_var = ctk.BooleanVar(value=False)
    test_mode_switch = ctk.CTkSwitch(
        test_frame,
        text="Enable Test Mode",
        variable=self.test_mode_var,
        command=self._on_test_mode_toggle
    )
    test_mode_switch.pack(anchor="w", pady=5)

    # Test controls frame
    self.test_controls_frame = ctk.CTkFrame(test_frame, fg_color="transparent")
    self.test_controls_frame.pack(fill="x", pady=10)

    # Scaling mode selector for testing
    mode_label = ctk.CTkLabel(
        self.test_controls_frame,
        text="Test Scaling Mode:"
    )
    mode_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")

    self.test_mode_selector = ctk.CTkOptionMenu(
        self.test_controls_frame,
        values=["fixed", "progressive", "adaptive"],
        width=150
    )
    self.test_mode_selector.set("fixed")
    self.test_mode_selector.grid(row=0, column=1, padx=5, pady=5)

    # Generate sequence button
    btn_generate = ctk.CTkButton(
        self.test_controls_frame,
        text="Generate Test Sequence",
        command=self._generate_test_sequence,
        width=180
    )
    btn_generate.grid(row=1, column=0, columnspan=2, padx=5, pady=10)

    # Auto-run toggle
    self.auto_run_var = ctk.BooleanVar(value=False)
    auto_run_switch = ctk.CTkSwitch(
        self.test_controls_frame,
        text="Auto-run Sequence (1 trade/sec)",
        variable=self.auto_run_var,
        command=self._toggle_auto_run
    )
    auto_run_switch.grid(row=2, column=0, columnspan=2, padx=5, pady=5)

    # Test results display
    self.test_results_text = ctk.CTkTextbox(
        test_frame,
        height=150,
        wrap="word"
    )
    self.test_results_text.pack(fill="both", expand=True, pady=10)

    # Initially disable test controls
    self._set_test_controls_state("disabled")
```

#### 2.2 Implement Test Sequence Generation

**New methods in RecoveryPanel class**:

```python
def _on_test_mode_toggle(self):
    """Handle test mode toggle"""
    enabled = self.test_mode_var.get()
    state = "normal" if enabled else "disabled"
    self._set_test_controls_state(state)

    if enabled:
        self._add_test_log("Test mode enabled - Generate a sequence to begin")
    else:
        self._add_test_log("Test mode disabled")
        # Stop auto-run if active
        if hasattr(self, 'auto_run_after_id'):
            self.after_cancel(self.auto_run_after_id)

def _set_test_controls_state(self, state: str):
    """Enable/disable test controls"""
    widgets = [
        self.test_mode_selector,
        # Add other widgets
    ]
    for widget in widgets:
        widget.configure(state=state)

def _generate_test_sequence(self):
    """Generate a fake profit/loss sequence for testing"""
    try:
        import random
        from modules.auto_trade.strategies.gradual_recovery import (
            GradualRecoveryStrategy,
            RecoveryConfig
        )

        # Get selected mode
        selected_mode = self.test_mode_selector.get()

        # Create recovery strategy with selected mode
        initial_loss = float(self.loss_entry.get() or "100")

        config: RecoveryConfig = {
            "target_profit_per_trade": float(self.profit_pct_entry.get() or "5"),
            "max_recovery_trades": int(self.max_trades_entry.get() or "20"),
            "max_total_loss": 2.0 * initial_loss,
            "margin_scaling_mode": selected_mode,
            "leverage_scaling_mode": selected_mode,
            "min_leverage": int(self.min_lev_entry.get() or "2"),
            "max_leverage": int(self.max_lev_entry.get() or "10"),
            "enable_streak_bonus": self.streak_bonus_var.get(),
        }

        self.test_strategy = GradualRecoveryStrategy(
            initial_loss=initial_loss,
            config=config
        )

        # Generate realistic trade sequence
        # 70% win rate, varying profit/loss amounts
        self.test_sequence = []
        num_trades = random.randint(10, 25)

        for i in range(num_trades):
            is_win = random.random() < 0.7

            if is_win:
                # Profit: 3-8% of remaining loss
                remaining = self.test_strategy._state["remaining_loss"]
                profit = remaining * random.uniform(0.03, 0.08)
                self.test_sequence.append(("profit", profit))

                # Simulate trade for preview
                self.test_strategy.record_profit(profit)

                # Stop if complete
                if self.test_strategy._state["is_complete"]:
                    break
            else:
                # Loss: 2-5% of remaining loss
                remaining = self.test_strategy._state["remaining_loss"]
                loss = remaining * random.uniform(0.02, 0.05)
                self.test_sequence.append(("loss", loss))
                self.test_strategy.record_loss(loss)

        # Reset strategy for actual playback
        self.test_strategy.reset()
        self.test_strategy = GradualRecoveryStrategy(
            initial_loss=initial_loss,
            config=config
        )

        # Display sequence summary
        wins = sum(1 for t, _ in self.test_sequence if t == "profit")
        losses = len(self.test_sequence) - wins
        total_profit = sum(amt for t, amt in self.test_sequence if t == "profit")
        total_loss = sum(amt for t, amt in self.test_sequence if t == "loss")

        summary = f"""
Test Sequence Generated ({selected_mode.upper()} mode):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total Trades:     {len(self.test_sequence)}
Wins:             {wins} ({wins/len(self.test_sequence)*100:.1f}%)
Losses:           {losses} ({losses/len(self.test_sequence)*100:.1f}%)
Total Profit:     ${total_profit:.2f}
Total Loss:       ${total_loss:.2f}
Net Result:       ${total_profit - total_loss:.2f}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Ready to run! Enable auto-run or click 'Next Trade'
"""

        self._add_test_log(summary)
        self.test_trade_index = 0

        # Enable next trade button
        # (Add button in UI)

    except Exception as e:
        self._add_test_log(f"Error generating sequence: {e}")

def _toggle_auto_run(self):
    """Toggle automatic trade execution"""
    if self.auto_run_var.get():
        if hasattr(self, 'test_sequence') and self.test_sequence:
            self._execute_next_test_trade()
        else:
            self._add_test_log("Generate a test sequence first")
            self.auto_run_var.set(False)
    else:
        # Stop auto-run
        if hasattr(self, 'auto_run_after_id'):
            self.after_cancel(self.auto_run_after_id)

def _execute_next_test_trade(self):
    """Execute next trade in test sequence"""
    try:
        if not hasattr(self, 'test_sequence') or not self.test_sequence:
            return

        if self.test_trade_index >= len(self.test_sequence):
            self._add_test_log("✅ Test sequence complete!")
            self.auto_run_var.set(False)
            return

        # Get next trade
        trade_type, amount = self.test_sequence[self.test_trade_index]

        # Execute trade
        if trade_type == "profit":
            self.test_strategy.record_profit(amount)
            icon = "✅"
            color = "green"
        else:
            self.test_strategy.record_loss(amount)
            icon = "❌"
            color = "red"

        # Get updated state
        state = self.test_strategy.get_state()

        # Calculate position size and leverage for this trade
        position_size = self.test_strategy.calculate_next_position_size()
        leverage = self.test_strategy.calculate_next_leverage()

        # Log trade with scaling details
        log_msg = f"""
Trade #{self.test_trade_index + 1}: {icon} {trade_type.upper()} ${amount:.2f}
Position Size: ${position_size:.2f} | Leverage: {leverage}x
Progress: {state.recovery_percentage:.1f}% | Remaining: ${state.remaining_loss:.2f}
Win Streak: {state.win_streak} | Trades: {state.trades_count}
{state.progress_bar}
"""
        self._add_test_log(log_msg)

        # Update UI status display
        self._update_status_display(state)

        self.test_trade_index += 1

        # Continue if auto-run enabled
        if self.auto_run_var.get():
            self.auto_run_after_id = self.after(1000, self._execute_next_test_trade)

    except Exception as e:
        self._add_test_log(f"Error executing trade: {e}")

def _add_test_log(self, message: str):
    """Add message to test results textbox"""
    self.test_results_text.insert("end", message + "\n")
    self.test_results_text.see("end")

def _update_status_display(self, state):
    """Update the status tab with current recovery state"""
    # Update existing status labels
    if hasattr(self, 'initial_loss_label'):
        self.initial_loss_label.configure(text=f"${state.initial_loss:.2f}")
    if hasattr(self, 'remaining_loss_label'):
        self.remaining_loss_label.configure(text=f"${state.remaining_loss:.2f}")
    if hasattr(self, 'recovery_pct_label'):
        self.recovery_pct_label.configure(text=f"{state.recovery_percentage:.1f}%")
    if hasattr(self, 'trades_count_label'):
        self.trades_count_label.configure(text=str(state.trades_count))
    if hasattr(self, 'win_streak_label'):
        self.win_streak_label.configure(text=str(state.win_streak))
    if hasattr(self, 'progress_bar_label'):
        self.progress_bar_label.configure(text=state.progress_bar)
```

#### 2.3 Add Next Trade Button

```python
# In test_controls_frame (after auto-run switch)
btn_next_trade = ctk.CTkButton(
    self.test_controls_frame,
    text="Execute Next Trade",
    command=lambda: self._execute_next_test_trade() if hasattr(self, 'test_sequence') else None,
    width=180
)
btn_next_trade.grid(row=3, column=0, columnspan=2, padx=5, pady=5)
```

---

### Phase 3: Move Recovery to Settings Tab

#### 3.1 Reorganize ConfigPanel

**Modify**: `modules/auto_trade/gui/components/config_panel.py`

**Add 6th Tab - Recovery Strategy**:

```python
# In __init__ method, after creating tabview (line ~75)
self.recovery_tab = self.tabview.add("Recovery")
self._create_recovery_tab()

# New method
def _create_recovery_tab(self):
    """Create recovery strategy configuration tab"""
    container = ctk.CTkScrollableFrame(
        self.recovery_tab,
        fg_color="transparent"
    )
    container.pack(fill="both", expand=True, padx=20, pady=20)

    # Recovery Strategy Section
    recovery_frame = self._create_section(
        container,
        "Gradual Recovery Strategy",
        "Configure automatic loss recovery settings"
    )

    # Enable recovery checkbox
    self.enable_recovery_var = ctk.BooleanVar(value=False)
    enable_switch = ctk.CTkSwitch(
        recovery_frame,
        text="Enable Auto Recovery",
        variable=self.enable_recovery_var,
        command=self._on_recovery_toggle
    )
    enable_switch.pack(anchor="w", pady=10)

    # Initial loss threshold
    loss_frame = ctk.CTkFrame(recovery_frame, fg_color="transparent")
    loss_frame.pack(fill="x", pady=5)

    ctk.CTkLabel(
        loss_frame,
        text="Trigger Loss Amount ($):"
    ).pack(side="left", padx=(0, 10))

    self.recovery_trigger_entry = ctk.CTkEntry(
        loss_frame,
        placeholder_text="100.00",
        width=120
    )
    self.recovery_trigger_entry.pack(side="left")

    ctk.CTkLabel(
        loss_frame,
        text="Start recovery when loss exceeds this amount",
        text_color="gray",
        font=("Arial", 11)
    ).pack(side="left", padx=10)

    # Target profit per trade
    profit_frame = ctk.CTkFrame(recovery_frame, fg_color="transparent")
    profit_frame.pack(fill="x", pady=5)

    ctk.CTkLabel(
        profit_frame,
        text="Target Profit per Trade (%):"
    ).pack(side="left", padx=(0, 10))

    self.recovery_profit_entry = ctk.CTkEntry(
        profit_frame,
        placeholder_text="5.0",
        width=120
    )
    self.recovery_profit_entry.pack(side="left")

    # Scaling mode selectors
    scaling_frame = self._create_section(
        container,
        "Scaling Modes",
        "Configure position size and leverage scaling"
    )

    # Margin scaling
    margin_row = ctk.CTkFrame(scaling_frame, fg_color="transparent")
    margin_row.pack(fill="x", pady=5)

    ctk.CTkLabel(
        margin_row,
        text="Margin Scaling:",
        width=150
    ).pack(side="left")

    self.margin_scaling_menu = ctk.CTkOptionMenu(
        margin_row,
        values=["fixed", "progressive", "adaptive"],
        width=150
    )
    self.margin_scaling_menu.set("fixed")
    self.margin_scaling_menu.pack(side="left", padx=10)

    # Leverage scaling
    leverage_row = ctk.CTkFrame(scaling_frame, fg_color="transparent")
    leverage_row.pack(fill="x", pady=5)

    ctk.CTkLabel(
        leverage_row,
        text="Leverage Scaling:",
        width=150
    ).pack(side="left")

    self.leverage_scaling_menu = ctk.CTkOptionMenu(
        leverage_row,
        values=["fixed", "progressive", "adaptive"],
        width=150
    )
    self.leverage_scaling_menu.set("fixed")
    self.leverage_scaling_menu.pack(side="left", padx=10)

    # Leverage range
    lev_range_frame = self._create_section(
        container,
        "Leverage Range",
        "Min and max leverage for recovery trades"
    )

    range_row = ctk.CTkFrame(lev_range_frame, fg_color="transparent")
    range_row.pack(fill="x", pady=5)

    ctk.CTkLabel(range_row, text="Min:").pack(side="left", padx=(0, 5))
    self.min_leverage_entry = ctk.CTkEntry(range_row, width=80, placeholder_text="2")
    self.min_leverage_entry.pack(side="left", padx=5)

    ctk.CTkLabel(range_row, text="Max:").pack(side="left", padx=(20, 5))
    self.max_leverage_entry = ctk.CTkEntry(range_row, width=80, placeholder_text="10")
    self.max_leverage_entry.pack(side="left", padx=5)

    # Advanced options
    advanced_frame = self._create_section(
        container,
        "Advanced Options",
        "Fine-tune recovery behavior"
    )

    self.streak_bonus_var = ctk.BooleanVar(value=False)
    streak_switch = ctk.CTkSwitch(
        advanced_frame,
        text="Enable Win Streak Bonus",
        variable=self.streak_bonus_var
    )
    streak_switch.pack(anchor="w", pady=5)

    # Max trades
    max_trades_row = ctk.CTkFrame(advanced_frame, fg_color="transparent")
    max_trades_row.pack(fill="x", pady=5)

    ctk.CTkLabel(
        max_trades_row,
        text="Max Recovery Trades:"
    ).pack(side="left", padx=(0, 10))

    self.max_recovery_trades_entry = ctk.CTkEntry(
        max_trades_row,
        width=100,
        placeholder_text="20"
    )
    self.max_recovery_trades_entry.pack(side="left")

    # Presets section
    presets_frame = self._create_section(
        container,
        "Quick Presets",
        "Load predefined configurations"
    )

    preset_buttons_frame = ctk.CTkFrame(presets_frame, fg_color="transparent")
    preset_buttons_frame.pack(fill="x", pady=10)

    btn_conservative = ctk.CTkButton(
        preset_buttons_frame,
        text="Conservative",
        command=lambda: self._load_recovery_preset("conservative"),
        width=130
    )
    btn_conservative.grid(row=0, column=0, padx=5, pady=5)

    btn_moderate = ctk.CTkButton(
        preset_buttons_frame,
        text="Moderate",
        command=lambda: self._load_recovery_preset("moderate"),
        width=130
    )
    btn_moderate.grid(row=0, column=1, padx=5, pady=5)

    btn_aggressive = ctk.CTkButton(
        preset_buttons_frame,
        text="Aggressive",
        command=lambda: self._load_recovery_preset("aggressive"),
        width=130
    )
    btn_aggressive.grid(row=0, column=2, padx=5, pady=5)

def _on_recovery_toggle(self):
    """Handle recovery enable/disable"""
    enabled = self.enable_recovery_var.get()
    # Trigger settings change callback
    if self.on_settings_change:
        self.on_settings_change({
            "recovery": {
                "enabled": enabled
            }
        })

def _load_recovery_preset(self, preset: str):
    """Load a recovery preset configuration"""
    presets = {
        "conservative": {
            "trigger_loss": 50.0,
            "target_profit": 3.0,
            "margin_scaling": "fixed",
            "leverage_scaling": "fixed",
            "min_leverage": 2,
            "max_leverage": 5,
            "streak_bonus": False,
            "max_trades": 30
        },
        "moderate": {
            "trigger_loss": 100.0,
            "target_profit": 5.0,
            "margin_scaling": "progressive",
            "leverage_scaling": "progressive",
            "min_leverage": 2,
            "max_leverage": 10,
            "streak_bonus": True,
            "max_trades": 20
        },
        "aggressive": {
            "trigger_loss": 200.0,
            "target_profit": 8.0,
            "margin_scaling": "adaptive",
            "leverage_scaling": "adaptive",
            "min_leverage": 5,
            "max_leverage": 20,
            "streak_bonus": True,
            "max_trades": 15
        }
    }

    config = presets[preset]

    # Apply to UI
    self.recovery_trigger_entry.delete(0, "end")
    self.recovery_trigger_entry.insert(0, str(config["trigger_loss"]))

    self.recovery_profit_entry.delete(0, "end")
    self.recovery_profit_entry.insert(0, str(config["target_profit"]))

    self.margin_scaling_menu.set(config["margin_scaling"])
    self.leverage_scaling_menu.set(config["leverage_scaling"])

    self.min_leverage_entry.delete(0, "end")
    self.min_leverage_entry.insert(0, str(config["min_leverage"]))

    self.max_leverage_entry.delete(0, "end")
    self.max_leverage_entry.insert(0, str(config["max_leverage"]))

    self.streak_bonus_var.set(config["streak_bonus"])

    self.max_recovery_trades_entry.delete(0, "end")
    self.max_recovery_trades_entry.insert(0, str(config["max_trades"]))
```

#### 3.2 Update get_settings() Method

```python
# Add to get_settings() method (line ~650)
"recovery": {
    "enabled": self.enable_recovery_var.get(),
    "trigger_loss": float(self.recovery_trigger_entry.get() or "100"),
    "target_profit_pct": float(self.recovery_profit_entry.get() or "5"),
    "margin_scaling_mode": self.margin_scaling_menu.get(),
    "leverage_scaling_mode": self.leverage_scaling_menu.get(),
    "min_leverage": int(self.min_leverage_entry.get() or "2"),
    "max_leverage": int(self.max_leverage_entry.get() or "10"),
    "enable_streak_bonus": self.streak_bonus_var.get(),
    "max_recovery_trades": int(self.max_recovery_trades_entry.get() or "20"),
}
```

#### 3.3 Remove Recovery Tab from Main Window

**Modify**: `modules/auto_trade/gui/main_window.py`

```python
# Comment out or remove lines 79-80
# self.recovery_tab = self.tabview.add("Recovery")

# Comment out lines 157-162 (RecoveryPanel initialization)
# self.recovery_panel = RecoveryPanel(...)

# Remove recovery callback registration (lines 698-717)
```

#### 3.4 Add Recovery Status to Dashboard

**Add to Dashboard tab** (show active recovery status):

```python
# In _create_dashboard_tab method, add recovery status widget
if self.settings.get("recovery", {}).get("enabled"):
    recovery_status_frame = ctk.CTkFrame(left_panel)
    recovery_status_frame.pack(fill="x", padx=10, pady=10)

    ctk.CTkLabel(
        recovery_status_frame,
        text="🔄 Recovery Active",
        font=("Arial", 14, "bold")
    ).pack(pady=5)

    # Show current recovery progress
    # (Pull from recovery strategy instance)
```

---

## Testing Strategy

### Test Scenarios

#### 1. Fixed Mode Testing
- Initial loss: $100
- Target profit: 5%
- Verify position size stays constant: $10
- Verify leverage stays constant: 2x

#### 2. Progressive Mode Testing
- Initial loss: $100
- Target profit: 5%
- Verify position size increases as recovery progresses
- Verify leverage scales from min (2x) to max (10x)

#### 3. Adaptive Mode Testing
- Initial loss: $100
- Target profit: 5%
- Verify position size adjusts with win streak
- Verify leverage bonus on win streaks
- Verify reset on losses

#### 4. Edge Cases
- Complete recovery (remaining loss = 0)
- Max trades reached
- Max total loss exceeded
- Win streak behavior
- Loss streak recovery

---

## Database Schema Extension

### New Table: recovery_history

```sql
CREATE TABLE recovery_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    initial_loss REAL NOT NULL,
    remaining_loss REAL NOT NULL,
    total_profit_accumulated REAL NOT NULL,
    recovery_percentage REAL NOT NULL,
    trades_count INTEGER NOT NULL,
    win_streak INTEGER NOT NULL,
    is_complete BOOLEAN NOT NULL,
    margin_scaling_mode TEXT NOT NULL,
    leverage_scaling_mode TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    config JSON
);

CREATE TABLE recovery_trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    recovery_id INTEGER NOT NULL,
    trade_number INTEGER NOT NULL,
    trade_type TEXT NOT NULL, -- 'profit' or 'loss'
    amount REAL NOT NULL,
    position_size REAL NOT NULL,
    leverage INTEGER NOT NULL,
    remaining_loss_before REAL NOT NULL,
    remaining_loss_after REAL NOT NULL,
    win_streak INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (recovery_id) REFERENCES recovery_history(id)
);
```

---

## Implementation Checklist

### Phase 1: Database Tab
- [ ] Add Recovery Testing section to DatabasePanel
- [ ] Implement `_create_test_recovery()` method
- [ ] Implement `_simulate_trade_result()` method
- [ ] Implement `_get_recovery_stats()` method
- [ ] Implement `_reset_test_recovery()` method
- [ ] Add Recovery data type to data viewer
- [ ] Test recovery creation and simulation

### Phase 2: Test Mode
- [ ] Add test mode toggle to RecoveryPanel (DRY_RUN only)
- [ ] Implement test sequence generation
- [ ] Add scaling mode selector
- [ ] Implement auto-run functionality
- [ ] Add next trade button
- [ ] Implement status display updates
- [ ] Test all three modes (fixed, progressive, adaptive)
- [ ] Verify UI updates correctly

### Phase 3: Settings Integration
- [ ] Add Recovery tab to ConfigPanel
- [ ] Implement recovery configuration UI
- [ ] Add preset buttons (Conservative, Moderate, Aggressive)
- [ ] Update `get_settings()` method
- [ ] Update `load_settings()` method
- [ ] Remove Recovery tab from main window
- [ ] Add recovery status to Dashboard
- [ ] Update settings persistence

### Phase 4: Database Schema
- [ ] Create `recovery_history` table
- [ ] Create `recovery_trades` table
- [ ] Add database migration script
- [ ] Update DatabaseManager with recovery queries
- [ ] Test data persistence

### Phase 5: Testing & Documentation
- [ ] Test fixed mode with fake data
- [ ] Test progressive mode with fake data
- [ ] Test adaptive mode with fake data
- [ ] Test edge cases (completion, max trades, max loss)
- [ ] Update user documentation
- [ ] Create video tutorial (optional)

---

## Expected Outcomes

### User Experience Improvements
1. **Centralized Configuration**: All recovery settings in one place (Settings tab)
2. **Easy Testing**: Test mode allows experimentation without real trades
3. **Visual Feedback**: Progress bars, charts, and real-time updates
4. **Database Insights**: Query and analyze recovery performance
5. **Presets**: Quick-start with predefined strategies

### Developer Benefits
1. **Modular Design**: Clear separation between UI, strategy, and database
2. **Testability**: Fake data generation for comprehensive testing
3. **Extensibility**: Easy to add new scaling modes or features
4. **Documentation**: Clear integration patterns for future features

---

## Timeline Estimate

| Phase | Estimated Time | Priority |
|-------|----------------|----------|
| Phase 1: Database Tab | 3-4 hours | High |
| Phase 2: Test Mode | 4-5 hours | High |
| Phase 3: Settings Integration | 3-4 hours | Medium |
| Phase 4: Database Schema | 2-3 hours | Medium |
| Phase 5: Testing & Docs | 3-4 hours | High |
| **TOTAL** | **15-20 hours** | - |

---

## Risk Assessment

### Potential Issues
1. **UI Complexity**: Adding test mode increases UI complexity
   - *Mitigation*: Hide test controls when not in DRY_RUN mode

2. **State Management**: Managing both real and test recovery state
   - *Mitigation*: Use separate instances for test vs production

3. **Database Performance**: Additional tables may slow queries
   - *Mitigation*: Add indexes on foreign keys and timestamps

4. **User Confusion**: Too many options may overwhelm users
   - *Mitigation*: Provide clear presets and tooltips

---

## Conclusion

This plan provides a comprehensive approach to integrating gradual recovery into the auto_trade GUI with:

1. ✅ Database tab enhancement for testing
2. ✅ Fake data generation for mode validation
3. ✅ Settings tab reorganization for better UX
4. ✅ Comprehensive testing strategy
5. ✅ Clear implementation roadmap

The modular design ensures easy maintenance and future extensibility while providing users with powerful recovery tools and developers with robust testing capabilities.
