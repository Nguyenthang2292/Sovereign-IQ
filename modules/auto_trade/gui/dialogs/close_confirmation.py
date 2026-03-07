import json
import os
from typing import Callable, Dict, Optional

import customtkinter as ctk

from modules.auto_trade.gui.utils.colors import Colors
from modules.auto_trade.gui.utils.fonts import Fonts
from modules.auto_trade.gui.utils.windows_utils import apply_dark_titlebar
from modules.common.ui.logging import log_warn


class CloseConfirmationDialog(ctk.CTkToplevel):
    """
    Position Close Confirmation Dialog
    Displays comprehensive trade summary and requires explicit confirmation
    """

    CONFIRMATION_SETTINGS_FILE = "close_confirmation_settings.json"

    def __init__(
        self,
        parent,
        position: Dict,
        action_type: str = "close",
        close_details: Optional[Dict] = None,
        on_confirm: Optional[Callable] = None,
        on_cancel: Optional[Callable] = None,
    ):
        super().__init__(parent)

        apply_dark_titlebar(self)

        self.position = position
        self.action_type = action_type
        self.close_details = close_details or {}
        self.on_confirm = on_confirm
        self.on_cancel = on_cancel

        self.confirm_count = 0
        self.required_confirms = 2
        self.skip_confirmation = False

        # Load settings
        self._load_settings()

        # Check if skip confirmation
        if self._should_skip_confirmation():
            self._execute_confirmation()
            return

        # Window configuration
        self.title(f"Confirm {action_type.title()}")
        self.geometry("550x450")
        self.transient(parent)
        self.grab_set()

        # Center dialog
        self.update_idletasks()
        x = (self.winfo_screenwidth() // 2) - (550 // 2)
        y = (self.winfo_screenheight() // 2) - (450 // 2)
        self.geometry(f"550x450+{x}+{y}")

        # Setup UI
        self._create_ui()

        # Bind close event
        self.protocol("WM_DELETE_WINDOW", self._on_window_close)

    def _load_settings(self):
        """Load confirmation settings from file"""
        try:
            if os.path.exists(self.CONFIRMATION_SETTINGS_FILE):
                with open(self.CONFIRMATION_SETTINGS_FILE, "r") as f:
                    settings = json.load(f)
                    self.skip_confirmation = settings.get("skip_confirmation", False)
                    self.required_confirms = settings.get("required_confirms", 2)
        except Exception as e:
            log_warn("Error loading confirmation settings: %s", e)

    def _save_settings(self):
        """Save confirmation settings to file"""
        try:
            settings = {"skip_confirmation": self.skip_confirmation, "required_confirms": self.required_confirms}
            with open(self.CONFIRMATION_SETTINGS_FILE, "w") as f:
                json.dump(settings, f)
        except Exception as e:
            log_warn("Error saving confirmation settings: %s", e)

    def _should_skip_confirmation(self) -> bool:
        """Check if confirmation should be skipped"""
        # Never skip for critical actions
        if self.action_type in ["close_position", "partial_close"]:
            return False
        return self.skip_confirmation

    def _create_ui(self):
        """Create dialog UI"""
        # Main content frame
        content_frame = ctk.CTkFrame(self, fg_color=Colors.CARD_DIALOG, corner_radius=0)
        content_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # Header with warning icon
        self._create_header(content_frame)

        # Trade summary
        self._create_trade_summary(content_frame)

        # Estimated P&L section
        self._create_pnl_summary(content_frame)

        # Final return section
        self._create_final_return(content_frame)

        # Confirmation controls
        self._create_confirmation_controls(content_frame)

        # Action buttons
        self._create_action_buttons(content_frame)

    def _create_header(self, parent):
        """Create dialog header"""
        header = ctk.CTkFrame(parent, fg_color=Colors.TRANSPARENT)
        header.pack(fill="x", pady=(0, 15))

        # Warning icon
        icon_label = ctk.CTkLabel(header, text="⚠️", font=(Fonts.FAMILY, 32), text_color=Colors.BTN_WARNING)
        icon_label.pack(side="left", padx=(0, 15))

        # Title
        title_text = f"Confirm {self._get_action_title()}"
        title_label = ctk.CTkLabel(header, text=title_text, font=(Fonts.FAMILY, 18, "bold"), text_color=Colors.LOSS)
        title_label.pack(side="left")

    def _get_action_title(self) -> str:
        """Get formatted action title"""
        action_titles = {
            "close_position": "Position Close",
            "partial_close": "Partial Close",
            "modify_tp_sl": "TP/SL Modification",
            "breakeven": "Breakeven",
            "cancel_orders": "Cancel Orders",
        }
        return action_titles.get(self.action_type, "Action")

    def _create_trade_summary(self, parent):
        """Create trade summary section"""
        summary_frame = ctk.CTkFrame(parent, fg_color=Colors.CARD_MUTED, corner_radius=0)
        summary_frame.pack(fill="x", pady=(0, 10))

        # Title
        title = ctk.CTkLabel(summary_frame, text="📋 Trade Summary", font=Fonts.H3)
        title.pack(pady=(10, 8))

        # Summary items
        self._create_summary_items(summary_frame)

    def _create_summary_items(self, parent):
        """Create summary item rows"""
        items_frame = ctk.CTkFrame(parent, fg_color=Colors.TRANSPARENT)
        items_frame.pack(fill="x", padx=10, pady=(0, 10))

        # Define summary items based on action type
        items = self._get_summary_items()

        for label, value, color in items:
            row = ctk.CTkFrame(items_frame, fg_color=Colors.TRANSPARENT)
            row.pack(fill="x", pady=3)

            label_widget = ctk.CTkLabel(row, text=label, font=Fonts.BODY, text_color=Colors.TEXT_MUTED, width=120, anchor="w")
            label_widget.pack(side="left")

            value_widget = ctk.CTkLabel(row, text=value, font=(Fonts.FAMILY, 11, "bold"), text_color=color)
            value_widget.pack(side="right")

    def _get_summary_items(self) -> list:
        """Get summary items based on action type"""
        symbol = self.position.get("symbol", "N/A")
        side = self.position.get("side", "LONG")
        side_color = Colors.PROFIT if side == "LONG" else Colors.LOSS

        base_items = [("Symbol:", symbol, Colors.WHITE), ("Side:", side, side_color)]

        if self.action_type == "close_position":
            size = self.position.get("size", 0)
            base_items.append(("Size:", f"{size:.4f}", Colors.WHITE))

            close_type = self.close_details.get("type", "market").title()
            base_items.append(("Type:", close_type, Colors.INFO))

            if close_type == "Limit":
                limit_price = self.close_details.get("limit_price", 0)
                base_items.append(("Limit Price:", f"${limit_price:,.2f}", Colors.INFO))

        elif self.action_type == "partial_close":
            total_size = self.position.get("size", 0)
            close_pct = self.close_details.get("percentage", 0)
            close_size = total_size * (close_pct / 100)
            remaining_size = total_size - close_size

            base_items.extend(
                [
                    ("Total Size:", f"{total_size:.4f}", Colors.WHITE),
                    ("Close:", f"{close_size:.4f} ({close_pct}%)", Colors.BTN_WARNING),
                    ("Remaining:", f"{remaining_size:.4f}", Colors.INFO),
                ]
            )

        elif self.action_type in ["modify_tp_sl", "breakeven"]:
            entry_price = self.position.get("entry_price", 0)
            base_items.append(("Entry Price:", f"${entry_price:,.2f}", Colors.WHITE))

            if self.action_type == "breakeven":
                base_items.append(("New SL:", f"${entry_price:,.2f}", Colors.PROFIT))
            else:
                new_tp = self.close_details.get("take_profit", 0)
                new_sl = self.close_details.get("stop_loss", 0)

                if new_tp > 0:
                    base_items.append(("New TP:", f"${new_tp:,.2f}", Colors.PROFIT))
                if new_sl > 0:
                    base_items.append(("New SL:", f"${new_sl:,.2f}", Colors.LOSS))

        return base_items

    def _create_pnl_summary(self, parent):
        """Create P&L summary section"""
        pnl_frame = ctk.CTkFrame(parent, fg_color=Colors.CARD_MUTED, corner_radius=0)
        pnl_frame.pack(fill="x", pady=(0, 10))

        # Title
        title = ctk.CTkLabel(pnl_frame, text="💰 Estimated P&L", font=Fonts.H3)
        title.pack(pady=(10, 8))

        # P&L items
        items_frame = ctk.CTkFrame(pnl_frame, fg_color=Colors.TRANSPARENT)
        items_frame.pack(fill="x", padx=10, pady=(0, 10))

        # Unrealized P&L
        unrealized_pnl = self.position.get("unrealized_pnl", 0)
        if self.action_type == "partial_close":
            close_pct = self.close_details.get("percentage", 0)
            unrealized_pnl = unrealized_pnl * (close_pct / 100)

        pnl_color = Colors.PROFIT if unrealized_pnl >= 0 else Colors.LOSS
        pnl_sign = "+" if unrealized_pnl >= 0 else ""

        # Calculate estimated P&L
        row1 = ctk.CTkFrame(items_frame, fg_color=Colors.TRANSPARENT)
        row1.pack(fill="x", pady=3)

        label1 = ctk.CTkLabel(row1, text="Estimated P&L:", font=Fonts.BODY, text_color=Colors.TEXT_MUTED, width=120, anchor="w")
        label1.pack(side="left")

        value1 = ctk.CTkLabel(
            row1, text=f"{pnl_sign}${unrealized_pnl:,.2f}", font=Fonts.H2, text_color=pnl_color
        )
        value1.pack(side="right")

        # ROI (if applicable)
        if unrealized_pnl != 0 and self.action_type in ["close_position", "partial_close"]:
            margin = self.position.get("margin_used", 1)
            roi_pct = (unrealized_pnl / margin) * 100
            roi_sign = "+" if roi_pct >= 0 else ""
            roi_color = Colors.PROFIT if roi_pct >= 0 else Colors.LOSS

            row2 = ctk.CTkFrame(items_frame, fg_color=Colors.TRANSPARENT)
            row2.pack(fill="x", pady=3)

            label2 = ctk.CTkLabel(
                row2, text="Estimated ROI:", font=Fonts.BODY, text_color=Colors.TEXT_MUTED, width=120, anchor="w"
            )
            label2.pack(side="left")

            value2 = ctk.CTkLabel(
                row2, text=f"{roi_sign}{roi_pct:.2f}%", font=Fonts.H3, text_color=roi_color
            )
            value2.pack(side="right")

    def _create_final_return(self, parent):
        """Create final return section with fees"""
        return_frame = ctk.CTkFrame(parent, fg_color=Colors.CARD_MUTED, corner_radius=0)
        return_frame.pack(fill="x", pady=(0, 10))

        # Title
        title = ctk.CTkLabel(return_frame, text="📦 Final Return", font=Fonts.H3)
        title.pack(pady=(10, 8))

        # Return items
        items_frame = ctk.CTkFrame(return_frame, fg_color=Colors.TRANSPARENT)
        items_frame.pack(fill="x", padx=10, pady=(0, 10))

        # Calculate final return
        unrealized_pnl = self.position.get("unrealized_pnl", 0)
        if self.action_type == "partial_close":
            close_pct = self.close_details.get("percentage", 0)
            unrealized_pnl = unrealized_pnl * (close_pct / 100)

        # Estimate fees (roughly 0.1% for most exchanges)
        estimated_fees = abs(unrealized_pnl) * 0.001
        final_return = unrealized_pnl - estimated_fees

        # Display items
        # P&L before fees
        row1 = ctk.CTkFrame(items_frame, fg_color=Colors.TRANSPARENT)
        row1.pack(fill="x", pady=3)

        label1 = ctk.CTkLabel(row1, text="P&L:", font=Fonts.BODY, text_color=Colors.TEXT_MUTED, width=120, anchor="w")
        label1.pack(side="left")

        pnl_sign = "+" if unrealized_pnl >= 0 else ""
        pnl_color = Colors.PROFIT if unrealized_pnl >= 0 else Colors.LOSS
        value1 = ctk.CTkLabel(row1, text=f"{pnl_sign}${unrealized_pnl:,.2f}", font=Fonts.BODY, text_color=pnl_color)
        value1.pack(side="right")

        # Estimated fees
        row2 = ctk.CTkFrame(items_frame, fg_color=Colors.TRANSPARENT)
        row2.pack(fill="x", pady=3)

        label2 = ctk.CTkLabel(row2, text="Est. Fees:", font=Fonts.BODY, text_color=Colors.TEXT_MUTED, width=120, anchor="w")
        label2.pack(side="left")

        value2 = ctk.CTkLabel(row2, text=f"-${estimated_fees:,.2f}", font=Fonts.BODY, text_color=Colors.LOSS)
        value2.pack(side="right")

        # Final return (larger font)
        row3 = ctk.CTkFrame(items_frame, fg_color=Colors.TRANSPARENT)
        row3.pack(fill="x", pady=(10, 0))

        label3 = ctk.CTkLabel(
            row3, text="Final Return:", font=Fonts.H3, text_color=Colors.TEXT_MUTED, width=120, anchor="w"
        )
        label3.pack(side="left")

        final_sign = "+" if final_return >= 0 else ""
        final_color = Colors.PROFIT if final_return >= 0 else Colors.LOSS
        value3 = ctk.CTkLabel(
            row3, text=f"{final_sign}${final_return:,.2f}", font=Fonts.H1, text_color=final_color
        )
        value3.pack(side="right")

    def _create_confirmation_controls(self, parent):
        """Create confirmation controls section"""
        controls_frame = ctk.CTkFrame(parent, fg_color=Colors.TRANSPARENT)
        controls_frame.pack(fill="x", pady=(0, 10))

        # Confirmation counter
        self.confirm_label = ctk.CTkLabel(
            controls_frame,
            text=f"Press Confirm {self.required_confirms} times to proceed",
            font=Fonts.INPUT,
            text_color=Colors.BTN_WARNING,
        )
        self.confirm_label.pack(pady=(0, 10))

        # Progress indicator
        self.progress_bar = ctk.CTkProgressBar(controls_frame, width=300, height=10)
        self.progress_bar.set(0)
        self.progress_bar.pack()

        # Don't ask again checkbox
        if self.action_type not in ["close_position", "partial_close"]:
            skip_frame = ctk.CTkFrame(controls_frame, fg_color=Colors.TRANSPARENT)
            skip_frame.pack(fill="x", pady=(15, 0))

            self.skip_var = ctk.BooleanVar(value=False)

            skip_checkbox = ctk.CTkCheckBox(
                skip_frame,
                text="Don't ask again for this action",
                variable=self.skip_var,
                command=self._on_skip_changed,
                font=Fonts.SMALL,
                text_color=Colors.TEXT_MUTED,
            )
            skip_checkbox.pack(anchor="center")

    def _on_skip_changed(self):
        """Handle skip checkbox change"""
        self.skip_confirmation = self.skip_var.get()

    def _create_action_buttons(self, parent):
        """Create action buttons"""
        btn_frame = ctk.CTkFrame(parent, fg_color=Colors.TRANSPARENT)
        btn_frame.pack(fill="x", pady=(0, 10))

        # Confirm button
        self.confirm_btn = ctk.CTkButton(
            btn_frame,
            text=f"Confirm ({self.required_confirms - self.confirm_count})",
            font=Fonts.BUTTON_SM,
            height=40,
            fg_color=Colors.BTN_DANGER,
            hover_color=Colors.BTN_DANGER_HOVER,
            command=self._on_confirm_click,
        )
        self.confirm_btn.pack(side="left", fill="x", expand=True, padx=(0, 10))

        # Cancel button
        cancel_btn = ctk.CTkButton(
            btn_frame,
            text="❌ CANCEL",
            font=Fonts.BUTTON_SM,
            height=40,
            fg_color=Colors.TEXT_MUTED,
            hover_color=Colors.TEXT_MUTED_DARK,
            command=self._on_cancel,
        )
        cancel_btn.pack(side="left", fill="x", expand=True, padx=(10, 0))

    def _on_confirm_click(self):
        """Handle confirm button click"""
        self.confirm_count += 1
        remaining = self.required_confirms - self.confirm_count

        # Update progress
        progress = self.confirm_count / self.required_confirms
        self.progress_bar.set(progress)

        if remaining > 0:
            self.confirm_btn.configure(text=f"Confirm ({remaining})")
        else:
            # All confirmations done
            self._execute_confirmation()

    def _execute_confirmation(self):
        """Execute the confirmed action"""
        # Save settings if needed
        if hasattr(self, "skip_var"):
            self._save_settings()

        # Call confirm callback
        if self.on_confirm:
            self.on_confirm()

        # Close dialog
        self.destroy()

    def _on_cancel(self):
        """Handle cancel button click"""
        if self.on_cancel:
            self.on_cancel()
        self.destroy()

    def _on_window_close(self):
        """Handle window close event"""
        if self.on_cancel:
            self.on_cancel()
        self.destroy()

    @staticmethod
    def show_confirmation(
        parent, position: Dict, action_type: str = "close", close_details: Optional[Dict] = None
    ) -> bool:
        """
        Static method to show confirmation dialog

        Args:
            parent: Parent window
            position: Position data
            action_type: Type of action (close, partial_close, modify_tp_sl, etc.)
            close_details: Additional details about the close action

        Returns:
            bool: True if confirmed, False otherwise
        """
        confirmed = [False]

        def on_confirm():
            confirmed[0] = True

        dialog = CloseConfirmationDialog(
            parent, position, action_type, close_details, on_confirm=on_confirm, on_cancel=lambda: None
        )

        parent.wait_window(dialog)
        return confirmed[0]

