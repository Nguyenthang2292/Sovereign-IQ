from tkinter import messagebox
from typing import Callable, Dict, Optional

import ccxt
import customtkinter as ctk

from modules.auto_trade.gui.utils.colors import Colors
from modules.auto_trade.gui.utils.fonts import Fonts
from modules.auto_trade.gui.utils.retry_utils import retry_with_exponential_backoff
from modules.auto_trade.gui.utils.toast import show_toast


class PositionActions(ctk.CTkFrame):
    """
    Position Actions Panel
    Provides controls for closing positions, partial closing, and modifying TP/SL
    """

    def _format_pnl(self, pnl: float) -> str:
        """Format P&L with sign and color"""
        sign = "+" if pnl >= 0 else ""
        return f"{sign}${pnl:,.2f}"

    def __init__(self, parent, position: Dict, on_action_callback: Optional[Callable] = None):
        super().__init__(parent)

        self.position = position
        self.on_action_callback = on_action_callback

        # Title
        title = ctk.CTkLabel(self, text="⚡ Position Actions", font=Fonts.H1)
        title.pack(pady=(10, 15))

        # Create sections
        self._create_close_section()
        self._create_partial_close_section()
        self._create_modify_tp_sl_section()
        self._create_margin_section()  # New section

    def _create_margin_section(self):
        """Create margin management controls (for Isolated mode)"""
        # Only show if not cross margin (if we can detect it)
        # Assuming position has 'isolated' or similar flag
        # If unknown, show it anyway but it might fail on backend

        margin_frame = ctk.CTkFrame(self, fg_color=Colors.get_card_bg(), corner_radius=0)
        margin_frame.pack(fill="x", padx=15, pady=10)

        title = ctk.CTkLabel(margin_frame, text="Add Margin (Isolated)", font=Fonts.H2)
        title.pack(pady=(10, 5))

        # Input frame
        input_frame = ctk.CTkFrame(margin_frame, fg_color=Colors.TRANSPARENT)
        input_frame.pack(fill="x", padx=10, pady=(5, 10))

        label = ctk.CTkLabel(input_frame, text="Amount (USDT):", font=Fonts.BODY, text_color=Colors.TEXT_MUTED)
        label.pack(side="left")

        self.margin_entry = ctk.CTkEntry(input_frame, width=120, placeholder_text="Enter amount")
        self.margin_entry.pack(side="left", padx=(10, 0))

        # Add button
        self.add_margin_btn = ctk.CTkButton(
            margin_frame,
            text="➕ ADD MARGIN",
            font=Fonts.BUTTON_SM,
            height=35,
            fg_color=Colors.BTN_NEUTRAL,
            hover_color=Colors.BTN_NEUTRAL_HOVER,
            command=self._confirm_add_margin,
        )
        self.add_margin_btn.pack(fill="x", padx=10, pady=(0, 10))

    def _confirm_add_margin(self):
        """Confirm add margin action"""
        try:
            amount_str = self.margin_entry.get()
            if not amount_str:
                messagebox.showerror("Error", "Please enter an amount")
                return

            amount = float(amount_str)
            if amount <= 0:
                messagebox.showerror("Error", "Amount must be positive")
                return

            symbol = self.position.get("symbol", "N/A")

            msg = f"""
➕ Confirm Add Margin

Symbol: {symbol}
Amount: ${amount:,.2f} USDT

This will increase the margin for this isolated position.
(Only works in Isolated Mode)
            """

            confirm = messagebox.askyesno("Confirm Add Margin", msg.strip())
            if confirm:
                self._execute_add_margin(amount)
        except ValueError:
            messagebox.showerror("Error", "Invalid amount format")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to prepare add margin: {e}")

    def _execute_add_margin(self, amount: float):
        """Execute add margin with retry logic"""
        try:
            self.add_margin_btn.configure(state="disabled", text="Adding...")
            self.update()

            symbol = self.position.get("symbol", "")

            # Call with retry logic
            result = self._execute_with_retry(
                {
                    "action": "add_margin",
                    "symbol": symbol,
                    "amount": amount,
                }
            )

            if result and result.get("success"):
                show_toast(self, "Margin added successfully!", type="success")
                self.margin_entry.delete(0, "end")
            else:
                error_msg = result.get("error", "Unknown error") if result else "No response"
                # Check for common error (Cross Margin)
                if "cross" in str(error_msg).lower():
                    error_msg = "Cannot add margin in Cross Margin mode"

                show_toast(self, f"Failed to add margin: {error_msg}", type="error")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to execute add margin: {e}")
        finally:
            if self.winfo_exists():
                self.add_margin_btn.configure(state="normal", text="➕ Add Margin")

    def _execute_with_retry(self, action_data: Dict) -> Dict:
        """
        Execute action with retry logic for transient network errors

        Args:
            action_data: Dictionary containing action details

        Returns:
            Result dictionary with 'success' and optional 'error' keys
        """

        @retry_with_exponential_backoff(
            max_retries=3, base_delay=1.0, exceptions=(ccxt.NetworkError, ccxt.RequestTimeout, ConnectionError)
        )
        def execute_action():
            if self.on_action_callback:
                return self.on_action_callback(action_data)
            return {"success": False, "error": "No callback configured"}

        try:
            return execute_action()
        except (ccxt.NetworkError, ccxt.RequestTimeout, ConnectionError) as e:
            return {"success": False, "error": f"Network error after retries: {str(e)}"}
        except Exception as e:
            return {"success": False, "error": f"Unexpected error: {str(e)}"}

    def _create_close_section(self):
        """Create close position controls"""
        close_frame = ctk.CTkFrame(self, fg_color=Colors.get_card_bg(), corner_radius=0)
        close_frame.pack(fill="x", padx=15, pady=(0, 10))

        title = ctk.CTkLabel(close_frame, text="Close Position", font=Fonts.H2)
        title.pack(pady=(10, 5))

        # Close type selection
        type_frame = ctk.CTkFrame(close_frame, fg_color=Colors.TRANSPARENT)
        type_frame.pack(fill="x", padx=10, pady=(5, 10))

        self.close_type_var = ctk.StringVar(value="market")

        market_radio = ctk.CTkRadioButton(
            type_frame,
            text="Market (Immediate)",
            variable=self.close_type_var,
            value="market",
            command=self._on_close_type_change,
        )
        market_radio.pack(side="left", padx=(0, 20))

        limit_radio = ctk.CTkRadioButton(
            type_frame,
            text="Limit (At Price)",
            variable=self.close_type_var,
            value="limit",
            command=self._on_close_type_change,
        )
        limit_radio.pack(side="left")

        # Limit price input (hidden by default)
        self.limit_price_frame = ctk.CTkFrame(close_frame, fg_color=Colors.TRANSPARENT)
        self.limit_price_frame.pack(fill="x", padx=10, pady=(0, 10))
        self.limit_price_frame.pack_forget()

        limit_label = ctk.CTkLabel(self.limit_price_frame, text="Limit Price:", font=Fonts.BODY, text_color=Colors.TEXT_MUTED)
        limit_label.pack(side="left")

        self.limit_price_entry = ctk.CTkEntry(self.limit_price_frame, width=150, placeholder_text="Enter price")
        self.limit_price_entry.pack(side="left", padx=(10, 0))

        # Close button
        self.close_btn = ctk.CTkButton(
            close_frame,
            text="🔴 CLOSE POSITION",
            font=Fonts.BUTTON,
            height=35,
            fg_color=Colors.BTN_DANGER,
            hover_color=Colors.BTN_DANGER_HOVER,
            command=self._confirm_close_position,
        )
        self.close_btn.pack(fill="x", padx=10, pady=(0, 10))

    def _on_close_type_change(self):
        """Handle close type radio button change"""
        close_type = self.close_type_var.get()
        if close_type == "limit":
            self.limit_price_frame.pack(fill="x", padx=10, pady=(0, 10), after=self._find_type_frame())
        else:
            self.limit_price_frame.pack_forget()

    def _find_type_frame(self):
        """Helper to find the type frame"""
        for widget in self.winfo_children():
            if isinstance(widget, ctk.CTkFrame) and widget.winfo_children():
                for child in widget.winfo_children():
                    if isinstance(child, ctk.CTkFrame):
                        return child
        return None

    def _create_partial_close_section(self):
        """Create partial close controls"""
        partial_frame = ctk.CTkFrame(self, fg_color=Colors.get_card_bg(), corner_radius=0)
        partial_frame.pack(fill="x", padx=15, pady=10)

        title = ctk.CTkLabel(partial_frame, text="Partial Close", font=Fonts.H2)
        title.pack(pady=(10, 5))

        # Percentage buttons
        pct_frame = ctk.CTkFrame(partial_frame, fg_color=Colors.TRANSPARENT)
        pct_frame.pack(fill="x", padx=10, pady=(5, 10))

        self.partial_pct_var = ctk.StringVar(value="25")

        percentages = ["25%", "50%", "75%", "Max", "Custom"]
        for i, pct in enumerate(percentages):
            value = pct.replace("%", "").replace("Max", "100")
            btn = ctk.CTkButton(
                pct_frame,
                text=pct,
                width=60,
                height=30,
                command=lambda v=value: self._set_partial_pct(v),
            )
            btn.pack(side="left", padx=2)

        # Custom percentage input
        custom_frame = ctk.CTkFrame(partial_frame, fg_color=Colors.TRANSPARENT)
        custom_frame.pack(fill="x", padx=10, pady=(0, 5))

        custom_label = ctk.CTkLabel(custom_frame, text="Custom %:", font=Fonts.BODY, text_color=Colors.TEXT_MUTED)
        custom_label.pack(side="left")

        self.custom_pct_entry = ctk.CTkEntry(custom_frame, width=80, placeholder_text="50")
        self.custom_pct_entry.pack(side="left", padx=(10, 0))
        self.custom_pct_entry.bind("<KeyRelease>", self._on_custom_pct_change)

        # Partial close button
        self.partial_btn = ctk.CTkButton(
            partial_frame,
            text="⚡ PARTIAL CLOSE",
            font=Fonts.BUTTON_SM,
            height=35,
            fg_color=Colors.BTN_WARNING,
            hover_color=Colors.BTN_WARNING_HOVER,
            command=self._confirm_partial_close,
        )
        self.partial_btn.pack(fill="x", padx=10, pady=(0, 10))

    def _set_partial_pct(self, value: str):
        """Set partial close percentage"""
        self.partial_pct_var.set(value)
        if value != "Custom":
            self.custom_pct_entry.delete(0, "end")

    def _on_custom_pct_change(self, event):
        """Handle custom percentage input"""
        value = self.custom_pct_entry.get()
        if value:
            self.partial_pct_var.set(value)

    def _create_modify_tp_sl_section(self):
        """Create TP/SL modification controls"""
        modify_frame = ctk.CTkFrame(self, fg_color=Colors.get_card_bg(), corner_radius=0)
        modify_frame.pack(fill="x", padx=15, pady=10)

        title = ctk.CTkLabel(modify_frame, text="Modify TP/SL", font=Fonts.H2)
        title.pack(pady=(10, 5))

        # Current TP/SL display
        current_frame = ctk.CTkFrame(modify_frame, fg_color=Colors.TRANSPARENT)
        current_frame.pack(fill="x", padx=10, pady=(5, 10))

        current_tp = self.position.get("take_profit", 0)
        current_sl = self.position.get("stop_loss", 0)

        ct_label = ctk.CTkLabel(current_frame, text="Current:", font=Fonts.BODY, text_color=Colors.TEXT_MUTED)
        ct_label.grid(row=0, column=0, sticky="w", pady=2)

        ct_tp = ctk.CTkLabel(
            current_frame,
            text=f"TP: ${current_tp:,.2f}" if current_tp > 0 else "TP: Not set",
            font=Fonts.BODY,
            text_color=Colors.PROFIT,
        )
        ct_tp.grid(row=0, column=1, sticky="w", pady=2, padx=(10, 20))

        ct_sl = ctk.CTkLabel(
            current_frame,
            text=f"SL: ${current_sl:,.2f}" if current_sl > 0 else "SL: Not set",
            font=Fonts.BODY,
            text_color=Colors.LOSS,
        )
        ct_sl.grid(row=0, column=2, sticky="w", pady=2)

        # New TP/SL inputs
        input_frame = ctk.CTkFrame(modify_frame, fg_color=Colors.TRANSPARENT)
        input_frame.pack(fill="x", padx=10, pady=(5, 10))

        # TP input
        tp_label = ctk.CTkLabel(input_frame, text="New TP Price:", font=Fonts.BODY, text_color=Colors.TEXT_MUTED)
        tp_label.grid(row=0, column=0, sticky="w", pady=5)

        self.tp_entry = ctk.CTkEntry(input_frame, width=120, placeholder_text="Enter TP price")
        self.tp_entry.grid(row=0, column=1, sticky="w", pady=5, padx=(10, 20))

        # SL input
        sl_label = ctk.CTkLabel(input_frame, text="New SL Price:", font=Fonts.BODY, text_color=Colors.TEXT_MUTED)
        sl_label.grid(row=1, column=0, sticky="w", pady=5)

        self.sl_entry = ctk.CTkEntry(input_frame, width=120, placeholder_text="Enter SL price")
        self.sl_entry.grid(row=1, column=1, sticky="w", pady=5, padx=(10, 20))

        # Action buttons
        btn_frame = ctk.CTkFrame(modify_frame, fg_color=Colors.TRANSPARENT)
        btn_frame.pack(fill="x", padx=10, pady=(0, 10))

        # Modify button
        self.modify_btn = ctk.CTkButton(
            btn_frame,
            text="✏️ APPLY CHANGES",
            font=Fonts.BUTTON_SM,
            height=35,
            fg_color=Colors.BTN_PRIMARY,
            hover_color=Colors.BTN_PRIMARY_HOVER,
            command=self._confirm_modify_tp_sl,
        )
        self.modify_btn.pack(side="left", fill="x", expand=True, padx=(0, 5))

        # Breakeven button
        self.be_btn = ctk.CTkButton(
            btn_frame,
            text="🎯 BREAKEVEN",
            font=Fonts.BUTTON_SM,
            height=35,
            fg_color=Colors.BTN_SUCCESS,
            hover_color=Colors.BTN_SUCCESS_HOVER,
            command=self._confirm_breakeven,
        )
        self.be_btn.pack(side="left", fill="x", expand=True, padx=(5, 0))

        # Cancel orders button
        self.cancel_btn = ctk.CTkButton(
            modify_frame,
            text="🚫 CANCEL OPEN ORDERS",
            font=Fonts.BUTTON_SM,
            height=35,
            fg_color=Colors.TEXT_MUTED,
            hover_color=Colors.TEXT_MUTED_DARK,
            command=self._confirm_cancel_orders,
        )
        self.cancel_btn.pack(fill="x", padx=10, pady=(0, 10))

    def _confirm_close_position(self):
        """Confirm close position action"""
        try:
            close_type = self.close_type_var.get()

            # Get position data
            symbol = self.position.get("symbol", "N/A")
            side = self.position.get("side", "LONG")
            size = self.position.get("size", 0)
            unrealized_pnl = self.position.get("unrealized_pnl", 0)

            # Build confirmation message
            if close_type == "market":
                msg = f"""
⚠️ Confirm Close Position

Symbol: {symbol}
Side: {side}
Size: {size}
Type: Market (Immediate)

Unrealized P&L: {self._format_pnl(unrealized_pnl)}

This will close the position immediately at current market price.
                """
            else:  # limit
                limit_price = self.limit_price_entry.get()
                if not limit_price:
                    messagebox.showerror("Error", "Please enter a limit price")
                    return

                limit_price = float(limit_price)
                msg = f"""
⚠️ Confirm Close Position

Symbol: {symbol}
Side: {side}
Size: {size}
Type: Limit @ ${limit_price:,.2f}

Unrealized P&L: {self._format_pnl(unrealized_pnl)}

This will close the position when price reaches ${limit_price:,.2f}.
                """

            confirm = messagebox.askyesno("Confirm Close", msg.strip())
            if confirm:
                self._execute_close_position(close_type)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to prepare close confirmation: {e}")

    def _confirm_partial_close(self):
        """Confirm partial close action"""
        try:
            # Get percentage
            pct_str = self.partial_pct_var.get()
            if pct_str == "Custom":
                pct_str = self.custom_pct_entry.get()

            if not pct_str:
                messagebox.showerror("Error", "Please select or enter a percentage")
                return

            pct = float(pct_str)
            if pct <= 0 or pct > 100:
                messagebox.showerror("Error", "Percentage must be between 1 and 100")
                return

            # Get position data
            symbol = self.position.get("symbol", "N/A")
            side = self.position.get("side", "LONG")
            size = self.position.get("size", 0)
            close_size = size * (pct / 100)
            remaining_size = size - close_size
            unrealized_pnl = self.position.get("unrealized_pnl", 0)
            estimated_pnl = unrealized_pnl * (pct / 100)

            # Build confirmation message
            msg = f"""
⚠️ Confirm Partial Close

Symbol: {symbol}
Side: {side}

Current Size: {size}
Close: {close_size} ({pct}%)
Remaining: {remaining_size}

Estimated P&L on close: {self._format_pnl(estimated_pnl)}
            """

            confirm = messagebox.askyesno("Confirm Partial Close", msg.strip())
            if confirm:
                self._execute_partial_close(pct)
        except ValueError:
            messagebox.showerror("Error", "Invalid percentage value")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to prepare partial close: {e}")

    def _confirm_modify_tp_sl(self):
        """Confirm TP/SL modification"""
        try:
            # Get new values
            tp_str = self.tp_entry.get()
            sl_str = self.sl_entry.get()

            if not tp_str and not sl_str:
                messagebox.showerror("Error", "Please enter new TP or SL price")
                return

            new_tp = float(tp_str) if tp_str else self.position.get("take_profit", 0)
            new_sl = float(sl_str) if sl_str else self.position.get("stop_loss", 0)

            # Validate TP/SL
            if not self._validate_tp_sl(new_tp, new_sl):
                return

            # Get position data
            symbol = self.position.get("symbol", "N/A")
            side = self.position.get("side", "LONG")

            # Build confirmation message
            msg = f"""
✏️ Confirm TP/SL Modification

Symbol: {symbol}
Side: {side}

New TP: ${new_tp:,.2f} (Current: ${self.position.get("take_profit", 0):,.2f})
New SL: ${new_sl:,.2f} (Current: ${self.position.get("stop_loss", 0):,.2f})

This will modify the stop loss and take profit orders for this position.
            """

            confirm = messagebox.askyesno("Confirm Modification", msg.strip())
            if confirm:
                self._execute_modify_tp_sl(new_tp, new_sl)
        except ValueError:
            messagebox.showerror("Error", "Invalid price format")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to prepare TP/SL modification: {e}")

    def _confirm_breakeven(self):
        """Confirm breakeven action"""
        try:
            entry_price = self.position.get("entry_price", 0)
            if entry_price <= 0:
                messagebox.showerror("Error", "Entry price not available")
                return

            symbol = self.position.get("symbol", "N/A")

            msg = f"""
🎯 Confirm Breakeven

Symbol: {symbol}

This will move your Stop Loss to breakeven (Entry Price: ${entry_price:,.2f}).

This protects your profits while keeping the trade open.
            """

            confirm = messagebox.askyesno("Confirm Breakeven", msg.strip())
            if confirm:
                self._execute_breakeven()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to prepare breakeven: {e}")

    def _confirm_cancel_orders(self):
        """Confirm cancel open orders action"""
        try:
            symbol = self.position.get("symbol", "N/A")

            msg = f"""
🚫 Confirm Cancel Orders

Symbol: {symbol}

This will cancel all open orders (TP/SL) for this position.

The position itself will remain open.
            """

            confirm = messagebox.askyesno("Confirm Cancel", msg.strip())
            if confirm:
                self._execute_cancel_orders()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to prepare cancel orders: {e}")

    def _validate_tp_sl(self, tp: float, sl: float) -> bool:
        """Validate TP/SL prices"""
        side = self.position.get("side", "LONG")
        entry_price = self.position.get("entry_price", 0)
        current_price = self.position.get("current_price", entry_price)

        errors = []

        if tp > 0:
            if side == "LONG":
                if tp <= entry_price:
                    errors.append("Take Profit must be above entry price for LONG")
            else:  # SHORT
                if tp >= entry_price:
                    errors.append("Take Profit must be below entry price for SHORT")

        if sl > 0:
            if side == "LONG":
                if sl >= entry_price:
                    errors.append("Stop Loss must be below entry price for LONG")

                # Warn if SL too close to current price
                if side == "LONG" and sl >= current_price * 0.98:
                    errors.append("Warning: Stop Loss is very close to current price!")
                elif side == "SHORT" and sl <= current_price * 1.02:
                    errors.append("Warning: Stop Loss is very close to current price!")
            else:  # SHORT
                if sl <= entry_price:
                    errors.append("Stop Loss must be above entry price for SHORT")

                # Warn if SL too close to current price
                if side == "SHORT" and sl <= current_price * 1.02:
                    errors.append("Warning: Stop Loss is very close to current price!")

        if errors:
            messagebox.showerror("Validation Error", "\n".join(errors))
            return False

        return True

    def _execute_close_position(self, close_type: str):
        """Execute close position with retry logic"""
        try:
            self.close_btn.configure(state="disabled", text="Closing...")
            self.update()

            symbol = self.position.get("symbol", "")
            side = self.position.get("side", "LONG").lower()
            size = self.position.get("size", 0)

            limit_price = None
            if close_type == "limit":
                limit_price = float(self.limit_price_entry.get())

            # Call with retry logic
            result = self._execute_with_retry(
                {
                    "action": "close_position",
                    "symbol": symbol,
                    "side": side,
                    "size": size,
                    "type": close_type,
                    "limit_price": limit_price,
                }
            )

            if result and result.get("success"):
                show_toast(self, "Position closed successfully!", type="success")
            else:
                error_msg = result.get("error", "Unknown error") if result else "No response"
                show_toast(self, f"Failed to close position: {error_msg}", type="error")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to execute close: {e}")
        finally:
            if self.winfo_exists():
                self.close_btn.configure(state="normal", text="🔴 Close Position")

    def _execute_partial_close(self, percentage: float):
        """Execute partial close with retry logic"""
        try:
            self.partial_btn.configure(state="disabled", text="Executing...")
            self.update()

            symbol = self.position.get("symbol", "")
            side = self.position.get("side", "LONG").lower()
            size = self.position.get("size", 0)
            close_size = size * (percentage / 100)

            # Call with retry logic
            result = self._execute_with_retry(
                {
                    "action": "partial_close",
                    "symbol": symbol,
                    "side": side,
                    "size": close_size,
                    "percentage": percentage,
                }
            )

            if result and result.get("success"):
                show_toast(self, f"Partial close executed ({percentage}%)!", type="success")
            else:
                error_msg = result.get("error", "Unknown error") if result else "No response"
                show_toast(self, f"Failed to partial close: {error_msg}", type="error")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to execute partial close: {e}")
        finally:
            if self.winfo_exists():
                self.partial_btn.configure(state="normal", text="⚡ Partial Close")

    def _execute_modify_tp_sl(self, tp: float, sl: float):
        """Execute TP/SL modification with retry logic"""
        try:
            self.modify_btn.configure(state="disabled", text="Applying...")
            self.update()

            symbol = self.position.get("symbol", "")
            position_id = self.position.get("id", "")

            # Call with retry logic
            result = self._execute_with_retry(
                {
                    "action": "modify_tp_sl",
                    "symbol": symbol,
                    "position_id": position_id,
                    "take_profit": tp if tp > 0 else None,
                    "stop_loss": sl if sl > 0 else None,
                }
            )

            if result and result.get("success"):
                show_toast(self, "TP/SL modified successfully!", type="success")
            else:
                error_msg = result.get("error", "Unknown error") if result else "No response"
                show_toast(self, f"Failed to modify TP/SL: {error_msg}", type="error")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to execute TP/SL modification: {e}")
        finally:
            if self.winfo_exists():
                self.modify_btn.configure(state="normal", text="✏️ Apply Changes")

    def _execute_breakeven(self):
        """Execute breakeven action"""
        try:
            entry_price = self.position.get("entry_price", 0)
            self.position["stop_loss"] = entry_price

            # Call modify TP/SL with new SL at breakeven
            self._execute_modify_tp_sl(self.position.get("take_profit", 0), entry_price)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to execute breakeven: {e}")

    def _execute_cancel_orders(self):
        """Execute cancel orders action with retry logic"""
        try:
            self.cancel_btn.configure(state="disabled", text="Cancelling...")
            self.update()

            symbol = self.position.get("symbol", "")

            # Call with retry logic
            result = self._execute_with_retry({"action": "cancel_orders", "symbol": symbol})

            if result and result.get("success"):
                show_toast(self, "Open orders cancelled successfully!", type="success")
            else:
                error_msg = result.get("error", "Unknown error") if result else "No response"
                show_toast(self, f"Failed to cancel orders: {error_msg}", type="error")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to cancel orders: {e}")
        finally:
            if self.winfo_exists():
                self.cancel_btn.configure(state="normal", text="🚫 Cancel Open Orders")
