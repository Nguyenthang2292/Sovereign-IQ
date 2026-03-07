from typing import Any, Callable, Optional

import customtkinter as ctk

from modules.auto_trade.gui.utils.colors import Colors
from modules.auto_trade.gui.utils.fonts import Fonts
from modules.auto_trade.gui.utils.svg_icons import get_button_icon, get_icon
from modules.auto_trade.gui.utils.windows_utils import apply_dark_titlebar
from modules.common.ui.logging import log_error, log_success


class TradeFormFrame(ctk.CTkFrame):
    """
    Manual trading interface
    Allows users to place LONG/SHORT orders with TP/SL
    """

    def __init__(self, parent: Any, on_trade_callback: Optional[Callable[..., Any]] = None):
        super().__init__(parent)

        self.on_trade_callback = on_trade_callback

        # Title
        icon_crosshair = get_icon("crosshair", size=(20, 20))
        title = ctk.CTkLabel(self, text=" Manual Trade", image=icon_crosshair, compound="left", font=Fonts.H1)
        title.pack(pady=(10, 15))

        # Form fields
        self._create_form()

        # Risk calculator display
        self._create_risk_display()

        # Trade button
        self._create_trade_button()

    def _create_form(self):
        form_frame = ctk.CTkFrame(self, fg_color=Colors.TRANSPARENT)
        form_frame.pack(fill="both", expand=True, padx=15, pady=10)

        # Symbol selection
        symbol_label = ctk.CTkLabel(form_frame, text="Symbol:", font=Fonts.INPUT)
        symbol_label.grid(row=0, column=0, sticky="w", pady=5)

        # Dropdown with popular symbols
        self.symbol_var = ctk.StringVar(value="BTC/USDT")
        self.symbol_dropdown = ctk.CTkComboBox(
            form_frame,
            values=["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT", "ADA/USDT", "DOGE/USDT"],
            variable=self.symbol_var,
            command=self._on_symbol_change,
            width=200,
        )
        self.symbol_dropdown.grid(row=0, column=1, sticky="ew", pady=5, padx=(10, 0))

        # Current price display
        self.current_price_label = ctk.CTkLabel(form_frame, text="Price: $0.00", font=Fonts.BODY, text_color=Colors.TEXT_MUTED)
        self.current_price_label.grid(row=0, column=2, sticky="w", pady=5, padx=(10, 0))

        # Configure grid columns
        form_frame.grid_columnconfigure(1, weight=1)

        # Side selection (LONG/SHORT)
        side_label = ctk.CTkLabel(form_frame, text="Side:", font=Fonts.INPUT)
        side_label.grid(row=1, column=0, sticky="w", pady=5)

        side_frame = ctk.CTkFrame(form_frame, fg_color=Colors.TRANSPARENT)
        side_frame.grid(row=1, column=1, sticky="w", pady=5, padx=(10, 0))

        self.side_var = ctk.StringVar(value="LONG")

        long_radio = ctk.CTkRadioButton(
            side_frame,
            text="LONG",
            variable=self.side_var,
            value="LONG",
            text_color=Colors.PROFIT,
            command=self._calculate_risk,
        )
        long_radio.pack(side="left", padx=(0, 20))

        short_radio = ctk.CTkRadioButton(
            side_frame,
            text="SHORT",
            variable=self.side_var,
            value="SHORT",
            text_color=Colors.LOSS,
            command=self._calculate_risk,
        )
        short_radio.pack(side="left")

        # Amount (USDT)
        amount_label = ctk.CTkLabel(form_frame, text="Amount (USDT):", font=Fonts.INPUT)
        amount_label.grid(row=2, column=0, sticky="w", pady=5)

        self.amount_entry = ctk.CTkEntry(form_frame, placeholder_text="10.00", width=200)
        self.amount_entry.grid(row=2, column=1, sticky="ew", pady=5, padx=(10, 0))
        self.amount_entry.bind("<KeyRelease>", lambda e: self._calculate_risk())

        # Quick amount buttons
        quick_amounts_frame = ctk.CTkFrame(form_frame, fg_color=Colors.TRANSPARENT)
        quick_amounts_frame.grid(row=2, column=2, sticky="w", pady=5, padx=(10, 0))

        for amount in [5, 10, 20, 50]:
            btn = ctk.CTkButton(
                quick_amounts_frame,
                text=f"${amount}",
                width=50,
                height=25,
                command=lambda a=amount: self._set_amount(a),
            )
            btn.pack(side="left", padx=2)

        # Leverage
        leverage_label = ctk.CTkLabel(form_frame, text="Leverage:", font=Fonts.INPUT)
        leverage_label.grid(row=3, column=0, sticky="w", pady=5)

        self.leverage_var = ctk.StringVar(value="10x")
        self.leverage_dropdown = ctk.CTkComboBox(
            form_frame,
            values=["1x", "2x", "3x", "5x", "10x", "20x", "50x", "100x"],
            variable=self.leverage_var,
            command=lambda _: self._calculate_risk(),
            width=200,
        )
        self.leverage_dropdown.grid(row=3, column=1, sticky="ew", pady=5, padx=(10, 0))

        # Warning for high leverage
        self.leverage_warning = ctk.CTkLabel(
            form_frame, text="⚠️ High leverage = High risk", font=Fonts.SMALL, text_color=Colors.BTN_WARNING
        )
        self.leverage_warning.grid(row=3, column=2, sticky="w", pady=5, padx=(10, 0))
        self.leverage_warning.grid_remove()  # Hide initially

        # Stop Loss
        sl_label = ctk.CTkLabel(form_frame, text="Stop Loss (%):", font=Fonts.INPUT)
        sl_label.grid(row=4, column=0, sticky="w", pady=5)

        self.sl_entry = ctk.CTkEntry(form_frame, placeholder_text="2.5", width=200)
        self.sl_entry.grid(row=4, column=1, sticky="ew", pady=5, padx=(10, 0))
        self.sl_entry.insert(0, "2.5")  # Default 2.5%
        self.sl_entry.bind("<KeyRelease>", lambda e: self._calculate_risk())

        # SL price display
        self.sl_price_label = ctk.CTkLabel(form_frame, text="Price: $0.00", font=Fonts.SMALL, text_color=Colors.LOSS)
        self.sl_price_label.grid(row=4, column=2, sticky="w", pady=5, padx=(10, 0))

        # Take Profit
        tp_label = ctk.CTkLabel(form_frame, text="Take Profit (%):", font=Fonts.INPUT)
        tp_label.grid(row=5, column=0, sticky="w", pady=5)

        self.tp_entry = ctk.CTkEntry(form_frame, placeholder_text="5.0", width=200)
        self.tp_entry.grid(row=5, column=1, sticky="ew", pady=5, padx=(10, 0))
        self.tp_entry.insert(0, "5.0")  # Default 5%
        self.tp_entry.bind("<KeyRelease>", lambda e: self._calculate_risk())

        # TP price display
        self.tp_price_label = ctk.CTkLabel(form_frame, text="Price: $0.00", font=Fonts.SMALL, text_color=Colors.PROFIT)
        self.tp_price_label.grid(row=5, column=2, sticky="w", pady=5, padx=(10, 0))

    def _on_symbol_change(self, choice: str):
        """Update current price when symbol changes"""
        try:
            from modules.auto_trade.gui.services.data_service import DataService

            service = DataService()
            price = service.get_current_price(choice)
            self.current_price_label.configure(text=f"Price: ${price:,.2f}")

            # Recalculate risk if form is filled
            self._calculate_risk()
        except Exception as e:
            log_error("Error fetching price: %s", e)
            self.current_price_label.configure(text="Price: N/A")

    def _calculate_risk(self):
        """Calculate and display risk metrics"""
        try:
            from modules.auto_trade.gui.services.data_service import DataService
            from modules.auto_trade.gui.utils.risk_calculator import RiskCalculator

            # Get form values
            symbol = self.symbol_var.get()
            side = self.side_var.get()
            amount_str = self.amount_entry.get()
            leverage_str = self.leverage_var.get().replace("x", "")
            tp_str = self.tp_entry.get()
            sl_str = self.sl_entry.get()

            # Validate inputs
            if not all([amount_str, leverage_str, tp_str, sl_str]):
                return

            amount = float(amount_str)
            leverage = int(leverage_str)
            tp_percent = float(tp_str)
            sl_percent = float(sl_str)

            # Get current price
            service = DataService()
            current_price = service.get_current_price(symbol)

            # Calculate risk
            risk = RiskCalculator.calculate(
                symbol=symbol,
                side=side,
                amount_usdt=amount,
                leverage=leverage,
                current_price=current_price,
                tp_percent=tp_percent,
                sl_percent=sl_percent,
            )

            if not risk:
                return

            # Update UI
            self._update_risk_display(risk, symbol)

            # Show warning if leverage > 10x
            if leverage > 10:
                self.leverage_warning.grid()
            else:
                self.leverage_warning.grid_remove()

        except Exception as e:
            log_error("Error in risk calculation: %s", e)

    def _update_risk_display(self, risk, symbol: str):
        """Update risk labels with calculated values"""
        # Contract size
        base_asset = symbol.split("/")[0]
        self.risk_labels["contract_size"].configure(text=f"{risk['contract_size']:.6f} {base_asset}")

        # Margin required
        self.risk_labels["margin_required"].configure(text=f"${risk['margin_required']:.2f}")

        # Max profit (green)
        self.risk_labels["max_profit"].configure(text=f"+${risk['max_profit']:.2f}", text_color=Colors.PROFIT)

        # Max loss (red)
        self.risk_labels["max_loss"].configure(text=f"-${risk['max_loss']:.2f}", text_color=Colors.LOSS)

        # Risk/Reward ratio
        rr = risk["risk_reward_ratio"]
        color = Colors.PROFIT if rr >= 2.0 else Colors.BTN_WARNING if rr >= 1.5 else Colors.LOSS
        self.risk_labels["risk_reward"].configure(text=f"{rr:.2f}:1", text_color=color)

        # Liquidation price
        self.risk_labels["liquidation"].configure(text=f"${risk['liquidation_price']:,.2f}", text_color=Colors.BTN_WARNING)

        # Update TP/SL price labels
        self.sl_price_label.configure(text=f"Price: ${risk['sl_price']:,.2f}")
        self.tp_price_label.configure(text=f"Price: ${risk['tp_price']:,.2f}")

    def _set_amount(self, amount: float):
        """Set amount from quick button"""
        self.amount_entry.delete(0, "end")
        self.amount_entry.insert(0, str(amount))
        self._calculate_risk()

    def _create_risk_display(self):
        """Display calculated risk metrics"""
        risk_frame = ctk.CTkFrame(self, fg_color=Colors.get_card_bg(), corner_radius=0)
        risk_frame.pack(fill="x", padx=15, pady=10)

        # Title
        icon_calc = get_icon("calculator", size=(18, 18))
        risk_title = ctk.CTkLabel(risk_frame, text=" Calculated Risk", image=icon_calc, compound="left", font=Fonts.H2)
        risk_title.pack(pady=(10, 5))

        # Grid for metrics
        metrics_frame = ctk.CTkFrame(risk_frame, fg_color=Colors.TRANSPARENT)
        metrics_frame.pack(fill="x", padx=10, pady=(5, 10))

        # Metric rows
        self.risk_labels = {}
        metrics = [
            ("contract_size", "Contract Size:", "0.000 BTC"),
            ("margin_required", "Margin Required:", "$0.00"),
            ("max_profit", "Max Profit:", "$0.00"),
            ("max_loss", "Max Loss:", "$0.00"),
            ("risk_reward", "Risk/Reward:", "0:0"),
            ("liquidation", "Liquidation Price:", "$0.00"),
        ]

        for i, (key, label_text, default_value) in enumerate(metrics):
            # Label
            label = ctk.CTkLabel(metrics_frame, text=label_text, font=Fonts.BODY, text_color=Colors.TEXT_MUTED)
            label.grid(row=i, column=0, sticky="w", pady=2)

            # Value
            value_label = ctk.CTkLabel(metrics_frame, text=default_value, font=Fonts.H3)
            value_label.grid(row=i, column=1, sticky="e", pady=2)

            self.risk_labels[key] = value_label

        metrics_frame.grid_columnconfigure(1, weight=1)

    def _create_trade_button(self):
        """Create the main trade execution button"""
        icon_rocket = get_button_icon("rocket", size=(20, 20), variant="danger")
        self.trade_button = ctk.CTkButton(
            self,
            text=" PLACE ORDER",
            image=icon_rocket,
            compound="left",
            font=Fonts.BUTTON,
            height=40,
            fg_color=Colors.BTN_DANGER,
            hover_color=Colors.BTN_DANGER_HOVER,
            command=self._confirm_trade,
        )
        self.trade_button.pack(fill="x", padx=15, pady=(5, 15))

    def _confirm_trade(self):
        """Show confirmation dialog before executing trade"""
        try:
            # Validate form
            if not self._validate_form():
                return

            # Get trade details
            symbol = self.symbol_var.get()
            side = self.side_var.get()
            amount = float(self.amount_entry.get())
            leverage = int(self.leverage_var.get().replace("x", ""))

            # Create confirmation dialog
            dialog = ctk.CTkToplevel(self)
            dialog.title("Confirm Trade")

            apply_dark_titlebar(dialog)

            dialog.geometry("400x300")
            dialog.transient(self.winfo_toplevel())
            dialog.grab_set()

            # Center dialog
            dialog.update_idletasks()
            x = (dialog.winfo_screenwidth() // 2) - (400 // 2)
            y = (dialog.winfo_screenheight() // 2) - (300 // 2)
            dialog.geometry(f"400x300+{x}+{y}")

            # Confirmation message
            msg_frame = ctk.CTkFrame(dialog)
            msg_frame.pack(fill="both", expand=True, padx=20, pady=20)

            title = ctk.CTkLabel(msg_frame, text="⚠️ Confirm Trade", font=Fonts.H1)
            title.pack(pady=(10, 15))

            details = f"""
Symbol: {symbol}
Side: {side}
Amount: ${amount:.2f} USDT
Leverage: {leverage}x

TP: {self.tp_entry.get()}%
SL: {self.sl_entry.get()}%

Max Profit: +${self.risk_labels["max_profit"].cget("text")}
Max Loss: -{self.risk_labels["max_loss"].cget("text")}
            """

            details_label = ctk.CTkLabel(msg_frame, text=details.strip(), font=Fonts.INPUT, justify="left")
            details_label.pack(pady=10)

            # Buttons
            btn_frame = ctk.CTkFrame(msg_frame, fg_color=Colors.TRANSPARENT)
            btn_frame.pack(side="bottom", pady=10)

            confirm_btn = ctk.CTkButton(
                btn_frame,
                text="✅ EXECUTE TRADE",
                font=Fonts.BUTTON,
                fg_color=Colors.BTN_SUCCESS,
                hover_color=Colors.BTN_SUCCESS_HOVER,
                command=lambda: self._execute_trade(dialog),
            )
            confirm_btn.pack(side="left", padx=5)

            cancel_btn = ctk.CTkButton(
                btn_frame,
                text="❌ CANCEL",
                font=Fonts.BUTTON_SM,
                fg_color=Colors.TEXT_MUTED,
                hover_color=Colors.TEXT_MUTED_DARK,
                command=dialog.destroy,
            )
            cancel_btn.pack(side="left", padx=5)

        except Exception as e:
            self._show_error(f"Error: {e}")

    def _validate_form(self) -> bool:
        """
        Validate all form inputs before trade
        Returns True if valid, False otherwise
        """
        errors = []

        # Amount
        try:
            amount = float(self.amount_entry.get())
            if amount <= 0:
                errors.append("Amount must be greater than 0")
            if amount > 1000:  # Max limit
                errors.append("Amount exceeds maximum limit ($1000)")
        except ValueError:
            errors.append("Invalid amount format")

        # Leverage
        try:
            leverage = int(self.leverage_var.get().replace("x", ""))
            if leverage < 1 or leverage > 100:
                errors.append("Leverage must be between 1x and 100x")
        except ValueError:
            errors.append("Invalid leverage format")

        # TP/SL percentages
        try:
            tp = float(self.tp_entry.get())
            sl = float(self.sl_entry.get())

            if tp <= 0 or sl <= 0:
                errors.append("TP/SL must be greater than 0")
            if tp > 100 or sl > 100:
                errors.append("TP/SL cannot exceed 100%")
            if tp < sl * 1.5:
                errors.append("TP should be at least 1.5x SL for good R:R")
        except ValueError:
            errors.append("Invalid TP/SL format")

        # Show errors if any
        if errors:
            error_msg = "\n".join(errors)
            self._show_error(error_msg)
            return False

        return True

    def _execute_trade(self, dialog):
        """Execute the trade via OrderExecutor"""
        try:
            from modules.auto_trade.execution.order_executor import OrderExecutor
            from modules.auto_trade.gui.services.data_service import DataService

            # Close confirmation dialog
            dialog.destroy()

            # Disable trade button
            self.trade_button.configure(state="disabled", text="⏳ Executing...")

            # Get trade parameters
            symbol = self.symbol_var.get()
            side = self.side_var.get()
            amount = float(self.amount_entry.get())
            leverage = int(self.leverage_var.get().replace("x", ""))
            tp_percent = float(self.tp_entry.get())
            sl_percent = float(self.sl_entry.get())

            # Get current price
            service = DataService()
            current_price = service.get_current_price(symbol)

            # Calculate TP/SL prices
            # tp_percent / sl_percent are ROI% on capital → convert to price-move%
            tp_price_pct = tp_percent / max(leverage, 1)
            sl_price_pct = sl_percent / max(leverage, 1)
            if side == "LONG":
                tp_price = current_price * (1 + tp_price_pct / 100)
                sl_price = current_price * (1 - sl_price_pct / 100)
            else:
                tp_price = current_price * (1 - tp_price_pct / 100)
                sl_price = current_price * (1 + sl_price_pct / 100)

            # Execute order
            executor = OrderExecutor()
            order_result = executor.place_order(
                symbol=symbol,
                side=side.lower(),
                amount=amount,
                leverage=leverage,
                take_profit=tp_price,
                stop_loss=sl_price,
            )

            if order_result and order_result.get("success"):
                self._show_success("Trade executed successfully!")

                # Call callback if provided
                if self.on_trade_callback:
                    self.on_trade_callback()

                # Reset form
                self._reset_form()
            else:
                error_msg = order_result.get("error", "Unknown error")
                self._show_error(f"Trade failed: {error_msg}")

        except Exception as e:
            self._show_error(f"Execution error: {e}")

        finally:
            # Re-enable trade button
            self.trade_button.configure(state="normal", text=" Place Order")

    def _show_success(self, message: str):
        """Show success notification"""
        # Could use tkinter.messagebox or custom dialog
        try:
            from tkinter import messagebox

            messagebox.showinfo("Trade Success", message)
        except Exception:
            log_success("SUCCESS: %s", message)

    def _show_error(self, message: str):
        """Show error notification"""
        try:
            from tkinter import messagebox

            messagebox.showerror("Trade Error", message)
        except Exception:
            log_error("ERROR: %s", message)

    def _reset_form(self):
        """Reset form to default values"""
        self.symbol_var.set("BTC/USDT")
        self.side_var.set("LONG")
        self.amount_entry.delete(0, "end")
        self.leverage_var.set("10x")
        self.tp_entry.delete(0, "end")
        self.tp_entry.insert(0, "5.0")
        self.sl_entry.delete(0, "end")
        self.sl_entry.insert(0, "2.5")

        # Recalculate risk
        self._calculate_risk()
