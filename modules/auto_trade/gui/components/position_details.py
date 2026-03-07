from typing import Callable, Dict, Optional

import customtkinter as ctk

from modules.auto_trade.gui.components.position_actions import PositionActions
from modules.auto_trade.gui.utils.colors import Colors
from modules.auto_trade.gui.utils.formatters import format_asset_price, format_price
from modules.auto_trade.gui.utils.windows_utils import apply_dark_titlebar
from modules.common.ui.logging import log_error


class PositionDetails(ctk.CTkToplevel):
    """
    Position Details Modal
    Displays comprehensive position information including P&L, TP/SL, liquidation,
    and provides position management actions.
    """

    def __init__(
        self,
        parent,
        position: Dict,
        on_close_callback: Optional[Callable] = None,
        on_action_callback: Optional[Callable] = None,
    ):
        super().__init__(parent)

        apply_dark_titlebar(self)

        self.position = position
        self.on_close_callback = on_close_callback
        self.on_action_callback = on_action_callback

        # Window configuration
        self.title(f"Position Details - {position.get('symbol', 'N/A')}")
        self.geometry("700x600")
        self.transient(parent)
        self.grab_set()

        # Center dialog
        self.update_idletasks()
        x = (self.winfo_screenwidth() // 2) - (700 // 2)
        y = (self.winfo_screenheight() // 2) - (600 // 2)
        self.geometry(f"700x600+{x}+{y}")

        # Setup UI
        self._create_header()
        self._create_main_content()
        self._create_pnl_section()
        self._create_actions()

    def _create_header(self):
        """Create position header with symbol and side"""
        header_frame = ctk.CTkFrame(self, fg_color=Colors.get_card_bg(), corner_radius=10)
        header_frame.pack(fill="x", padx=15, pady=(15, 10))

        # Symbol and side
        info_frame = ctk.CTkFrame(header_frame, fg_color="transparent")
        info_frame.pack(fill="x", padx=15, pady=10)

        symbol_label = ctk.CTkLabel(info_frame, text=self.position.get("symbol", "N/A"), font=("Arial", 20, "bold"))
        symbol_label.pack(side="left")

        side = self.position.get("side", "LONG")
        side_color = "#00ff88" if side == "LONG" else "#ff4444"
        side_label = ctk.CTkLabel(
            info_frame,
            text=side,
            font=("Arial", 16, "bold"),
            text_color=side_color,
            fg_color="#1a1a1a",
            corner_radius=5,
            padx=10,
            pady=5,
        )
        side_label.pack(side="right")

    def _create_main_content(self):
        """Create main content area with position details"""
        content_frame = ctk.CTkFrame(self, fg_color="transparent")
        content_frame.pack(fill="both", expand=True, padx=15, pady=10)

        # Position metrics grid
        self._create_metrics_grid(content_frame)

        # TP/SL visualization
        self._create_tp_sl_visualization(content_frame)

        # Liquidation warning
        self._create_liquidation_warning(content_frame)

    def _create_metrics_grid(self, parent):
        """Create grid of position metrics"""
        metrics_frame = ctk.CTkFrame(parent, fg_color=Colors.get_card_bg(), corner_radius=10)
        metrics_frame.pack(fill="x", pady=(0, 10))

        title = ctk.CTkLabel(metrics_frame, text="📊 Position Metrics", font=("Arial", 14, "bold"))
        title.pack(pady=(10, 5))

        # Metrics grid
        grid_frame = ctk.CTkFrame(metrics_frame, fg_color="transparent")
        grid_frame.pack(fill="x", padx=10, pady=(5, 10))

        self.metric_labels = {}

        # Define metrics
        metrics = [
            ("entry_price", "Entry Price:", "0.00"),
            ("mark_price", "Mark Price:", "0.00"),
            ("size", "Position Size:", "0.00"),
            ("leverage", "Leverage:", "0x"),
            ("margin", "Margin Used:", "$0.00"),
            ("liquidation_price", "Liquidation Price:", "0.00"),
        ]

        for i, (key, label, default) in enumerate(metrics):
            row = i // 2
            col = (i % 2) * 2

            label_widget = ctk.CTkLabel(grid_frame, text=label, font=("Arial", 11), text_color="gray")
            label_widget.grid(row=row, column=col, sticky="w", pady=5, padx=(0, 10))

            value_widget = ctk.CTkLabel(grid_frame, text=default, font=("Arial", 11, "bold"))
            value_widget.grid(row=row, column=col + 1, sticky="e", pady=5)

            self.metric_labels[key] = value_widget

        # Configure grid weights
        grid_frame.grid_columnconfigure(1, weight=1)
        grid_frame.grid_columnconfigure(3, weight=1)

        # Update with position data
        self._update_metrics()

    def _update_metrics(self):
        """Update metrics with position data"""
        try:
            # Entry price
            entry_price = self.position.get("entry_price", 0)
            self.metric_labels["entry_price"].configure(text=format_asset_price(float(entry_price)))

            # Mark price (if available, otherwise use current price)
            mark_price = self.position.get("mark_price") or self.position.get("current_price", entry_price)
            mp_color = "#00ff88" if mark_price > entry_price else "#ff4444"
            if self.position.get("side") == "SHORT":
                mp_color = "#00ff88" if mark_price < entry_price else "#ff4444"

            self.metric_labels["mark_price"].configure(
                text=format_asset_price(float(mark_price)),
                text_color=mp_color,
            )

            # Position size (in quote currency, e.g. USDT)
            size = float(self.position.get("size", 0) or 0)
            symbol = self.position.get("symbol", "")
            quote_asset = symbol.split("/")[1] if "/" in symbol else "USDT"
            self.metric_labels["size"].configure(text=f"{size:.4f} {quote_asset}")

            # Leverage
            leverage = self.position.get("leverage", 1)
            self.metric_labels["leverage"].configure(text=f"{leverage}x")

            # Margin
            margin = self.position.get("margin_used", 0)
            self.metric_labels["margin"].configure(text=format_price(float(margin)))

            # Liquidation price
            liq_price = self.position.get("liquidation_price")
            if liq_price is not None and float(liq_price) > 0:
                self.metric_labels["liquidation_price"].configure(
                    text=format_asset_price(float(liq_price)),
                    text_color="#ff4444",
                )
            else:
                self.metric_labels["liquidation_price"].configure(text="N/A")
        except Exception as e:
            log_error("Error updating metrics: %s", e)

    def _create_tp_sl_visualization(self, parent):
        """Create visual representation of TP/SL relative to entry"""
        viz_frame = ctk.CTkFrame(parent, fg_color=Colors.get_card_bg(), corner_radius=10)
        viz_frame.pack(fill="x", pady=(0, 10))

        title = ctk.CTkLabel(viz_frame, text="🎯 TP/SL Visualization", font=("Arial", 14, "bold"))
        title.pack(pady=(10, 5))

        # Visualization canvas
        canvas_frame = ctk.CTkFrame(viz_frame, fg_color="transparent")
        canvas_frame.pack(fill="x", padx=10, pady=(5, 10))

        # Get prices
        entry_price = self.position.get("entry_price", 0)
        tp_price = self.position.get("take_profit", 0)
        sl_price = self.position.get("stop_loss", 0)
        current_price = self.position.get("current_price", entry_price)

        # Create visual representation
        self._create_price_visual(canvas_frame, entry_price, tp_price, sl_price, current_price)

    def _create_price_visual(self, parent, entry_price: float, tp_price: float, sl_price: float, current_price: float):
        """Create visual bar showing price levels"""
        viz_container = ctk.CTkFrame(parent, fg_color="transparent")
        viz_container.pack(fill="x")

        # Calculate relative positions
        prices = [entry_price, tp_price, sl_price, current_price]
        min_price = min([p for p in prices if p > 0])
        max_price = max([p for p in prices if p > 0])
        price_range = max_price - min_price if max_price != min_price else 1

        # Define price levels to display
        levels = []
        if tp_price > 0:
            levels.append(("TP", tp_price, "#00ff88"))
        if entry_price > 0:
            levels.append(("Entry", entry_price, "#ffffff"))
        if sl_price > 0:
            levels.append(("SL", sl_price, "#ff4444"))
        if current_price > 0:
            levels.append(("Current", current_price, "#ffaa00"))

        # Sort by price
        levels.sort(key=lambda x: x[1])

        # Display levels
        for label, price, color in levels:
            level_frame = ctk.CTkFrame(viz_container, fg_color="transparent")
            level_frame.pack(fill="x", pady=2)

            label_widget = ctk.CTkLabel(
                level_frame, text=label, font=("Arial", 10), text_color="gray", width=50, anchor="w"
            )
            label_widget.pack(side="left")

            # Price label
            price_widget = ctk.CTkLabel(
                level_frame, text=f"${price:,.2f}", font=("Arial", 11, "bold"), text_color=color
            )
            price_widget.pack(side="right")

            # Visual bar
            bar_frame = ctk.CTkFrame(viz_container, fg_color="#1a1a1a", height=20)
            bar_frame.pack(fill="x", pady=(0, 5))

            if price_range > 0:
                position = ((price - min_price) / price_range) * 100
                position = max(5, min(95, position))  # Clamp between 5% and 95%
            else:
                position = 50

            marker = ctk.CTkLabel(bar_frame, text="◆", font=("Arial", 14), text_color=color, width=20)
            marker.place(x=position, rely=0.5, anchor="center")

    def _create_liquidation_warning(self, parent):
        """Create liquidation distance warning"""
        warning_frame = ctk.CTkFrame(parent, fg_color="#3d2a1a", corner_radius=10)
        warning_frame.pack(fill="x", pady=(0, 10))

        warning_title = ctk.CTkLabel(
            warning_frame, text="⚠️ Liquidation Risk", font=("Arial", 12, "bold"), text_color="#ffaa00"
        )
        warning_title.pack(pady=(10, 5))

        # Calculate distance to liquidation
        self._calculate_liquidation_distance(warning_frame)

    def _calculate_liquidation_distance(self, parent):
        """Calculate and display distance to liquidation"""
        try:
            current_price = self.position.get("current_price") or self.position.get("mark_price", 0)
            liq_price = self.position.get("liquidation_price", 0)
            side = self.position.get("side", "LONG")

            if liq_price > 0 and current_price > 0:
                if side == "LONG":
                    distance_pct = ((current_price - liq_price) / current_price) * 100
                else:  # SHORT
                    distance_pct = ((liq_price - current_price) / current_price) * 100

                # Color coding based on risk level
                if distance_pct < 2:
                    risk_color = "#ff0000"
                    risk_level = "CRITICAL"
                elif distance_pct < 5:
                    risk_color = "#ff4444"
                    risk_level = "HIGH"
                elif distance_pct < 10:
                    risk_color = "#ffaa00"
                    risk_level = "MEDIUM"
                else:
                    risk_color = "#00ff88"
                    risk_level = "LOW"

                distance_label = ctk.CTkLabel(
                    parent,
                    text=f"Distance to Liquidation: {distance_pct:.2f}% | Risk Level: {risk_level}",
                    font=("Arial", 11, "bold"),
                    text_color=risk_color,
                )
                distance_label.pack(pady=(0, 5))

                # Tooltip/Explanation
                tooltip_text = "Distance = (Current - Liq) / Current"
                tooltip = ctk.CTkLabel(parent, text=tooltip_text, font=("Arial", 10), text_color="gray")
                tooltip.pack(pady=(0, 10))
            else:
                distance_label = ctk.CTkLabel(
                    parent, text="Liquidation data not available", font=("Arial", 11), text_color="gray"
                )
                distance_label.pack(pady=(0, 10))
        except Exception as e:
            log_error("Error calculating liquidation distance: %s", e)

    def _create_pnl_section(self):
        """Create P&L display section"""
        pnl_frame = ctk.CTkFrame(self, fg_color=Colors.get_card_bg(), corner_radius=10)
        pnl_frame.pack(fill="x", padx=15, pady=10)

        title = ctk.CTkLabel(pnl_frame, text="💰 Profit & Loss", font=("Arial", 14, "bold"))
        title.pack(pady=(10, 5))

        # P&L grid
        grid_frame = ctk.CTkFrame(pnl_frame, fg_color="transparent")
        grid_frame.pack(fill="x", padx=10, pady=(5, 10))

        # Unrealized P&L
        unrealized_pnl = self.position.get("unrealized_pnl", 0)
        pnl_color = "#00ff88" if unrealized_pnl >= 0 else "#ff4444"
        pnl_sign = "+" if unrealized_pnl >= 0 else ""

        u_pnl_label = ctk.CTkLabel(grid_frame, text="Unrealized P&L:", font=("Arial", 11), text_color="gray")
        u_pnl_label.grid(row=0, column=0, sticky="w", pady=5)

        u_pnl_value = ctk.CTkLabel(
            grid_frame, text=f"{pnl_sign}${unrealized_pnl:,.2f}", font=("Arial", 14, "bold"), text_color=pnl_color
        )
        u_pnl_value.grid(row=0, column=1, sticky="e", pady=5)

        # ROI percentage
        if unrealized_pnl != 0:
            margin = self.position.get("margin_used", 1)
            roi_pct = (unrealized_pnl / margin) * 100
            roi_color = "#00ff88" if roi_pct >= 0 else "#ff4444"
            roi_sign = "+" if roi_pct >= 0 else ""

            roi_label = ctk.CTkLabel(grid_frame, text="ROI:", font=("Arial", 11), text_color="gray")
            roi_label.grid(row=1, column=0, sticky="w", pady=5)

            roi_value = ctk.CTkLabel(
                grid_frame, text=f"{roi_sign}{roi_pct:.2f}%", font=("Arial", 12, "bold"), text_color=roi_color
            )
            roi_value.grid(row=1, column=1, sticky="e", pady=5)

        grid_frame.grid_columnconfigure(1, weight=1)

    def _create_actions(self):
        """Create action buttons section"""
        actions_frame = ctk.CTkFrame(self, fg_color="transparent")
        actions_frame.pack(fill="x", padx=15, pady=(10, 15))

        # Position Actions Panel
        position_actions = PositionActions(actions_frame, self.position, on_action_callback=self.on_action_callback)
        position_actions.pack(fill="x", pady=(0, 10))

        # Close button
        close_btn = ctk.CTkButton(
            actions_frame,
            text="❌ Close Details",
            font=("Arial", 12),
            fg_color="gray",
            hover_color="darkgray",
            command=self._close_window,
            height=35,
        )
        close_btn.pack(fill="x", padx=15, pady=5)

    def _close_window(self):
        """Close the details window"""
        if self.on_close_callback:
            self.on_close_callback()
        self.destroy()

    def update_position(self, position: Dict):
        """Update position data and refresh UI"""
        self.position = position
        self._update_metrics()

    def on_closing(self):
        """Handle window close event"""
        self._close_window()
