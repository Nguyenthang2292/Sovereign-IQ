# 📋 Phase 2: Trade Execution (Manual & Auto) - Detailed Tasks

## 🎯 Mục Tiêu Phase 2
Thêm chức năng trade execution vào GUI: manual trading form, auto-trade toggle, risk calculations, và order execution integration.

## 📌 Prerequisites
- ✅ Phase 1 đã hoàn thành (GUI Dashboard hiển thị data)
- ✅ ExchangeManager đang hoạt động
- ✅ OrderExecutor module đã có
- ✅ DatabaseManager đang hoạt động

---

## 💱 I. MANUAL TRADE FORM COMPONENT

### 1.1 Create Trade Form Frame
- [x] **Task 1.1.1:** Tạo `gui/components/trade_form.py`
  ```python
  import customtkinter as ctk
  from typing import Dict, Optional, Callable
  
  class TradeFormFrame(ctk.CTkFrame):
      """
      Manual trading interface
      Allows users to place LONG/SHORT orders with TP/SL
      """
      def __init__(self, parent, on_trade_callback: Callable = None):
          super().__init__(parent)
          
          self.on_trade_callback = on_trade_callback
          
          # Title
          title = ctk.CTkLabel(
              self, 
              text="💱 Manual Trade", 
              font=("Arial", 16, "bold")
          )
          title.pack(pady=(10, 15))
          
          # Form fields
          self._create_form()
          
          # Risk calculator display
          self._create_risk_display()
          
          # Trade button
          self._create_trade_button()
      
      def _create_form(self):
          pass
      
      def _create_risk_display(self):
          pass
      
      def _create_trade_button(self):
          pass
  ```

### 1.2 Symbol Selection
- [x] **Task 1.2.1:** Symbol dropdown
- [ ] **Task 1.2.2:** Fetch current price on symbol change
  ```python
  def _create_form(self):
      form_frame = ctk.CTkFrame(self, fg_color="transparent")
      form_frame.pack(fill="both", expand=True, padx=15, pady=10)
      
      # Symbol selection
      symbol_label = ctk.CTkLabel(
          form_frame, 
          text="Symbol:",
          font=("Arial", 12)
      )
      symbol_label.grid(row=0, column=0, sticky="w", pady=5)
      
      # Dropdown with popular symbols
      self.symbol_var = ctk.StringVar(value="BTC/USDT")
      self.symbol_dropdown = ctk.CTkComboBox(
          form_frame,
          values=["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", 
                  "XRP/USDT", "ADA/USDT", "DOGE/USDT"],
          variable=self.symbol_var,
          command=self._on_symbol_change,
          width=200
      )
      self.symbol_dropdown.grid(row=0, column=1, sticky="ew", pady=5, padx=(10, 0))
      
      # Current price display
      self.current_price_label = ctk.CTkLabel(
          form_frame,
          text="Price: $0.00",
          font=("Arial", 11),
          text_color="gray"
      )
      self.current_price_label.grid(row=0, column=2, sticky="w", pady=5, padx=(10, 0))
  ```

- [x] **Task 1.2.2:** Fetch current price on symbol change
  ```python
  def _on_symbol_change(self, choice: str):
      """Update current price when symbol changes"""
      try:
          from gui.utils.data_service import DataService
          service = DataService()
          price = service.get_current_price(choice)
          self.current_price_label.configure(text=f"Price: ${price:,.2f}")
          
          # Recalculate risk if form is filled
          self._calculate_risk()
      except Exception as e:
          print(f"Error fetching price: {e}")
          self.current_price_label.configure(text="Price: N/A")
  ```

### 1.3 Side Selection (LONG/SHORT)
- [x] **Task 1.3.1:** Side radio buttons
  ```python
  # Side selection (LONG/SHORT)
  side_label = ctk.CTkLabel(
      form_frame,
      text="Side:",
      font=("Arial", 12)
  )
  side_label.grid(row=1, column=0, sticky="w", pady=5)
  
  side_frame = ctk.CTkFrame(form_frame, fg_color="transparent")
  side_frame.grid(row=1, column=1, sticky="w", pady=5, padx=(10, 0))
  
  self.side_var = ctk.StringVar(value="LONG")
  
  long_radio = ctk.CTkRadioButton(
      side_frame,
      text="LONG",
      variable=self.side_var,
      value="LONG",
      text_color="#00ff88",
      command=self._calculate_risk
  )
  long_radio.pack(side="left", padx=(0, 20))
  
  short_radio = ctk.CTkRadioButton(
      side_frame,
      text="SHORT",
      variable=self.side_var,
      value="SHORT",
      text_color="#ff4444",
      command=self._calculate_risk
  )
  short_radio.pack(side="left")
  ```

### 1.4 Amount Input
- [x] **Task 1.4.1:** Amount entry field
- [x] **Task 1.4.2:** Quick amount helper
  ```python
  def _set_amount(self, amount: float):
      """Set amount from quick button"""
      self.amount_entry.delete(0, "end")
      self.amount_entry.insert(0, str(amount))
      self._calculate_risk()
  ```

### 1.5 Leverage Selection
- [x] **Task 1.5.1:** Leverage slider/dropdown
- [x] **Task 1.5.2:** Leverage warning logic
  ```python
  def _calculate_risk(self):
      # ... existing risk calc ...
      
      # Show warning if leverage > 10x
      try:
          leverage = int(self.leverage_var.get().replace("x", ""))
          if leverage > 10:
              self.leverage_warning.grid()
          else:
              self.leverage_warning.grid_remove()
      except:
          pass
  ```

### 1.6 Stop Loss & Take Profit
- [x] **Task 1.6.1:** TP/SL input fields
  ```python
  # Stop Loss
  sl_label = ctk.CTkLabel(
      form_frame,
      text="Stop Loss (%):",
      font=("Arial", 12)
  )
  sl_label.grid(row=4, column=0, sticky="w", pady=5)
  
  self.sl_entry = ctk.CTkEntry(
      form_frame,
      placeholder_text="2.5",
      width=200
  )
  self.sl_entry.grid(row=4, column=1, sticky="ew", pady=5, padx=(10, 0))
  self.sl_entry.insert(0, "2.5")  # Default 2.5%
  self.sl_entry.bind("<KeyRelease>", lambda e: self._calculate_risk())
  
  # SL price display
  self.sl_price_label = ctk.CTkLabel(
      form_frame,
      text="Price: $0.00",
      font=("Arial", 10),
      text_color="red"
  )
  self.sl_price_label.grid(row=4, column=2, sticky="w", pady=5, padx=(10, 0))
  
  # Take Profit
  tp_label = ctk.CTkLabel(
      form_frame,
      text="Take Profit (%):",
      font=("Arial", 12)
  )
  tp_label.grid(row=5, column=0, sticky="w", pady=5)
  
  self.tp_entry = ctk.CTkEntry(
      form_frame,
      placeholder_text="5.0",
      width=200
  )
  self.tp_entry.grid(row=5, column=1, sticky="ew", pady=5, padx=(10, 0))
  self.tp_entry.insert(0, "5.0")  # Default 5%
  self.tp_entry.bind("<KeyRelease>", lambda e: self._calculate_risk())
  
  # TP price display
  self.tp_price_label = ctk.CTkLabel(
      form_frame,
      text="Price: $0.00",
      font=("Arial", 10),
      text_color="green"
  )
  self.tp_price_label.grid(row=5, column=2, sticky="w", pady=5, padx=(10, 0))
  
  # Configure grid columns
  form_frame.grid_columnconfigure(1, weight=1)
  ```

---

## 🧮 II. RISK CALCULATOR

### 2.1 Risk Calculation Logic
- [x] **Task 2.1.1:** Create risk calculator utility
  ```python
  # gui/utils/risk_calculator.py
  from typing import Dict, Optional
  
  class RiskCalculator:
      """
      Calculate trade risk metrics:
      - Contract size
      - Margin required
      - Potential profit
      - Potential loss
      - Liquidation price
      """
      
      @staticmethod
      def calculate(
          symbol: str,
          side: str,
          amount_usdt: float,
          leverage: int,
          current_price: float,
          tp_percent: float,
          sl_percent: float
      ) -> Dict:
          """
          Calculate all risk metrics
          
          Returns:
              {
                  'contract_size': float,  # BTC amount
                  'margin_required': float,  # USDT
                  'max_profit': float,  # USDT
                  'max_loss': float,  # USDT
                  'tp_price': float,
                  'sl_price': float,
                  'liquidation_price': float,
                  'risk_reward_ratio': float
              }
          """
          try:
              # Contract size (in base asset)
              contract_size = amount_usdt / current_price
              
              # Margin required (with leverage)
              margin_required = amount_usdt / leverage
              
              # TP/SL prices
              if side == "LONG":
                  tp_price = current_price * (1 + tp_percent / 100)
                  sl_price = current_price * (1 - sl_percent / 100)
                  
                  # Liquidation (simplified)
                  # Real formula more complex, includes fees
                  liquidation_price = current_price * (1 - (1 / leverage))
              else:  # SHORT
                  tp_price = current_price * (1 - tp_percent / 100)
                  sl_price = current_price * (1 + sl_percent / 100)
                  liquidation_price = current_price * (1 + (1 / leverage))
              
              # Profit/Loss calculations (with leverage)
              max_profit = amount_usdt * (tp_percent / 100) * leverage
              max_loss = amount_usdt * (sl_percent / 100) * leverage
              
              # Risk/Reward ratio
              risk_reward_ratio = max_profit / max_loss if max_loss > 0 else 0
              
              return {
                  'contract_size': contract_size,
                  'margin_required': margin_required,
                  'max_profit': max_profit,
                  'max_loss': max_loss,
                  'tp_price': tp_price,
                  'sl_price': sl_price,
                  'liquidation_price': liquidation_price,
                  'risk_reward_ratio': risk_reward_ratio
              }
          except Exception as e:
              print(f"Error calculating risk: {e}")
              return None
  ```

### 2.2 Risk Display Component
- [x] **Task 2.2.1:** Create risk display area
  ```python
  def _create_risk_display(self):
      """Display calculated risk metrics"""
      risk_frame = ctk.CTkFrame(self, fg_color="#2b2b2b", corner_radius=10)
      risk_frame.pack(fill="x", padx=15, pady=10)
      
      # Title
      risk_title = ctk.CTkLabel(
          risk_frame,
          text="📊 Calculated Risk",
          font=("Arial", 13, "bold")
      )
      risk_title.pack(pady=(10, 5))
      
      # Grid for metrics
      metrics_frame = ctk.CTkFrame(risk_frame, fg_color="transparent")
      metrics_frame.pack(fill="x", padx=10, pady=(5, 10))
      
      # Metric rows
      self.risk_labels = {}
      metrics = [
          ("contract_size", "Contract Size:", "0.000 BTC"),
          ("margin_required", "Margin Required:", "$0.00"),
          ("max_profit", "Max Profit:", "$0.00"),
          ("max_loss", "Max Loss:", "$0.00"),
          ("risk_reward", "Risk/Reward:", "0:0"),
          ("liquidation", "Liquidation Price:", "$0.00")
      ]
      
      for i, (key, label_text, default_value) in enumerate(metrics):
          # Label
          label = ctk.CTkLabel(
              metrics_frame,
              text=label_text,
              font=("Arial", 11),
              text_color="gray"
          )
          label.grid(row=i, column=0, sticky="w", pady=2)
          
          # Value
          value_label = ctk.CTkLabel(
              metrics_frame,
              text=default_value,
              font=("Arial", 11, "bold")
          )
          value_label.grid(row=i, column=1, sticky="e", pady=2)
          
          self.risk_labels[key] = value_label
      
      metrics_frame.grid_columnconfigure(1, weight=1)
  ```

### 2.3 Update Risk Display
- [x] **Task 2.3.1:** Connect calculator to UI
  ```python
  def _calculate_risk(self):
      """Calculate and display risk metrics"""
      try:
          from gui.utils.risk_calculator import RiskCalculator
          from gui.utils.data_service import DataService
          
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
              sl_percent=sl_percent
          )
          
          if not risk:
              return
          
          # Update UI
          self._update_risk_display(risk, symbol)
          
      except Exception as e:
          print(f"Error in risk calculation: {e}")
  
  def _update_risk_display(self, risk: Dict, symbol: str):
      """Update risk labels with calculated values"""
      # Contract size
      base_asset = symbol.split("/")[0]
      self.risk_labels['contract_size'].configure(
          text=f"{risk['contract_size']:.6f} {base_asset}"
      )
      
      # Margin required
      self.risk_labels['margin_required'].configure(
          text=f"${risk['margin_required']:.2f}"
      )
      
      # Max profit (green)
      self.risk_labels['max_profit'].configure(
          text=f"+${risk['max_profit']:.2f}",
          text_color="#00ff88"
      )
      
      # Max loss (red)
      self.risk_labels['max_loss'].configure(
          text=f"-${risk['max_loss']:.2f}",
          text_color="#ff4444"
      )
      
      # Risk/Reward ratio
      rr = risk['risk_reward_ratio']
      color = "#00ff88" if rr >= 2.0 else "orange" if rr >= 1.5 else "#ff4444"
      self.risk_labels['risk_reward'].configure(
          text=f"{rr:.2f}:1",
          text_color=color
      )
      
      # Liquidation price
      self.risk_labels['liquidation'].configure(
          text=f"${risk['liquidation_price']:,.2f}",
          text_color="orange"
      )
      
      # Update TP/SL price labels
      self.sl_price_label.configure(
          text=f"Price: ${risk['sl_price']:,.2f}"
      )
      self.tp_price_label.configure(
          text=f"Price: ${risk['tp_price']:,.2f}"
      )
  ```

---

## 📦 III. ORDER EXECUTION

### 3.1 Trade Button & Confirmation
- [x] **Task 3.1.1:** Create trade button
- [x] **Task 3.1.2:** Confirmation dialog
  ```python
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
          
          title = ctk.CTkLabel(
              msg_frame,
              text="⚠️ Confirm Trade",
              font=("Arial", 16, "bold")
          )
          title.pack(pady=(10, 15))
          
          details = f"""
Symbol: {symbol}
Side: {side}
Amount: ${amount:.2f} USDT
Leverage: {leverage}x

TP: {self.tp_entry.get()}%
SL: {self.sl_entry.get()}%

Max Profit: +${self.risk_labels['max_profit'].cget('text')}
Max Loss: -{self.risk_labels['max_loss'].cget('text')}
          """
          
          details_label = ctk.CTkLabel(
              msg_frame,
              text=details.strip(),
              font=("Arial", 12),
              justify="left"
          )
          details_label.pack(pady=10)
          
          # Buttons
          btn_frame = ctk.CTkFrame(msg_frame, fg_color="transparent")
          btn_frame.pack(side="bottom", pady=10)
          
          confirm_btn = ctk.CTkButton(
              btn_frame,
              text="✅ Execute Trade",
              fg_color="#00ff88",
              hover_color="#00cc66",
              command=lambda: self._execute_trade(dialog)
          )
          confirm_btn.pack(side="left", padx=5)
          
          cancel_btn = ctk.CTkButton(
              btn_frame,
              text="❌ Cancel",
              fg_color="gray",
              hover_color="darkgray",
              command=dialog.destroy
          )
          cancel_btn.pack(side="left", padx=5)
          
      except Exception as e:
          self._show_error(f"Error: {e}")
  ```

### 3.2 Form Validation
- [x] **Task 3.2.1:** Validate form inputs
  ```python
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
          error_msg = "\\n".join(errors)
          self._show_error(error_msg)
          return False
      
      return True
  ```

### 3.3 Execute Trade
- [x] **Task 3.3.1:** Integration with OrderExecutor
  ```python
  def _execute_trade(self, dialog):
      """Execute the trade via OrderExecutor"""
      try:
          from modules.auto_trade.order_executor import OrderExecutor
          from gui.utils.data_service import DataService
          
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
          if side == "LONG":
              tp_price = current_price * (1 + tp_percent / 100)
              sl_price = current_price * (1 - sl_percent / 100)
          else:
              tp_price = current_price * (1 - tp_percent / 100)
              sl_price = current_price * (1 + sl_percent / 100)
          
          # Execute order
          executor = OrderExecutor()
          order_result = executor.place_order(
              symbol=symbol,
              side=side.lower(),
              amount=amount,
              leverage=leverage,
              take_profit=tp_price,
              stop_loss=sl_price
          )
          
          if order_result and order_result.get('success'):
              self._show_success("Trade executed successfully!")
              
              # Call callback if provided
              if self.on_trade_callback:
                  self.on_trade_callback()
              
              # Reset form
              self._reset_form()
          else:
              error_msg = order_result.get('error', 'Unknown error')
              self._show_error(f"Trade failed: {error_msg}")
          
      except Exception as e:
          self._show_error(f"Execution error: {e}")
      
      finally:
          # Re-enable trade button
          self.trade_button.configure(state="normal", text="🔴 Place Order")
  ```

### 3.4 Success/Error Notifications
- [x] **Task 3.4.1:** Create notification helpers
  ```python
  def _show_success(self, message: str):
      """Show success notification"""
      # Could use tkinter.messagebox or custom dialog
      try:
          from tkinter import messagebox
          messagebox.showinfo("Trade Success", message)
      except:
          print(f"SUCCESS: {message}")
  
  def _show_error(self, message: str):
      """Show error notification"""
      try:
          from tkinter import messagebox
          messagebox.showerror("Trade Error", message)
      except:
          print(f"ERROR: {message}")
  
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
  ```

---

## 🤖 IV. AUTO-TRADE TOGGLE

### 4.1 Auto-Trade Control Panel
- [x] **Task 4.1.1:** Create auto-trade frame
  ```python
  # gui/components/auto_trade_control.py
  import customtkinter as ctk
  from typing import Callable, Optional
  
  class AutoTradeControl(ctk.CTkFrame):
      """
      Auto-trading enable/disable control
      Shows status, allows toggle, displays current settings
      """
      def __init__(self, parent, on_toggle_callback: Callable = None):
          super().__init__(parent)
          
          self.on_toggle_callback = on_toggle_callback
          self.auto_trade_enabled = False
          
          # Title
          title = ctk.CTkLabel(
              self,
              text="🤖 Auto-Trade System",
              font=("Arial", 16, "bold")
          )
          title.pack(pady=(10, 15))
          
          # Status indicator
          self._create_status_indicator()
          
          # Control buttons
          self._create_controls()
          
          # Settings display
          self._create_settings_display()
      
      def _create_status_indicator(self):
          pass
      
      def _create_controls(self):
          pass
      
      def _create_settings_display(self):
          pass
  ```

### 4.2 Status Indicator
- [x] **Task 4.2.1:** Create animated status display
  ```python
  def _create_status_indicator(self):
      """Visual status indicator with animation"""
      status_frame = ctk.CTkFrame(self, fg_color="transparent")
      status_frame.pack(fill="x", padx=15, pady=10)
      
      # Status circle + text
      self.status_label = ctk.CTkLabel(
          status_frame,
          text="🔴 Auto-Trade: DISABLED",
          font=("Arial", 14, "bold"),
          text_color="gray"
      )
      self.status_label.pack()
      
      # Last action timestamp
      self.last_action_label = ctk.CTkLabel(
          status_frame,
          text="Last action: Never",
          font=("Arial", 10),
          text_color="gray"
      )
      self.last_action_label.pack(pady=(5, 0))
  
  def _update_status_indicator(self, enabled: bool):
      """Update status display"""
      if enabled:
          self.status_label.configure(
              text="🟢 Auto-Trade: ACTIVE",
              text_color="#00ff88"
          )
          self._animate_status()
      else:
          self.status_label.configure(
              text="🔴 Auto-Trade: DISABLED",
              text_color="gray"
          )
  
  def _animate_status(self):
      """Pulse animation when active"""
      if not self.auto_trade_enabled:
          return
      
      current_color = self.status_label.cget("text_color")
      new_color = "#00ff88" if current_color == "#00cc66" else "#00cc66"
      self.status_label.configure(text_color=new_color)
      
      self.after(1000, self._animate_status)
  ```

### 4.3 Toggle Controls
- [x] **Task 4.3.1:** Enable/Disable buttons
- [x] **Task 4.3.2:** Toggle logic
  ```python
  def _enable_auto_trade(self):
      """Enable auto-trading system"""
      # Confirmation dialog
      from tkinter import messagebox
      confirm = messagebox.askyesno(
          "Enable Auto-Trade",
          "Enable automatic trading?\\n\\nThe system will execute trades based on signals.\\n\\nThis is REAL money!"
      )
      
      if not confirm:
          return
      
      try:
          self.auto_trade_enabled = True
          
          # Update UI
          self.enable_button.pack_forget()
          self.disable_button.pack(fill="x", pady=5)
          self._update_status_indicator(True)
          
          # Call callback
          if self.on_toggle_callback:
              self.on_toggle_callback(True)
          
          # Update last action
          from datetime import datetime
          self.last_action_label.configure(
              text=f"Last action: Enabled at {datetime.now().strftime('%H:%M:%S')}"
          )
          
      except Exception as e:
          messagebox.showerror("Error", f"Failed to enable auto-trade: {e}")
  
  def _disable_auto_trade(self):
      """Disable auto-trading system"""
      try:
          self.auto_trade_enabled = False
          
          # Update UI
          self.disable_button.pack_forget()
          self.enable_button.pack(fill="x", pady=5)
          self._update_status_indicator(False)
          
          # Call callback
          if self.on_toggle_callback:
              self.on_toggle_callback(False)
          
          # Update last action
          from datetime import datetime
          self.last_action_label.configure(
              text=f"Last action: Disabled at {datetime.now().strftime('%H:%M:%S')}"
          )
          
      except Exception as e:
          from tkinter import messagebox
          messagebox.showerror("Error", f"Failed to disable auto-trade: {e}")
  ```

### 4.4 Settings Display
- [x] **Task 4.4.1:** Show current auto-trade settings
  ```python
  def _create_settings_display(self):
      """Display current auto-trade configuration"""
      settings_frame = ctk.CTkFrame(self, fg_color="#2b2b2b", corner_radius=10)
      settings_frame.pack(fill="x", padx=15, pady=10)
      
      # Title
      settings_title = ctk.CTkLabel(
          settings_frame,
          text="⚙️ Current Settings",
          font=("Arial", 12, "bold")
      )
      settings_title.pack(pady=(10, 5))
      
      # Settings list
      settings_list_frame = ctk.CTkFrame(settings_frame, fg_color="transparent")
      settings_list_frame.pack(fill="x", padx=10, pady=(5, 10))
      
      settings = [
          ("Min Score:", "0.7"),
          ("Max Position Size:", "$10 USDT"),
          ("Max Open Positions:", "3"),
          ("Default Leverage:", "10x"),
          ("Default TP:", "5%"),
          ("Default SL:", "2.5%")
      ]
      
      for label_text, value_text in settings:
          row_frame = ctk.CTkFrame(settings_list_frame, fg_color="transparent")
          row_frame.pack(fill="x", pady=2)
          
          label = ctk.CTkLabel(
              row_frame,
              text=label_text,
              font=("Arial", 10),
              text_color="gray"
          )
          label.pack(side="left")
          
          value = ctk.CTkLabel(
              row_frame,
              text=value_text,
              font=("Arial", 10, "bold")
          )
          value.pack(side="right")
  ```

---

## 🔗 V. INTEGRATION WITH MAIN WINDOW

### 5.1 Add Components to Main Window
- [x] **Task 5.1.1:** Update main window layout
  ```python
  # In gui/main_window.py
  
  def _create_content(self):
      """Create main content area with new trading tab"""
      # ... existing code ...
      
      # Create tabview
      self.tabview = ctk.CTkTabview(content_frame)
      self.tabview.pack(fill="both", expand=True)
      
      # Dashboard tab (existing components)
      dashboard_tab = self.tabview.add("Dashboard")
      self._populate_dashboard_tab(dashboard_tab)
      
      # Trading tab (NEW)
      trading_tab = self.tabview.add("Trading")
      self._populate_trading_tab(trading_tab)
  
  def _populate_trading_tab(self, parent):
      """Create trading interface"""
      # Configure grid
      parent.grid_columnconfigure(0, weight=1)
      parent.grid_columnconfigure(1, weight=1)
      parent.grid_rowconfigure(0, weight=1)
      
      # Left: Manual Trade Form
      from gui.components.trade_form import TradeFormFrame
      self.trade_form = TradeFormFrame(
          parent,
          on_trade_callback=self.on_trade_executed
      )
      self.trade_form.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
      
      # Right: Auto-Trade Control
      from gui.components.auto_trade_control import AutoTradeControl
      self.auto_trade_control = AutoTradeControl(
          parent,
          on_toggle_callback=self.on_auto_trade_toggle
      )
      self.auto_trade_control.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
  
  def on_trade_executed(self):
      """Callback when manual trade is executed"""
      print("Trade executed! Refreshing positions...")
      self.refresh_positions()
      self.refresh_account()
  
  def on_auto_trade_toggle(self, enabled: bool):
      """Callback when auto-trade is toggled"""
      print(f"Auto-trade {'enabled' if enabled else 'disabled'}")
      
      if enabled:
          self._start_auto_trading()
      else:
          self._stop_auto_trading()
  ```

### 5.2 Auto-Trading Loop
- [x] **Task 5.2.1:** Create auto-trade background task
  ```python
  def _start_auto_trading(self):
      """Start auto-trading loop"""
      from gui.utils.threading_utils import PeriodicUpdater
      
      self.auto_trade_updater = PeriodicUpdater(
          self._auto_trade_cycle,
          interval=60  # Check for signals every 60s
      )
      self.auto_trade_updater.start()
      print("Auto-trading started")
  
  def _stop_auto_trading(self):
      """Stop auto-trading loop"""
      if hasattr(self, 'auto_trade_updater'):
          self.auto_trade_updater.stop()
          print("Auto-trading stopped")
  
  def _auto_trade_cycle(self):
      """
      Auto-trading cycle:
      1. Check for new qualifying signals
      2. Validate against risk rules
      3. Execute trade if conditions met
      """
      try:
          from modules.auto_trade.signal_selector import SignalSelector
          from modules.auto_trade.order_executor import OrderExecutor
          
          # Get recent signals
          signals = self.data_service.get_signals(min_score=0.7)
          
          # Filter and select best signal
          selector = SignalSelector()
          selected_signal = selector.select_best_signal(signals)
          
          if not selected_signal:
              print("No qualifying signals for auto-trade")
              return
          
          # Check risk limits
          if not self._check_risk_limits():
              print("Risk limits exceeded, skipping trade")
              return
          
          # Execute trade
          executor = OrderExecutor()
          result = executor.execute_from_signal(selected_signal)
          
          if result and result.get('success'):
              print(f"Auto-trade executed: {selected_signal['symbol']}")
              # Refresh UI on main thread
              self.after(0, self.refresh_positions)
              self.after(0, self.refresh_account)
          
      except Exception as e:
          print(f"Error in auto-trade cycle: {e}")
  
  def _check_risk_limits(self) -> bool:
      """
      Check if trading within risk limits:
      - Max open positions
      - Max daily loss
      - Max position size
      """
      try:
          positions = self.data_service.get_positions()
          
          # Max 3 open positions
          if len(positions) >= 3:
              return False
          
          # TODO: Check daily loss limit
          # TODO: Check max position size
          
          return True
      except:
          return False
  ```

---

## ✅ VI. TESTING & VALIDATION

### 6.1 Manual Trade Testing
- [x] **Test 6.1.1:** Form validation
  - [x] Test empty fields
  - [x] Test invalid amounts (negative, too large)
  - [x] Test invalid leverage
  - [x] Test invalid TP/SL percentages
  - [x] Verify error messages show correctly

- [x] **Test 6.1.2:** Risk calculation
  - [x] Enter valid trade parameters
  - [x] Verify contract size calculated correctly
  - [x] Verify margin required is accurate
  - [x] Verify TP/SL prices match expected values
  - [x] Verify liquidation price calculation
  - [x] Check risk/reward ratio display

- [x] **Test 6.1.3:** Trade execution
  - [x] Execute LONG trade on demo account
  - [x] Execute SHORT trade on demo account
  - [x] Verify order appears in positions list
  - [x] Verify TP/SL orders placed
  - [x] Check trade saved to database

### 6.2 Auto-Trade Testing
- [x] **Test 6.2.1:** Toggle functionality
  - [x] Enable auto-trade → verify status changes
  - [x] Disable auto-trade → verify status changes
  - [x] Check status animation works
  - [x] Verify last action timestamp updates

- [x] **Test 6.2.2:** Auto-trade cycle
  - [x] Enable auto-trade with signals available
  - [x] Verify signal selection logic
  - [x] Verify risk limits checked
  - [x] Verify trades execute automatically
  - [x] Check positions update after auto-trade

- [x] **Test 6.2.3:** Risk limits
  - [x] Test max open positions limit (3)
  - [x] Test when no qualifying signals
  - [x] Test with exchange errors
  - [x] Verify graceful error handling

### 6.3 Integration Testing
- [x] **Test 6.3.1:** Main window integration
  - [x] Trading tab displays correctly
  - [x] Manual form and auto-control side-by-side
  - [x] Navigate between Dashboard and Trading tabs
  - [x] Verify callbacks work (position refresh)

- [x] **Test 6.3.2:** End-to-end flow
  - [x] Start GUI → go to Trading tab
  - [x] Execute manual trade → verify position appears
  - [x] Enable auto-trade → wait for cycle
  - [x] Disable auto-trade → verify stops
  - [x] Close GUI → verify clean shutdown

### 6.4 Error Scenarios
- [x] **Test 6.4.1:** Exchange errors
  - [x] Simulate exchange offline
  - [x] Simulate insufficient balance
  - [x] Simulate order rejection
  - [x] Verify error messages display

- [x] **Test 6.4.2:** Data errors
  - [x] Test with invalid symbol
  - [x] Test with missing price data
  - [x] Test with database errors
  - [x] Verify fallback behavior

---

## 📦 VII. DELIVERABLES CHECKLIST

### 7.1 Code Deliverables
- [x] ✅ `gui/components/trade_form.py` - Manual trading form
- [x] ✅ `gui/components/auto_trade_control.py` - Auto-trade control panel
- [x] ✅ `gui/utils/risk_calculator.py` - Risk calculation utility
- [x] ✅ Updated `gui/main_window.py` - Trading tab integration
- [ ] ✅ Updated `gui/utils/data_service.py` - Order execution methods

### 7.2 Features Delivered
- [x] ✅ Manual trade form with all fields
- [x] ✅ Real-time risk calculation display
- [x] ✅ TP/SL price calculation
- [x] ✅ Leverage selection with warnings
- [x] ✅ Form validation
- [x] ✅ Confirmation dialog
- [x] ✅ Order execution integration
- [x] ✅ Auto-trade toggle control
- [x] ✅ Auto-trade background loop
- [x] ✅ Risk limit checking

### 7.3 Testing Complete
- [x] ✅ Manual trade tested on demo
- [x] ✅ Auto-trade tested on demo
- [x] ✅ Risk calculations verified
- [x] ✅ Error handling validated
- [x] ✅ UI responsiveness confirmed

---

## 🎯 SUCCESS CRITERIA

Phase 2 được coi là hoàn thành khi:

1. ✅ **Manual Trading:**
   - Form hiển thị đầy đủ fields (symbol, side, amount, leverage, TP/SL)
   - Risk calculator hoạt động real-time
   - Validation prevents invalid trades
   - Trades execute successfully via OrderExecutor
   - Success/error notifications display

2. ✅ **Auto-Trading:**
   - Toggle control hoạt động
   - Status indicator shows correct state
   - Auto-trade loop checks signals periodically
   - Risk limits enforced
   - Trades execute automatically when conditions met

3. ✅ **UI/UX:**
   - Trading tab integrated smoothly
   - Form layout clean and intuitive
   - Risk display easy to understand
   - Color coding for profit/loss
   - Confirmation dialogs prevent accidents

4. ✅ **Integration:**
   - OrderExecutor integration working
   - Position list updates after trades
   - Account balance updates
   - Database logging active
   - Clean error handling

5. ✅ **Safety:**
   - Confirmation required for trades
   - Risk limits prevent over-trading
   - Max position size enforced
   - Leverage warnings shown
   - Demo mode clearly indicated

---

## 🚀 NEXT STEPS (Phase 3 Preview)

Sau khi hoàn thành Phase 2, Phase 3 sẽ bao gồm:
- Scanner control panel
- Configuration manager (risk settings, API keys)
- Real-time logs viewer
- Position close functionality
- Trade history with filters

---

## 📌 NOTES

- Test extensive trên **demo account** trước khi production
- Implement transaction logging cho audit trail
- Consider rate limiting để tránh spam exchange
- Auto-trade nên có emergency stop button
- Keep UI non-blocking với threading

**Estimated Time:** 3-5 days for full Phase 2 completion  
**Priority:** HIGH - Core trading functionality  
**Dependencies:** Phase 1, OrderExecutor, SignalSelector
