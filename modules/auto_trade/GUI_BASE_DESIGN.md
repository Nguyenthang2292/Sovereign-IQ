# 🎨 Auto Trade Dashboard - Đề Xuất GUI

## 🏗️ Kiến Trúc Đề Xuất (Python GUI Desktop)

- **GUI Framework:** CustomTkinter (modern, đẹp, dễ sử dụng)
- **Alternative:** Tkinter (built-in, không cần cài thêm)
- **Backend:** Integrate trực tiếp với existing modules
- **Real-time:** Threading cho updates không block UI
- **Pattern:** Single window application với tabs/frames
- **Deployment:** Standalone Python app (có thể package thành .exe)

### Ưu Điểm Kiến Trúc Này:
✅ **Đơn giản**: 1 file Python, chạy ngay, không cần server  
✅ **Dễ triển khai**: `pip install customtkinter` và chạy  
✅ **Nhẹ**: Không cần browser, không cần port 8003/5175  
✅ **Tích hợp dễ**: Gọi trực tiếp các modules hiện có  
✅ **Package được**: PyInstaller → tạo .exe cho Windows  
✅ **Offline hoàn toàn**: Không cần network stack phức tạp

## 📊 Các Chức Năng Chính

### 1. Dashboard Overview (Trang Chủ)

```
┌─────────────────────────────────────────┐
│  Account Overview                        │
│  • Balance: $0.89 USDT                   │
│  • Available: $0.89                      │
│  • Margin Used: $0.00                    │
│  • Unrealized P&L: $0.00                │
│  • Daily P&L: +$0.00 (0%)               │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│  Quick Stats                             │
│  • Open Positions: 0                     │
│  • Today's Trades: 0                     │
│  • Win Rate: 0%                          │
│  • Mode: 🔴 PRODUCTION                   │
└─────────────────────────────────────────┘
```

### 2. Live Signal Monitor (Quan Trọng!)

```
┌─────────────────────────────────────────┐
│  Live Signals (Auto-refresh 30s)        │
├─────────────────────────────────────────┤
│  Symbol  │ Signal │ Score │ Time │ Action│
│  BTC/USDT│ LONG  │ 0.85  │ 14:30│ [Trade]│
│  ETH/USDT│ SHORT │ 0.72  │ 14:28│ [Trade]│
│  SOL/USDT│ NEUTRAL│ 0.45 │ 14:25│   -   │
└─────────────────────────────────────────┘
```

**Filters:**

- [x] LONG signals only
- [x] Score > 0.7
- [ ] XGBoost filtered

### 3. Position Management

```
┌─────────────────────────────────────────┐
│  Open Positions                          │
├─────────────────────────────────────────┤
│ Symbol    │ Side │ Size │ Entry │ P&L   │
│ BTC/USDT  │ LONG │ 0.01 │ 78000 │ +$50  │
│           │      │      │       │[Close]│
└─────────────────────────────────────────┘
```

**Position Details:**

- Leverage: 10x
- TP: $82,000 (+5%)
- SL: $76,000 (-2.5%)
- Liquidation: $71,000

### 4. Trade History & Performance

```
┌─────────────────────────────────────────┐
│  Trade History (Last 30 days)           │
├─────────────────────────────────────────┤
│ Date  │ Symbol │ Side │ P&L │ ROI      │
│ 02/03 │ BTC/USDT│ LONG│ +$10│ +12.5%   │
│ 02/02 │ ETH/USDT│ SHORT│ -$5│ -3.2%    │
└─────────────────────────────────────────┘
```

**Performance Metrics:**

- Total Trades: 50
- Win Rate: 62%
- Avg Win: $15.30
- Avg Loss: -$8.20
- Profit Factor: 1.87
- Sharpe Ratio: 1.45

### 5. Configuration Manager

```
┌─────────────────────────────────────────┐
│  Trading Configuration                   │
├─────────────────────────────────────────┤
│  Mode: ○ Demo  ● Production  ○ Dry Run  │
│                                          │
│  Risk Management:                        │
│  • Max Position Size: [10] USDT         │
│  • Max Risk Per Trade: [2] %             │
│  • Max Open Positions: [3]               │
│  • Stop Loss: [2.5] %                    │
│  • Take Profit: [5] %                    │
│                                          │
│  Signal Filters:                         │
│  ☑ Enable XGBoost Filter                │
│  ☑ Min Confidence: [0.7]                 │
│  ☑ Volume Filter                         │
│                                          │
│  [Save Config]  [Reset]                  │
└─────────────────────────────────────────┘
```

### 6. Scanner Status & Control

```
┌─────────────────────────────────────────┐
│  Scanner Control                         │
├─────────────────────────────────────────┤
│  Status: 🟢 Running                      │
│  Last Scan: 2 minutes ago                │
│  Symbols Scanned: 150                    │
│  Signals Found: 5                        │
│                                          │
│  [⏸ Pause] [▶ Resume] [⚙ Settings]      │
│                                          │
│  Scan Interval: [30] seconds             │
│  Symbols: [Auto-detect ▼]               │
└─────────────────────────────────────────┘
```

### 7. Backtest Module (Integrated)

```
┌─────────────────────────────────────────┐
│  Backtesting                             │
├─────────────────────────────────────────┤
│  Symbol: [BTC/USDT ▼]                    │
│  Timeframe: [1h ▼]                       │
│  Period: [2024-01-01] to [2024-12-31]    │
│  Initial Balance: [$1000]                │
│                                          │
│  Strategy Config:                        │
│  • Signal Threshold: [0.7]               │
│  • Stop Loss: [2.5%]                     │
│  • Take Profit: [5%]                     │
│                                          │
│  [🚀 Run Backtest]                       │
│                                          │
│  Results:                                │
│  • Total Return: +45.2%                  │
│  • Max Drawdown: -12.5%                  │
│  • Win Rate: 58%                         │
│  • Trades: 120                           │
│  [📊 View Charts]                        │
└─────────────────────────────────────────┘
```

### 8. Live Logs Viewer

```
┌─────────────────────────────────────────┐
│  System Logs (Live)                      │
├─────────────────────────────────────────┤
│ [14:30:25] ✅ Signal detected: BTC LONG  │
│ [14:30:26] 🔄 Creating order...          │
│ [14:30:27] ✅ Order filled at $78,500    │
│ [14:30:28] 📊 TP/SL orders placed        │
│ [14:25:10] ⚠️  Low balance warning       │
│ [14:20:05] 🔄 Scanner cycle completed    │
└─────────────────────────────────────────┘
```

**Filters:** [x] Info [x] Warning [x] Error  
**Actions:** [Clear] [Export] [Auto-scroll ▼]

### 9. Manual Trading Interface

```
┌─────────────────────────────────────────┐
│  Manual Trade                            │
├─────────────────────────────────────────┤
│  Symbol: [BTC/USDT ▼]                    │
│  Current Price: $78,756.80               │
│                                          │
│  Side: ○ LONG  ○ SHORT                   │
│  Amount: [10] USDT                       │
│  Leverage: [10x ▼]                       │
│                                          │
│  Stop Loss: [76,000] (-3.5%)             │
│  Take Profit: [82,000] (+4.1%)           │
│                                          │
│  Calculated:                             │
│  • Contract Size: 0.127 BTC              │
│  • Margin Required: $1.00                │
│  • Max Loss: -$3.50                      │
│  • Max Profit: +$4.10                    │
│                                          │
│  [🔴 Place Order]                        │
└─────────────────────────────────────────┘
```

### 10. Risk Dashboard

```
┌─────────────────────────────────────────┐
│  Risk Metrics                            │
├─────────────────────────────────────────┤
│  Portfolio Risk: Low                     │
│  [████░░░░░░] 40%                        │
│                                          │
│  • Total Exposure: $0.00 (0%)            │
│  • Margin Level: Healthy (∞%)            │
│  • Daily Loss Limit: 0/5%                │
│  • Max Drawdown: 0%                      │
│                                          │
│  Warnings:                               │
│  ⚠️  Low account balance ($0.89)         │
│  ℹ️  No open positions                   │
└─────────────────────────────────────────┘
```

## 🎨 Layout Design Suggestion

```
┌─────────────────────────────────────────────────────────┐
│  Auto Trade Dashboard          [●] PROD    [⚙] Settings │
├────────┬────────────────────────────────────────────────┤
│ 📊 Dash│  ┌─ Account Overview ──────────────────────┐   │
│ 🎯 Sig │  │ Balance: $0.89  P&L: $0.00  Pos: 0      │   │
│ 📈 Pos │  └──────────────────────────────────────────┘   │
│ 📜 Hist│                                                 │
│ ⚙️  Cfg │  ┌─ Live Signals ──────────────────────────┐   │
│ 🤖 Scan│  │ BTC/USDT  LONG  0.85  14:30  [Trade]   │ │
│ 🧪 Back│  │ ETH/USDT  SHORT 0.72  14:28  [Trade]   │ │
│ 📝 Logs│  └──────────────────────────────────────────┘ │
│ 💱 Trade│                                               │
│ 🛡️  Risk│  ┌─ Open Positions ────────────────────────┐ │
│        │  │ (No open positions)                      │ │
│        │  └──────────────────────────────────────────┘ │
└────────┴────────────────────────────────────────────────┘
```

## 🚀 Implementation Plan (Python GUI)

### Phase 1: Basic GUI Setup & Data Display
- Setup CustomTkinter window
- Create main layout với tabs/frames
- Display Account Overview (Balance, P&L)
- Display Quick Stats (Positions, Trades, Win Rate)
- **Time:** 1-2 days

### Phase 2: Live Signal Monitor
- Signal list với auto-refresh (threading)
- Signal filters (LONG/SHORT, Score threshold)
- Color coding cho signal types
- **Time:** 1-2 days

### Phase 3: Position Management Display
- Open positions list
- Real-time P&L updates
- TP/SL/Liquidation display
- Position details panel
- **Time:** 1 day

### Phase 4: Trade Execution
- Manual trade form
- Auto-trade toggle
- Risk calculations
- Order execution integration
- **Time:** 2-3 days

### Phase 5: Advanced Features
- Trade history table
- Performance charts (matplotlib)
- Configuration panel
- Logs viewer
- **Time:** 2-3 days




## 💡 Tính Năng Đặc Biệt (Python GUI)

- **Real-time Updates:** Background threading cho signal/position updates
- **Lightweight:** Chỉ cần Python, không cần browser hay server
- **Easy Setup:** `pip install customtkinter` → chạy ngay
- **Dark/Light Theme:** CustomTkinter có sẵn theme switching
- **System Tray:** Minimize to tray, chạy background
- **Notifications:** Desktop notifications cho signals/trades (Windows/Mac/Linux)
- **Export Data:** Export trades thành CSV/Excel (pandas)
- **Portable:** Package thành .exe standalone (PyInstaller)
- **Multi-language:** English + Tiếng Việt (dễ implement)
- **Offline:** Hoạt động hoàn toàn offline, chỉ cần API key

