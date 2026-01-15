# ATC Visualizer - Complete Guide

Web-based visualization tool for Adaptive Trend Classification (ATC) algorithm.

## 🚀 Quick Start (Recommended)

### Option 1: Using Management Scripts (New Architecture)
```bash
# From project root
cd web
python scripts/start_app.py atc_visualizer
```

### Option 2: Direct Python (Cross-platform)
```bash
python run_atc_visualizer.py
```

### Option 3: Manual Start
```bash
# Terminal 1 - Backend
cd web/apps/atc_visualizer/backend
python main.py

# Terminal 2 - Frontend
cd web/apps/atc_visualizer/frontend
npm run dev
```

## 📋 Command Options

### Install Dependencies Only
```bash
python run_atc_visualizer.py --install
# or
python run_atc_visualizer.py -i
```

### Skip Dependency Check
```bash
python run_atc_visualizer.py --no-install
```

### Skip npm/Node.js Check
```bash
python run_atc_visualizer.py --skip-npm-check
```

### Port Management
```bash
# Kill processes on ports 8002, 5174
python web/scripts/kill_ports.py 8002 5174

# Check which ports are in use
python web/scripts/check_ports.py
```

## 🌐 Access Points

After startup, access:

- **Frontend**: http://localhost:5174
- **Backend API**: http://localhost:8002
- **Health Check**: http://localhost:8002/api/health
- **API Docs (Swagger)**: http://localhost:8002/docs
- **API Docs (ReDoc)**: http://localhost:8002/redoc

## ⏸️ Stop Servers

Press `Ctrl+C` in the terminal to stop both servers gracefully.

## 📁 What This Does

The startup script:

1. ✅ Checks if Python and Node.js are installed
2. 🔧 Installs/Updates backend dependencies (pip install)
3. 🔧 Installs/Updates frontend dependencies (npm install)
4. 🚀 Starts FastAPI backend on port 8002
5. 🎨 Starts Vue.js + Vite frontend on port 5174
6. 📊 Displays all access URLs
7. ⏸️ Stops both servers on Ctrl+C

## ⭐ Features

- **Real-time OHLCV chart visualization** with ApexCharts
- **Display of all 6 Moving Average types** (EMA, HMA, WMA, DEMA, LSMA, KAMA)
- **9 Moving Averages per type** with different length offsets
- **Signal visualization** with arrow markers (↑ for buy, ↓ for sell)
- **Multiple timeframe support** (1m to 1d)
- **Toggle visibility** for each MA type and signal
- **Dark mode optimized** for trading analysis

## 📁 Project Structure

```
web/apps/atc_visualizer/
├── README.md                          # This file
├── backend/
│   ├── main.py                       # FastAPI REST API entry point
│   ├── config.py                     # App configuration
│   ├── services/
│   │   ├── atc_service.py            # ATC computation service
│   │   └── __init__.py
│   ├── api/                          # API routes
│   │   └── __init__.py
│   └── requirements.txt
├── frontend/
│   ├── package.json
│   ├── vite.config.js
│   ├── src/
│   │   ├── App.vue                   # Main Vue component
│   │   ├── main.js                   # Vue app entry point
│   │   ├── components/
│   │   │   ├── ParameterPanel.vue    # Symbol/timeframe selector
│   │   │   ├── SignalLegend.vue      # MA/signal legend
│   │   │   └── ChartView.vue         # ApexCharts wrapper
│   │   ├── services/
│   │   │   └── api.js                # API client
│   │   └── utils/
│   │       └── chartHelper.js        # Chart formatting utilities
│   ├── public/
│   └── dist/                         # Production build
```

## 🔧 Installation

### Backend (Python)

1. Install Python dependencies:
```bash
cd web/apps/atc_visualizer/backend
pip install -r requirements.txt
```

2. Start the API server:
```bash
python main.py
```

The API will be available at `http://localhost:8002`

### Frontend (Node.js)

1. Install Node.js dependencies:
```bash
cd web/apps/atc_visualizer/frontend
npm install
```

2. Start the development server:
```bash
npm run dev
```

The application will be available at `http://localhost:5174`

### Production Build

```bash
cd web/apps/atc_visualizer/frontend
npm run build
```

## 📖 Usage

1. **Enter a trading symbol** (e.g., `BTC/USDT`, `ETH/USDT`)
2. **Select a timeframe** (1m, 5m, 15m, 1h, 4h, 1d)
3. **Set the number of candles** to display (100-5000)
4. **Click "Load Data"** to fetch and visualize

### Understanding the Chart

- **Candlesticks**: OHLCV price data (green = bullish, red = bearish)
- **Colored Lines**: Moving Averages
  - EMA: Green (#00E396)
  - HMA: Yellow (#FEB019)
  - WMA: Purple (#775DD0)
  - DEMA: Blue (#008FFB)
  - LSMA: Red (#FF4560)
  - KAMA: Purple (#775DD0)
- **Triangles**: Trading signals
  - ↑ (Green): Buy signal (signal = 1)
  - ↓ (Red): Sell signal (signal = -1)

### ATC Algorithm Visualization

The visualizer displays:

1. **OHLCV Data**: Complete candlestick chart
2. **Moving Averages**: 9 MAs for each type (MA, MA1-MA4, MA_1-MA_4)
3. **Signals**:
   - Average_Signal: Combined signal from all MA types
   - Individual MA signals: EMA_Signal, HMA_Signal, etc.

## 🔌 API Endpoints

- `GET /` - Health check with app info
- `GET /api/health` - API health check
- `GET /api/symbols` - List available trading symbols
- `GET /api/ohlcv` - Fetch OHLCV data
- `GET /api/atc-signals` - Compute ATC signals
- `GET /api/moving-averages` - Get all Moving Averages
- `GET /api/timeframes` - Get available timeframes

## 🛠️ Technologies

- **Backend**: FastAPI, Pandas, NumPy
- **Frontend**: Vue.js 3, ApexCharts, Vite
- **Data**: ccxt (exchange integration), pandas_ta (technical indicators)
- **API Server**: Uvicorn

## 📚 API Documentation

When the backend is running, access the interactive API documentation at:
- **Swagger UI**: http://localhost:8002/docs
- **ReDoc**: http://localhost:8002/redoc

## ❌ Common Issues & Solutions

### Port Already in Use (Errno 10048)

**Solution:**
```bash
python web/scripts/kill_ports.py 8002 5174
python scripts/start_app.py atc_visualizer
```

### Frontend Shows "Page Not Found"

**Causes:**
1. Frontend not actually running
2. Wrong URL
3. Browser cache

**Solutions:**
1. Check terminal for `➜  Local: http://localhost:5174/`
2. Clear browser cache (Ctrl+Shift+Delete)
3. Try Incognito/Private mode

### Backend Returns {"detail":"Not Found"}

**Cause:** Accessing root `/` instead of API endpoints

**Correct URLs:**
- ✅ http://localhost:8002/api/health
- ✅ http://localhost:8002/docs
- ✅ http://localhost:8002/api/ohlcv?symbol=BTC/USDT

### ModuleNotFoundError

**Solution:**
```bash
# Run from project root
cd C:\Users\Admin\Desktop\i-ching\crypto-probability
python scripts/start_app.py atc_visualizer
```

### Backend Fails to Start

- Check Python version: `python --version` (needs 3.9+)
- Install dependencies manually:
  ```bash
  cd web/apps/atc_visualizer/backend
  pip install -r requirements.txt
  python main.py
  ```

### Frontend Fails to Start

- Check Node.js version: `node --version` (needs 18+)
- Install dependencies manually:
  ```bash
  cd web/apps/atc_visualizer/frontend
  npm install
  npm run dev
  ```

## 🔧 Port Configuration

### Change Backend Port
Edit `web/apps/atc_visualizer/backend/config.py`:
```python
BACKEND_PORT = 8003  # Change to 8003
```

### Change Frontend Port
Edit `web/apps/atc_visualizer/frontend/vite.config.js`:
```javascript
server: {
  port: 5175,  // Change to 5175
  proxy: {
    '/api': {
      target: 'http://localhost:8003',  // Match backend port
      changeOrigin: true
    }
  }
}
```

## 💡 Tips

1. **Always kill ports before starting** - prevents "port in use" errors
2. **Use two separate terminals** - easier to see logs from both servers
3. **Check API docs first** - http://localhost:8002/docs shows all endpoints
4. **Clear browser cache** if frontend seems broken
5. **Restart IDE** after path changes - LSP errors may be stale

## 🎯 Key Files

| File | Purpose |
|------|----------|
| `scripts/start_app.py` | ⭐ Main Python entry point |
| `web/scripts/kill_ports.py` | Kill processes on ports |
| `web/apps/atc_visualizer/backend/main.py` | FastAPI REST API |
| `web/apps/atc_visualizer/frontend/src/App.vue` | Main Vue component |
| `web/apps/atc_visualizer/README.md` | This documentation |

## 📝 Notes

- Ensure backend API is running before starting frontend
- The visualizer uses modules from parent `modules/` directory
- First data load may take 10-30 seconds depending on network
- FastAPI provides automatic validation and type checking for all parameters
- Port 8002 (backend) and 5174 (frontend) are allocated for this app

## 🔗 Related Links

- [Main Web Documentation](../README.md)
- [Architecture Overview](../docs/ARCHITECTURE.md)
- [Adding New Apps Guide](../docs/ADDING_NEW_APP.md)
- [Migration Summary](../MIGRATION_SUMMARY.md)
