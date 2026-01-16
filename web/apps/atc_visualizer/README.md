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

## 📊 ATC Algorithm Parameters

### Moving Average Lengths
- **ema_len, hma_len, wma_len, dema_len, lsma_len, kama_len**: Base length for each MA type (default: 28)
- Can be adjusted independently for different MA types

### Robustness Parameter
Controls the spread of MA offsets from base length:
- **Narrow**: Small offsets (±1, ±2, ±3, ±4)
- **Medium** (default): Medium offsets (±1, ±2, ±4, ±6)
- **Wide**: Large offsets (±1, ±3, ±5, ±7)

### Lambda Parameter
- **lambda_param** (0.0 - 1.0, default: 0.02)
- Controls the adaptation rate for signal smoothing

### Decay Parameter
- **decay** (0.0 - 1.0, default: 0.03)
- Exponential decay factor for signal weighting

### Cutout Parameter
- **cutout** (integer, default: 0)
- Number of candles to exclude from recent calculations

## 🔢 MA Offset Pattern

For each MA type, 9 MAs are calculated with different length offsets:

| MA Name | Length Formula | Purpose |
|---------|---------------|---------|
| MA | base_length | Main MA at base length |
| MA1 | base_length + offset1 | Faster MA |
| MA2 | base_length + offset2 | |
| MA3 | base_length + offset3 | |
| MA4 | base_length + offset4 | |
| MA_1 | base_length - offset1 | Slower MA |
| MA_2 | base_length - offset2 | |
| MA_3 | base_length - offset3 | |
| MA_4 | base_length - offset4 | |

The offset values depend on the **robustness** setting (see above).

## 🎯 Signal Legend

### Signal Types
- **Average_Signal**: Combined signal from all 6 MA types (EMA, HMA, WMA, DEMA, LSMA, KAMA)
- **EMA_Signal**: Buy/sell signals based on EMA crossovers
- **HMA_Signal**: Buy/sell signals based on HMA crossovers
- **WMA_Signal**: Buy/sell signals based on WMA crossovers
- **DEMA_Signal**: Buy/sell signals based on DEMA crossovers
- **LSMA_Signal**: Buy/sell signals based on LSMA crossovers
- **KAMA_Signal**: Buy/sell signals based on KAMA crossovers

### Signal Values
- **+1.0**: Strong Buy signal (↑ green triangle)
- **0.0**: Neutral (no action)
- **-1.0**: Strong Sell signal (↓ red triangle)
- Values between -1 and 1 indicate signal strength

## ⚙️ Default Configuration

| Parameter | Default Value | Range | Description |
|-----------|---------------|-------|-------------|
| Symbol | BTC/USDT | - | Trading pair |
| Timeframe | 15m | 1m-1d | Chart timeframe |
| Limit | 1500 | 100-5000 | Number of candles |
| MA Lengths | 28 | 1+ | Base length for all MAs |
| Robustness | Medium | Narrow/Medium/Wide | MA offset spread |
| Lambda | 0.02 | 0.0-1.0 | Signal adaptation rate |
| Decay | 0.03 | 0.0-1.0 | Exponential decay factor |
| Cutout | 0 | 0+ | Candles to exclude |

## 🔧 Technical Details

### MA Types & Formulas
- **EMA** (Exponential Moving Average): Weights recent prices more heavily
- **HMA** (Hull Moving Average): Uses SMA (not classic HMA) - smooths price data
- **WMA** (Weighted Moving Average): Linear weighting of prices
- **DEMA** (Double EMA): Reduces lag more than single EMA
- **LSMA** (Least Squares MA): Linear regression-based MA
- **KAMA** (Kaufman Adaptive MA): Adjusts based on market volatility (fast=2, slow=30)

### ATC Signal Logic
1. For each MA type, compute 9 MAs with different lengths
2. Detect crossovers between faster and slower MAs
3. Generate individual signals per MA type
4. Average all MA signals to create combined signal
5. Apply smoothing with lambda and decay parameters

## 💾 Data Source & Exchange

### Primary Exchange
- **Binance Futures** for data fetching
- Real-time market data via ccxt library

### Fallback Mechanism
- Automatic fallback to alternative exchanges if primary fails
- Uses `ExchangeManager` for resilient data fetching
- Supports multiple USDT-M futures pairs

### Available Symbols
- Top 50 USDT-M futures pairs by volume
- Listed via `/api/symbols` endpoint

## 🔌 API Endpoints

### Endpoints
- `GET /` - Health check with app info
- `GET /api/health` - API health check
- `GET /api/symbols` - List available trading symbols
- `GET /api/ohlcv` - Fetch OHLCV data
- `GET /api/atc-signals` - Compute ATC signals
- `GET /api/moving-averages` - Get all Moving Averages
- `GET /api/timeframes` - Get available timeframes

### API Request Examples

#### Fetch OHLCV Data
```bash
curl "http://localhost:8002/api/ohlcv?symbol=BTC/USDT&timeframe=15m&limit=1500"
```

#### Compute ATC Signals
```bash
curl "http://localhost:8002/api/atc-signals?symbol=BTC/USDT&timeframe=15m&limit=1500&ema_len=28&robustness=Medium"
```

#### Get Moving Averages
```bash
curl "http://localhost:8002/api/moving-averages?symbol=BTC/USDT&timeframe=15m&limit=1500"
```

#### List Available Symbols
```bash
curl "http://localhost:8002/api/symbols"
```

## 📦 Dependencies & Module Integration

### Parent Module Dependencies
The visualizer uses the following modules from the parent `modules/` directory:

- `modules/adaptive_trend/` - ATC algorithm implementation
  - `core/analyzer.py` - Main ATC analysis logic
  - `core/compute_moving_averages.py` - MA calculations
  - `core/compute_atc_signals.py` - Signal computation
  - `utils/config.py` - ATC configuration classes
  - `utils/diflen.py` - MA offset calculations

- `modules/common/` - Shared utilities
  - `core/data_fetcher.py` - OHLCV data fetching
  - `core/exchange_manager.py` - Exchange connection management
  - `indicators/momentum.py` - KAMA and other momentum indicators
  - `utils/logging.py` - Logging utilities

### Backend Requirements
```txt
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
pandas>=2.1.4
numpy>=1.25.0
```

### Frontend Dependencies
```json
{
  "dependencies": {
    "vue": "^3.3.4",
    "apexcharts": "^4.0.0",
    "vue3-apexcharts": "^1.10.0"
  }
}
```

## 🛠️ Technologies

### Backend
- **FastAPI** (v0.104+): Modern, fast web framework for building APIs
- **Uvicorn**: ASGI server for FastAPI
- **Pandas** (v2.1.4+): Data manipulation and analysis
- **NumPy** (v1.25.0+): Numerical computing

### Frontend
- **Vue.js 3** (v3.3.4+): Progressive JavaScript framework
- **ApexCharts** (v4.0.0): Modern charting library
- **vue3-apexcharts** (v1.10.0): Vue 3 wrapper for ApexCharts
- **Vite** (v7.3.1+): Next-gen frontend build tool

### Data & Utilities
- **ccxt**: Exchange integration library (via modules/)
- **pandas_ta**: Technical analysis indicators (via modules/)
- **Custom modules**: adaptive_trend, common (from parent directory)

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

## ⚡ Performance Optimization

### Recommended Settings
| Use Case | Timeframe | Limit | Robustness |
|----------|-----------|-------|------------|
| Scalp Trading | 1m-5m | 500-1000 | Narrow |
| Day Trading | 15m-1h | 1500-2000 | Medium |
| Swing Trading | 4h-1d | 2000-5000 | Wide |

### Performance Tips
- **Lower limits (500-1000)** for faster loading on slower connections
- **Wider robustness** for trend-following strategies
- **Narrow robustness** for quick signal detection
- **Avoid loading 5000 candles on 1m timeframe** - may be slow

### Browser Performance
- Use Chrome or Firefox for best ApexCharts performance
- Close other tabs when viewing large datasets
- Enable hardware acceleration in browser settings

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

## ❓ FAQ

### Q: Why are some MA lines not showing?
A: Use the SignalLegend panel (bottom of page) to toggle MA types on/off. EMA and HMA are enabled by default.

### Q: What do the different MA colors mean?
A: Colors are for visual distinction only:
- EMA: Green (#00E396)
- HMA: Yellow (#FEB019)
- WMA: Purple (#775DD0)
- DEMA: Blue (#008FFB)
- LSMA: Red (#FF4560)
- KAMA: Purple (#775DD0)

### Q: How are signals calculated?
A: Signals are generated by comparing faster MAs (MA1-MA4) against slower MAs (MA_1-MA_4). Crossovers generate buy/sell signals.

### Q: Why use 9 MAs instead of just 1?
A: The ATC algorithm uses multiple MA offsets to detect trends at different sensitivities, then averages signals for more reliable predictions.

### Q: Can I customize MA colors?
A: Yes, edit `frontend/src/utils/chartHelper.js` and modify the color hex codes in the `maColors` object.

### Q: Which MA type is best for trading?
A: There's no "best" MA - each has different characteristics:
- EMA: Responsive, good for trending markets
- HMA: Smooth, reduces lag
- WMA: Simple, widely used
- DEMA: Very responsive, can be noisy
- LSMA: Predictive, good for reversals
- KAMA: Adaptive, adjusts to volatility

## 🛠️ Extending the Visualizer

### Adding New MA Types
1. Add calculation logic in `modules/common/indicators/`
2. Update `modules/adaptive_trend/core/compute_moving_averages.py`
3. Add to frontend legend in `SignalLegend.vue`
4. Update chart colors in `chartHelper.js`

### Customizing Chart Appearance
- Edit `frontend/src/components/ChartView.vue` for layout
- Edit `frontend/src/utils/chartHelper.js` for ApexCharts configuration
- ApexCharts API: https://apexcharts.com/docs/options/

### Adding New API Endpoints
1. Add endpoint function in `backend/main.py`
2. Add corresponding service method in `backend/services/atc_service.py`
3. Access via frontend in `frontend/src/services/api.js`

## 🔗 Related Links

- [Main Web Documentation](../README.md)
- [Architecture Overview](../docs/ARCHITECTURE.md)
- [Adding New Apps Guide](../docs/ADDING_NEW_APP.md)
- [Migration Summary](../MIGRATION_SUMMARY.md)

## 🔬 Development & Testing

### Verify Module Imports
To verify that all parent modules are correctly imported:
```bash
cd web/apps/atc_visualizer/backend
python test_import.py
```

Expected output: `✅ SUCCESS: All modules imported!`

### Debug Path Issues
If you encounter import errors, check the path configuration:
```bash
cd web/apps/atc_visualizer/backend
python test_path.py
```

### Backend Development
```bash
# Start backend with auto-reload
cd web/apps/atc_visualizer/backend
uvicorn main:app --reload --port 8002

# Check API documentation
# Open http://localhost:8002/docs in browser
```

### Frontend Development
```bash
cd web/apps/atc_visualizer/frontend
npm run dev

# Preview production build
npm run build
npm run preview
```

### Linting & Formatting
```bash
# Backend (Python)
pip install black flake8
cd web/apps/atc_visualizer/backend
black .
flake8 .

# Frontend (JavaScript/ESLint - if configured)
cd web/apps/atc_visualizer/frontend
npm run lint  # if package.json has lint script
```

## 📝 Change Log

### v1.0.0
- Initial release with ATC visualization
- Support for 6 MA types (EMA, HMA, WMA, DEMA, LSMA, KAMA)
- 9 MAs per type with configurable robustness
- Real-time chart with ApexCharts
- Vue.js 3 frontend with dark mode
- FastAPI backend with auto-generated docs

