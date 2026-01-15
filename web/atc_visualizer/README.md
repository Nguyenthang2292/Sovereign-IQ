# ATC Visualizer

Web-based visualization tool for Adaptive Trend Classification (ATC) algorithm.

## Features

- Real-time OHLCV chart visualization with ApexCharts
- Display of all 6 Moving Average types (EMA, HMA, WMA, DEMA, LSMA, KAMA)
- 9 Moving Averages per type with different length offsets
- Signal visualization with arrow markers (↑ for buy, ↓ for sell)
- Multiple timeframe support (1m to 1d)
- Toggle visibility for each MA type and signal
- Dark mode optimized for trading analysis

## Project Structure

```
web/atc_visualizer/
├── backend/
│   ├── api.py               # FastAPI REST API
│   ├── atc_service.py       # ATC computation service
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.vue
│   │   ├── main.js
│   │   ├── components/
│   │   │   ├── ParameterPanel.vue
│   │   │   ├── SignalLegend.vue
│   │   │   └── ChartView.vue
│   │   ├── services/
│   │   │   └── api.js
│   │   └── utils/
│   │       └── chartHelper.js
│   ├── package.json
│   └── vite.config.js
```

## Installation

### Backend

1. Install Python dependencies:

```bash
cd backend
pip install -r requirements.txt
```

2. Start the API server:

```bash
python api.py
```

The API will be available at `http://localhost:5000`

### Frontend

1. Install Node.js dependencies:

```bash
cd frontend
npm install
```

2. Start the development server:

```bash
npm run dev
```

The application will be available at `http://localhost:5173`

## Usage

1. Enter a trading symbol (e.g., `BTC/USDT`, `ETH/USDT`)
2. Select a timeframe (1m, 5m, 15m, 1h, 4h, 1d)
3. Set the number of candles to display
4. Click "Load Data" to fetch and visualize

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

## API Endpoints

- `GET /api/health` - Health check
- `GET /api/symbols` - List available symbols
- `GET /api/ohlcv` - Fetch OHLCV data
- `GET /api/atc-signals` - Compute ATC signals
- `GET /api/moving-averages` - Get all MAs
- `GET /api/timeframes` - Get available timeframes

## Technologies

- **Backend**: FastAPI, Pandas, NumPy
- **Frontend**: Vue.js 3, ApexCharts, Vite
- **Data**: ccxt (exchange integration), pandas_ta (technical indicators)
- **API Server**: Uvicorn

## API Documentation

When the backend is running, access the interactive API documentation at:
- Swagger UI: http://localhost:5000/docs
- ReDoc: http://localhost:5000/redoc

## Notes

- Ensure backend API is running before starting frontend
- The visualizer uses modules from parent `modules/` directory
- First data load may take 10-30 seconds depending on network
- FastAPI provides automatic validation and type checking for all parameters
