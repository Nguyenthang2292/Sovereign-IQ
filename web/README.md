# Web Apps - Crypto Probability

This directory contains web applications for the Crypto Probability project.

## 📦 Applications

### 1. Gemini Chart Analyzer
**Location:** `apps/gemini_analyzer/`
**Port:** 8001 (backend), 5173 (frontend dev)

Web interface for analyzing cryptocurrency charts using Google Gemini AI.

**Features:**
- Single & multi-timeframe chart analysis
- Batch market scanning
- Real-time log streaming
- Bilingual support (EN/VI)

[Read more →](apps/gemini_analyzer/README.md)

### 2. ATC Visualizer
**Location:** `apps/atc_visualizer/`
**Port:** 8002 (backend), 5174 (frontend dev)

Standalone visualization tool for Adaptive Trend Classification algorithm.

**Features:**
- Real-time OHLCV charts
- 6 Moving Average types (EMA, HMA, WMA, DEMA, LSMA, KAMA)
- Signal visualization
- Interactive parameter controls

[Read more →](apps/atc_visualizer/README.md)

## 🔧 Shared Resources

### Shared Utilities (`shared/`)
Common code used across all applications:

- **`shared/utils/`** - Task manager, log manager, error handling
- **`shared/middleware/`** - CORS, authentication (future)
- **`shared/models/`** - Common Pydantic models
- **`shared/services/`** - Shared business logic (future)

### Scripts (`scripts/`)
Management and deployment scripts:

- `start_all.py` - Start all applications
- `start_app.py` - Start specific app
- `kill_ports.py` - Kill processes on ports
- `health_check.py` - Check all apps health

## 🚀 Quick Start

### Start All Apps
```bash
cd web
python scripts/start_all.py
```

### Start Individual App
```bash
# Gemini Analyzer
cd web/apps/gemini_analyzer/backend && python main.py

# ATC Visualizer
cd web/apps/atc_visualizer/backend && python main.py
```

## 📊 Port Allocation

| Application | Backend Port | Frontend Dev Port |
|-------------|--------------|-------------------|
| Gemini Analyzer | 8001 | 5173 |
| ATC Visualizer | 8002 | 5174 |
| API Gateway (future) | 8000 | - |

## 🏗️ Architecture

```
web/
├── shared/                    # Shared utilities
│   ├── utils/
│   ├── middleware/
│   ├── models/
│   └── services/
├── apps/                      # Applications
│   ├── gemini_analyzer/
│   │   ├── backend/
│   │   └── frontend/
│   └── atc_visualizer/
│       ├── backend/
│       └── frontend/
├── gateway/                   # API Gateway (future)
├── scripts/                   # Management scripts
├── docker/                    # Docker configs
└── docs/                      # Documentation
```

## 🔮 Future Applications

### Portfolio Dashboard (Planned)
**Port:** 8003 (backend), 5175 (frontend dev)

Real-time portfolio management and risk analysis.

### Pairs Trading Monitor (Planned)
**Port:** 8004 (backend), 5176 (frontend dev)

Monitor and analyze pairs trading opportunities.

## 🛠️ Development

### Adding New App

1. Create app structure:
```bash
mkdir -p web/apps/new_app/backend web/apps/new_app/frontend
```

2. Copy template files from existing app

3. Update port configuration in `config.py`

4. Add app to `scripts/start_all.py`

See `docs/ADDING_NEW_APP.md` for detailed guide.

### Testing

```bash
# Test individual app
cd apps/gemini_analyzer/frontend
npm test

# Test all apps
python scripts/test_all.py
```

## 📚 Documentation

- [Architecture Overview](docs/ARCHITECTURE.md)
- [Deployment Guide](docs/DEPLOYMENT.md)
- [Adding New App](docs/ADDING_NEW_APP.md)
- [API Gateway](docs/API_GATEWAY.md) (future)

## 🐳 Docker

```bash
# Start all apps with Docker Compose
docker-compose -f docker/docker-compose.yml up

# Start specific app
docker-compose -f docker/docker-compose.yml up gemini-analyzer
```

## ⚠️ Notes

- Each app is self-contained and can run independently
- Shared utilities are imported from `web/shared/`
- Frontend dev servers proxy API requests to backend
- Production builds serve static files from backend

## 📞 Support

For issues related to specific apps, see their respective README files.
