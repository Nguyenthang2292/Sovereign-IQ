# Web Apps - Sovereign-IQ

This directory contains web applications for the Sovereign-IQ project.

## 📋 Migration Summary

**Status:** ✅ Migration Complete (2026-01-16)

The `web/` folder has been reorganized for better scalability and maintainability. Each application is now a self-contained module under `apps/`, with shared utilities separated into `shared/`.

**Key Changes:**
- Old structure (`web/app.py`, `web/api/`, `web/atc_visualizer/`) → New structure (`web/apps/*/`)
- All apps moved to `web/apps/` with independent backends/frontend
- Shared utilities extracted to `web/shared/`
- Management scripts created in `web/scripts/`
- Port allocations standardized

**Benefits:**
- Modular architecture - each app is independent
- Easy to add new apps without affecting existing ones
- Ready for microservices deployment
- Clear organization for new developers
- Code reusability through shared utilities

## 📁 Folder Structure

```
web/
├── shared/                     # Shared utilities
│   ├── utils/                 # Task manager, log manager, etc.
│   ├── middleware/            # CORS, auth (future)
│   ├── models/                # Pydantic models
│   └── services/              # Shared services (future)
│
├── apps/                       # All applications
│   ├── gemini_analyzer/       # Port 8001 (backend), 5173 (frontend)
│   │   ├── backend/
│   │   │   ├── main.py
│   │   │   ├── config.py
│   │   │   └── api/
│   │   └── frontend/
│   │       └── dist/
│   │
│   └── atc_visualizer/        # Port 8002 (backend), 5174 (frontend)
│       ├── backend/
│       │   ├── main.py
│       │   ├── config.py
│       │   └── services/
│       └── frontend/
│           └── dist/
│
├── scripts/                    # Management scripts
│   ├── start_all.py           # Start all apps
│   ├── start_app.py           # Start specific app
│   ├── kill_ports.py          # Kill processes on ports
│   └── health_check.py        # Check all apps health
│
├── gateway/                    # API Gateway (future)
├── docker/                     # Docker configs (future)
└── docs/                       # Documentation
```

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

### Start Specific App
```bash
# Using management script (recommended)
python scripts/start_app.py gemini_analyzer
python scripts/start_app.py atc_visualizer

# Backend only
python scripts/start_app.py gemini_analyzer --backend-only

# Frontend only
python scripts/start_app.py gemini_analyzer --frontend-only
```

### Manual Start (Development)
```bash
# Gemini Analyzer
cd web/apps/gemini_analyzer/backend && python main.py
cd web/apps/gemini_analyzer/frontend && npm run dev

# ATC Visualizer
cd web/apps/atc_visualizer/backend && python main.py
cd web/apps/atc_visualizer/frontend && npm run dev
```

## 📊 Port Allocation

| Application | Backend | Frontend Dev | Access Points |
|-------------|---------|--------------|---------------|
| Gemini Analyzer | 8001 | 5173 | http://localhost:5173 |
| ATC Visualizer | 8002 | 5174 | http://localhost:5174 |
| API Gateway (future) | 8000 | - | http://localhost:8000 |

## 🌐 Access Points

### Gemini Chart Analyzer
- **Frontend:** http://localhost:5173
- **Backend:** http://localhost:8001
- **API Docs:** http://localhost:8001/docs
- **Health Check:** http://localhost:8001/api/health

### ATC Visualizer
- **Frontend:** http://localhost:5174
- **Backend:** http://localhost:8002
- **API Docs:** http://localhost:8002/docs
- **Health Check:** http://localhost:8002/api/health

## 🏗️ Architecture

The new architecture follows a modular microservices pattern:

```
┌─────────────────────────────────────────────────────┐
│                     API Gateway                       │
│                  (future - port 8000)                │
└────────────────────┬────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
        ▼                         ▼
┌───────────────────┐    ┌───────────────────┐
│  Gemini Analyzer  │    │  ATC Visualizer   │
│  (port 8001)      │    │  (port 8002)      │
│  FastAPI + Vue.js │    │  FastAPI + Vue.js │
└───────────────────┘    └───────────────────┘
        │                         │
        └────────────┬────────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
        ▼                         ▼
┌───────────────────┐    ┌───────────────────┐
│   Shared Utils    │    │   Modules/        │
│   (web/shared/)   │    │   (parent/)       │
│   • task_manager  │    │   • adaptive_trend│
│   • log_manager   │    │   • common        │
│   • cors          │    │   • indicators    │
└───────────────────┘    └───────────────────┘
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

### Core Documentation
- [API Reference](docs/API_REFERENCE.md) - Complete API documentation for all applications
- [Architecture Overview](docs/ARCHITECTURE.md) - Detailed architecture documentation
- [Adding New App](docs/ADDING_NEW_APP.md) - Guide for creating new applications
- [Deployment Guide](docs/DEPLOYMENT.md) - Production deployment guide (future)
- [API Gateway](docs/API_GATEWAY.md) - API Gateway configuration (future)

### App-Specific Documentation
- [Gemini Analyzer README](apps/gemini_analyzer/README.md) - Chart analysis & batch scanning
- [ATC Visualizer README](apps/atc_visualizer/README.md) - ATC algorithm visualization

### App-Specific Documentation
- [Gemini Analyzer README](apps/gemini_analyzer/README.md)
- [ATC Visualizer README](apps/atc_visualizer/README.md)
