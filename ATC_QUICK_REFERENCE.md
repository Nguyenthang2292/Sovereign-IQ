# ATC Visualizer - Quick Reference

## 🚀 Quick Start (Recommended)

```bash
start_visualizer_clean.bat
```

This will:
1. ✅ Kill processes on ports 5000, 5173
2. ✅ Start Backend (FastAPI)
3. ✅ Start Frontend (Vue.js)

## 📋 Alternative Commands

### Start with Dependency Installation
```bash
python run_atc_visualizer.py
```

### Skip npm Check
```bash
python run_atc_visualizer.py --skip-npm-check
```

### Install Dependencies Only
```bash
python run_atc_visualizer.py --install
```

### Kill Ports Only
```bash
python kill_ports.py
# or
kill_ports.bat
```

### Check Ports
```bash
python check_ports.py
```

## 🧪 Test Individually

### Backend Only
```bash
test_backend.bat
```
Then visit: http://localhost:5000/api/health

### Frontend Only
```bash
test_frontend.bat
```
Then visit: http://localhost:5173

### Test Imports
```bash
test_import.bat
```

## 🔧 Manual Start (If automated fails)

### Terminal 1 - Backend
```bash
cd web\atc_visualizer\backend
python api.py
```

### Terminal 2 - Frontend
```bash
cd web\atc_visualizer\frontend
npm run dev
```

## 🌐 Access Points

- **Frontend**: http://localhost:5173
- **Backend Root**: http://localhost:5000
- **Health Check**: http://localhost:5000/api/health
- **API Docs**: http://localhost:5000/docs
- **API Docs (ReDoc)**: http://localhost:5000/redoc

## ❌ Common Issues

### Port Already in Use (Errno 10048)

**Solution:**
```bash
start_visualizer_clean.bat
```

Or manually:
```bash
python kill_ports.py
```

### Frontend Shows "Page Not Found"

**Causes:**
1. Frontend not actually running
2. Wrong URL
3. Browser cache

**Solutions:**
1. Check terminal for `➜  Local: http://localhost:5173/`
2. Clear browser cache (Ctrl+Shift+Delete)
3. Try Incognito/Private mode

### Backend Returns {"detail":"Not Found"}

**Cause:** Accessing root `/` instead of API endpoints

**Correct URLs:**
- ✅ http://localhost:5000/api/health
- ✅ http://localhost:5000/docs
- ✅ http://localhost:5000/api/ohlcv?symbol=BTC/USDT

### ModuleNotFoundError

**Should be fixed!** If still happens:

```bash
# Run from project root
cd C:\Users\Admin\Desktop\i-ching\crypto-probability
start_visualizer_clean.bat
```

## 📁 Project Structure

```
crypto-probability/
├── modules/                          # Core trading modules
│   ├── adaptive_trend/              # ATC algorithm
│   └── common/                     # Shared utilities
├── web/
│   └── atc_visualizer/             # Visualizer app
│       ├── backend/                  # FastAPI
│       │   ├── api.py
│       │   ├── atc_service.py
│       │   └── requirements.txt
│       └── frontend/                 # Vue.js
│           ├── src/
│           │   ├── App.vue
│           │   ├── components/
│           │   └── services/
│           └── package.json
├── run_atc_visualizer.py            # Main entry point
├── kill_ports.py                    # Port cleanup
├── check_ports.py                  # Port checker
├── start_visualizer_clean.bat       # Recommended start
└── [other test/debug scripts]
```

## 💡 Tips

1. **Always kill ports before starting** - prevents "port in use" errors
2. **Use two separate terminals** - easier to see logs from both servers
3. **Check API docs first** - http://localhost:5000/docs shows all endpoints
4. **Clear browser cache** if frontend seems broken
5. **Restart IDE** after path changes - LSP errors may be stale

## 🎯 Key Files

| File | Purpose |
|------|----------|
| `start_visualizer_clean.bat` | ⭐ Main startup (kill ports + start) |
| `run_atc_visualizer.py` | Python entry point |
| `kill_ports.py` | Kill processes on ports 5000, 5173 |
| `check_ports.py` | Check which ports are in use |
| `web/atc_visualizer/backend/api.py` | FastAPI REST API |
| `web/atc_visualizer/frontend/src/App.vue` | Main Vue component |
| `TROUBLESHOOTING.md` | Detailed troubleshooting guide |
