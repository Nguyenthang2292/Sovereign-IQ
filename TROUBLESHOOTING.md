# Troubleshooting Guide - ATC Visualizer

## ❌ Issue: Frontend shows "This site can't be reached" or "Page not found"

### Check 1: Verify Frontend is Running
```bash
# Check if Vite is actually running
# Open a new terminal and run:
netstat -ano | findstr :5173
```

### Check 2: Manually Start Frontend
```bash
# Stop the current run_atc_visualizer.py (Ctrl+C)
# Then start frontend separately:
cd web/atc_visualizer/frontend
npm run dev
```

If you see:
- ✅ `➜  Local:   http://localhost:5173/` → Frontend is working
- ❌ `Error` → Check error messages

### Check 3: Browser Cache
- Try **Incognito/Private Mode**
- Clear browser cache (Ctrl+Shift+Delete)
- Try different browser (Chrome, Firefox, Edge)

## ❌ Issue: Backend returns {"detail":"Not Found"}

### Check 1: Correct URL
Wrong URL: `http://localhost:5000/`
Correct URL:
- `http://localhost:5000/` (root - shows API info)
- `http://localhost:5000/api/health` (health check)
- `http://localhost:5000/docs` (Swagger documentation)

### Check 2: Verify Backend is Running
```bash
# Check if FastAPI is running
curl http://localhost:5000/api/health
# or in browser: http://localhost:5000/api/health
```

### Check 3: Manually Start Backend
```bash
# Stop the current run_atc_visualizer.py (Ctrl+C)
# Then start backend separately:
cd web/atc_visualizer/backend
python api.py
```

Expected output:
```
Starting ATC Visualizer API on http://localhost:5000
API Documentation available at: http://localhost:5000/docs
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:5000
```

## ❌ Issue: Ports Already in Use

Error: `[Errno 10048] only one usage of each socket address`

### Solution 1: Automatic Cleanup (Recommended)
```bash
# Kill processes on ports 5000 and 5173 automatically
python kill_ports.py

# Or use batch file
kill_ports.bat
```

### Solution 2: Kill and Start
```bash
# Automatically kill ports and start servers
start_visualizer_clean.bat
```

### Solution 3: Manual Kill
```bash
# Check what's using port 5000
netstat -ano | findstr :5000

# Kill the process (replace <PID> with actual PID from above)
taskkill /F /PID <PID>

# Same for port 5173
netstat -ano | findstr :5173
taskkill /F /PID <PID>
```

### Check Ports
```bash
python check_ports.py
```

## ❌ Issue: ModuleNotFoundError

Error message: `ModuleNotFoundError: No module named 'modules'`

This should be fixed with the updated code. If it still happens:

### Solution 1: Run from Project Root
```bash
# Make sure you're in the project root
cd C:\Users\Admin\Desktop\i-ching\crypto-probability
python run_atc_visualizer.py
```

### Solution 2: Manual Start
```bash
# Backend
cd C:\Users\Admin\Desktop\i-ching\crypto-probability\web\atc_visualizer\backend
set PYTHONPATH=C:\Users\Admin\Desktop\i-ching\crypto-probability
python api.py
```

## 🧪 Test Individually

### Test Backend Only
```bash
test_backend.bat
```

Then check: http://localhost:5000/api/health

### Test Frontend Only
```bash
test_frontend.bat
```

Then check: http://localhost:5173

## 📋 Common Solutions

### 1. Restart Everything
```bash
# 1. Stop all processes (Ctrl+C in all terminals)
# 2. Run:
python check_ports.py

# 3. If ports are in use, kill them:
netstat -ano | findstr :5000
taskkill /F /PID <PID>

netstat -ano | findstr :5173
taskkill /F /PID <PID>

# 4. Start fresh:
python run_atc_visualizer.py
```

### 2. Clear Frontend Dependencies
```bash
cd web/atc_visualizer/frontend
rmdir /s /q node_modules
del package-lock.json
cd ../..
python run_atc_visualizer.py --install
```

### 3. Reinstall Node.js
Sometimes corrupted npm installation causes issues:
1. Uninstall Node.js
2. Download fresh from https://nodejs.org
3. Check "Add to PATH" during install
4. Restart computer
5. Try again

## 🔍 Get Logs

If everything fails, get the actual error logs:

```bash
# Backend logs
cd web/atc_visualizer/backend
python api.py 2>&1 > backend.log

# Frontend logs
cd web/atc_visualizer/frontend
npm run dev 2>&1 > frontend.log
```

Then check `backend.log` and `frontend.log` files.
