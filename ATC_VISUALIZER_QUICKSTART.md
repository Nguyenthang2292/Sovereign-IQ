# ATC Visualizer - Quick Start

One-command startup for backend and frontend servers.

## 🚀 Quick Start (Recommended)

### Windows
```bash
start_visualizer.bat
```

### Linux/Mac
```bash
./start_visualizer.sh
```

### Direct Python
```bash
python run_atc_visualizer.py
```

## 📋 Options

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

Use this if npm is installed but not detected by the script.

## 🌐 Access Points

After startup, access:

- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:5000
- **API Docs (Swagger)**: http://localhost:5000/docs
- **API Docs (ReDoc)**: http://localhost:5000/redoc

## ⏸️ Stop Servers

Press `Ctrl+C` in the terminal to stop both servers gracefully.

## 📁 What This Does

The startup script:

1. ✅ Checks if Python and Node.js are installed
2. 🔧 Installs/Updates backend dependencies (pip install)
3. 🔧 Installs/Updates frontend dependencies (npm install)
4. 🚀 Starts FastAPI backend on port 5000
5. 🎨 Starts Vue.js + Vite frontend on port 5173
6. 📊 Displays all access URLs
7. ⏸️ Stops both servers on Ctrl+C

## ⚠️ Troubleshooting

### Port Already in Use

If port 5000 or 5173 is already in use:

**Change Backend Port:**
Edit `web/atc_visualizer/backend/api.py`:
```python
uvicorn.run(app, host="0.0.0.0", port=8000)  # Change to 8000
```

**Change Frontend Port:**
Edit `web/atc_visualizer/frontend/vite.config.js`:
```javascript
server: {
  port: 3000,  // Change to 3000
  proxy: {
    '/api': {
      target: 'http://localhost:8000',  // Match backend port
      changeOrigin: true
    }
  }
}
```

### Backend Fails to Start

- Check Python version: `python --version` (needs 3.9+)
- Install dependencies manually:
  ```bash
  cd web/atc_visualizer/backend
  pip install -r requirements.txt
  python api.py
  ```

### Frontend Fails to Start

- Check Node.js version: `node --version` (needs 18+)
- Install dependencies manually:
  ```bash
  cd web/atc_visualizer/frontend
  npm install
  npm run dev
  ```

## 📚 Full Documentation

For detailed documentation, see: [web/atc_visualizer/README.md](web/atc_visualizer/README.md)
