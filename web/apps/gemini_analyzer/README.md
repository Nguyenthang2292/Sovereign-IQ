# Gemini Chart Analyzer

Web application for analyzing cryptocurrency charts using Google Gemini AI.

## 🚀 Quick Start

### Development

**Terminal 1 - Backend:**
```bash
cd web/apps/gemini_analyzer/backend
python main.py
```

**Terminal 2 - Frontend:**
```bash
cd web/apps/gemini_analyzer/frontend
npm run dev
```

### Production Build

```bash
cd web/apps/gemini_analyzer/frontend
npm run build
cd ../backend
python main.py
```

## 🌐 Access Points

- **Frontend**: http://localhost:5173 (dev) 
- **Backend**: http://localhost:8001
- **API Docs**: http://localhost:8001/docs

## 📁 Structure

```
gemini_analyzer/
├── backend/
│   ├── main.py              # FastAPI entry point
│   ├── config.py            # App configuration
│   ├── api/                 # API routes
│   │   ├── chart_analyzer.py
│   │   ├── batch_scanner.py
│   │   └── logs.py
│   ├── services/            # Business logic (future)
│   └── models/              # Pydantic models (future)
└── frontend/
    ├── src/
    │   ├── App.vue
    │   ├── components/
    │   ├── services/
    │   └── router/
    ├── vite.config.js
    └── package.json
```

## ⚙️ Configuration

Edit `backend/config.py` to change:
- Port numbers
- CORS origins
- API settings
- Default parameters

## 🧪 Testing

```bash
cd frontend
npm test
```

## 📚 Documentation

See main project README for full documentation.
