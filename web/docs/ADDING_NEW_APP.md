# Adding a New Web Application

This guide explains how to add a new web application to the `web/` folder following the established architecture.

## Prerequisites

- Python 3.9+
- Node.js 18+
- Familiarity with FastAPI and Vue 3

## Step-by-Step Guide

### 1. Create App Directory Structure

```bash
cd web/apps
mkdir new_app/{backend,frontend}
cd new_app/backend
touch __init__.py
cd ../frontend
```

### 2. Backend Setup

#### 2.1 Create `config.py`

```python
"""
Configuration for New App.
"""

from pathlib import Path
from typing import List

# Get paths
BASE_DIR = Path(__file__).parent
FRONTEND_DIR = BASE_DIR.parent / "frontend"

# Go up from: web/apps/new_app/backend -> project root
PROJECT_ROOT = BASE_DIR.parent.parent.parent

# App settings
APP_TITLE = "New App API"
APP_DESCRIPTION = "REST API for New App"
APP_VERSION = "1.0.0"

# Port configuration
BACKEND_PORT = 8003  # Next available port
FRONTEND_DEV_PORT = 5175  # Next available port

# CORS origins
CORS_ORIGINS = [
    "http://localhost:5175",
    "http://localhost:8003",
    "http://127.0.0.1:5175",
    "http://127.0.0.1:8003",
]

# API settings
API_PREFIX = "/api"
```

#### 2.2 Create `main.py`

```python
"""
FastAPI server for New App.
"""

from fastapi import FastAPI
from pathlib import Path
import sys

# Setup path for imports
current_file_path = Path(__file__).resolve()
backend_dir = current_file_path.parent
project_root = backend_dir.parent.parent.parent
sys.path.insert(0, str(project_root))

from .config import (
    APP_TITLE,
    APP_DESCRIPTION,
    APP_VERSION,
    BACKEND_PORT,
    CORS_ORIGINS,
    API_PREFIX,
)
from web.shared.middleware.cors import setup_cors

# Initialize FastAPI app
app = FastAPI(
    title=APP_TITLE,
    description=APP_DESCRIPTION,
    version=APP_VERSION,
)

# Setup CORS
setup_cors(app, allowed_origins=CORS_ORIGINS)

# Import and mount routes
# from .api import router
# app.include_router(router, prefix=API_PREFIX, tags=["New App"])

@app.get("/")
async def root():
    return {
        "message": APP_TITLE,
        "status": "running",
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": APP_TITLE}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=BACKEND_PORT)
```

#### 2.3 Create API Routes

```bash
cd web/apps/new_app/backend
mkdir api
touch api/__init__.py
touch api/routes.py
```

```python
# api/routes.py
from fastapi import APIRouter

router = APIRouter()

@router.get("/items")
async def list_items():
    return {"items": []}

@router.get("/items/{item_id}")
async def get_item(item_id: int):
    return {"item_id": item_id}
```

Import in `main.py`:

```python
from .api.routes import router
app.include_router(router, prefix=API_PREFIX, tags=["Items"])
```

### 3. Frontend Setup

#### 3.1 Initialize Vue Project

```bash
cd web/apps/new_app/frontend
npm create vite@latest . -- --template vue
npm install
```

Or manually create:

```bash
touch package.json index.html vite.config.js
mkdir -p src/components src/services src/router src/assets
```

#### 3.2 Create `package.json`

```json
{
  "name": "new-app-frontend",
  "version": "1.0.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview"
  },
  "dependencies": {
    "vue": "^3.3.0",
    "vue-router": "^4.2.0",
    "axios": "^1.6.0"
  },
  "devDependencies": {
    "@vitejs/plugin-vue": "^4.3.0",
    "vite": "^7.3.0"
  }
}
```

#### 3.3 Create `vite.config.js`

```javascript
import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  plugins: [vue()],
  server: {
    port: 5175,
    proxy: {
      '/api': {
        target: 'http://localhost:8003',
        changeOrigin: true
      }
    }
  }
})
```

#### 3.4 Create `index.html`

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>New App</title>
</head>
<body>
  <div id="app"></div>
  <script type="module" src="/src/main.js"></script>
</body>
</html>
```

#### 3.5 Create `src/main.js`

```javascript
import { createApp } from 'vue'
import { createRouter, createWebHistory } from 'vue-router'
import App from './App.vue'

const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/', component: App }
  ]
})

createApp(App).use(router).mount('#app')
```

#### 3.6 Create `src/App.vue`

```vue
<template>
  <div>
    <h1>New App</h1>
    <div v-if="items.length">
      <div v-for="item in items" :key="item.id">
        {{ item }}
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import axios from 'axios'

const items = ref([])

onMounted(async () => {
  const response = await axios.get('/api/items')
  items.value = response.data
})
</script>

<style scoped>
h1 {
  color: #333;
}
</style>
```

### 4. Update Management Scripts

#### 4.1 Update `scripts/start_app.py`

Add app configuration:

```python
APPS = {
    # ... existing apps
    "new_app": {
        "name": "New App",
        "backend_dir": APPS_DIR / "new_app" / "backend",
        "backend_port": 8003,
        "frontend_dir": APPS_DIR / "new_app" / "frontend",
        "frontend_port": 5175,
    },
}
```

#### 4.2 Update `scripts/start_all.py`

Same as above, add to `APPS` dict.

#### 4.3 Update `scripts/health_check.py`

```python
APPS = {
    # ... existing apps
    "new_app": {
        "name": "New App",
        "backend_url": "http://localhost:8003/health",
        "frontend_url": "http://localhost:5175",
        "api_docs_url": "http://localhost:8003/docs",
    },
}
```

### 5. Create README

Create `web/apps/new_app/README.md`:

```markdown
# New App

Description of what this app does.

## 🚀 Quick Start

### Development

**Terminal 1 - Backend:**
```bash
cd web/apps/new_app/backend
python main.py
```

**Terminal 2 - Frontend:**
```bash
cd web/apps/new_app/frontend
npm install
npm run dev
```

## 🌐 Access Points

- Frontend: http://localhost:5175
- Backend: http://localhost:8003
- API Docs: http://localhost:8003/docs
```

### 6. Update Main Documentation

Update `web/README.md` to include your new app:

```markdown
### 3. New App
**Location:** `apps/new_app/`
**Port:** 8003 (backend), 5175 (frontend dev)

Description of new app.

[Read more →](apps/new_app/README.md)
```

Also update port allocation table and architecture diagram.

### 7. Create API Documentation

Add detailed API documentation to your app's README:

```markdown
## API Endpoints

### List Items

`GET /api/items`

Returns a list of items.

Response:
```json
{
  "items": [...]
}
```

### Get Item

`GET /api/items/{id}`

Returns a specific item.

Parameters:
- `id` (path): Item identifier

Response:
```json
{
  "item_id": 123
}
```
```

## Best Practices

### Backend
- ✅ Use Pydantic models for request/response validation
- ✅ Implement error handling with HTTPException
- ✅ Use async/await for I/O operations
- ✅ Add docstrings to all functions
- ✅ Follow RESTful conventions
- ✅ Use shared utilities where appropriate
- ✅ Add health check endpoint

### Frontend
- ✅ Use Composition API (`<script setup>`)
- ✅ Implement proper error handling
- ✅ Add loading states
- ✅ Use semantic HTML
- ✅ Follow Vue style guide
- ✅ Lazy load routes with `import()`
- ✅ Use environment variables for API URLs

### Testing
- ✅ Write unit tests for services
- ✅ Write component tests for Vue components
- ✅ Test API endpoints
- ✅ Add integration tests
- ✅ Run tests in CI/CD

### Documentation
- ✅ Document all API endpoints
- ✅ Include code examples
- ✅ Add diagrams for complex flows
- ✅ Keep README up to date
- ✅ Document environment variables

## Port Allocation Guidelines

| Port Range | Use Case |
|------------|----------|
| 8000 | API Gateway (future) |
| 8001-8099 | Backend services |
| 5173-5199 | Frontend dev servers |

When adding a new app, use the next available port:
1. Backend: Check `scripts/start_app.py` for highest backend port
2. Frontend: Check `scripts/start_app.py` for highest frontend port

## Checklist

Before considering your new app complete:

- [ ] Directory structure created
- [ ] Backend server runs successfully
- [ ] API routes work correctly
- [ ] Frontend builds successfully
- [ ] Frontend dev server runs
- [ ] API proxy configured correctly
- [ ] Added to management scripts
- [ ] README created
- [ ] Main documentation updated
- [ ] Port allocation updated
- [ ] Health check added
- [ ] Code follows existing patterns
- [ ] Tests written
- [ ] Linting passes

## Common Issues

### Import Errors

```python
# If you get "module not found"
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
```

### CORS Errors

Check `vite.config.js` proxy target matches backend port.

### Port Already in Use

```bash
# Kill the port
python web/scripts/kill_ports.py 8003 5175
```

### Build Failures

Ensure all dependencies are installed:
```bash
cd web/apps/new_app/frontend
npm install
```

## Example: Complete Minimal App

See `apps/gemini_analyzer/` and `apps/atc_visualizer/` for complete examples.

## Support

For questions or issues, refer to:
- Main project README
- Architecture documentation
- Similar existing apps
