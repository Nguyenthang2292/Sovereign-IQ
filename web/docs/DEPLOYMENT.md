# Deployment Guide

This guide covers deployment options for web applications.

## Table of Contents

- [Development Deployment](#development-deployment)
- [Production Deployment](#production-deployment)
- [Docker Deployment](#docker-deployment)
- [Cloud Deployment](#cloud-deployment)
- [Environment Configuration](#environment-configuration)
- [Security Best Practices](#security-best-practices)
- [Monitoring & Logging](#monitoring--logging)

---

## Development Deployment

### Prerequisites

- Python 3.9+
- Node.js 18+
- Git

### Local Development Setup

```bash
# Clone repository
git clone <repository-url>
cd crypto-probability

# Install Python dependencies
pip install -r requirements.txt

# Install each app's frontend
cd web/apps/gemini_analyzer/frontend && npm install
cd ../../atc_visualizer/frontend && npm install

# Start all apps
cd ../../..
python web/scripts/start_all.py
```

### Using Management Scripts

**Start All Apps:**
```bash
cd web
python scripts/start_all.py
```

**Start Specific App:**
```bash
python scripts/start_app.py gemini_analyzer
python scripts/start_app.py atc_visualizer
```

**Start with Options:**
```bash
# Backend only
python scripts/start_app.py gemini_analyzer --backend-only

# Frontend only
python scripts/start_app.py gemini_analyzer --frontend-only

# Skip dependency check
python scripts/start_app.py gemini_analyzer --no-install
```

### Access Points

| App | Frontend | Backend | API Docs |
|-----|----------|---------|-----------|
| Gemini Analyzer | http://localhost:5173 | http://localhost:8001 | http://localhost:8001/docs |
| ATC Visualizer | http://localhost:5174 | http://localhost:8002 | http://localhost:8002/docs |

---

## Production Deployment

### Option 1: Nginx + Uvicorn

#### Backend Setup

```bash
# Create virtual environment
python -m venv /var/www/crypto-web/venv
source /var/www/crypto-web/venv/bin/activate

# Install dependencies
pip install fastapi uvicorn[standard] gunicorn

# Create systemd service for Gemini Analyzer
sudo nano /etc/systemd/system/gemini-analyzer.service
```

**Gemini Analyzer Service (`/etc/systemd/system/gemini-analyzer.service`):**
```ini
[Unit]
Description=Gemini Analyzer API
After=network.target

[Service]
Type=notify
User=www-data
WorkingDirectory=/var/www/crypto-web/apps/gemini_analyzer/backend
Environment="PATH=/var/www/crypto-web/venv/bin"
ExecStart=/var/www/crypto-web/venv/bin/gunicorn \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 127.0.0.1:8001 \
    main:app

[Install]
WantedBy=multi-user.target
```

**ATC Visualizer Service (`/etc/systemd/system/atc-visualizer.service`):**
```ini
[Unit]
Description=ATC Visualizer API
After=network.target

[Service]
Type=notify
User=www-data
WorkingDirectory=/var/www/crypto-web/apps/atc_visualizer/backend
Environment="PATH=/var/www/crypto-web/venv/bin"
ExecStart=/var/www/crypto-web/venv/bin/gunicorn \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 127.0.0.1:8002 \
    main:app

[Install]
WantedBy=multi-user.target
```

**Start Services:**
```bash
sudo systemctl daemon-reload
sudo systemctl start gemini-analyzer
sudo systemctl start atc-visualizer
sudo systemctl enable gemini-analyzer
sudo systemctl enable atc-visualizer
```

#### Frontend Setup

```bash
# Build production bundles
cd /var/www/crypto-web/apps/gemini_analyzer/frontend
npm run build

cd /var/www/crypto-web/apps/atc_visualizer/frontend
npm run build
```

#### Nginx Configuration

```bash
sudo nano /etc/nginx/sites-available/crypto-web
```

**Nginx Config (`/etc/nginx/sites-available/crypto-web`):**
```nginx
# Upstream for backend APIs
upstream gemini_analyzer {
    server 127.0.0.1:8001;
    keepalive 64;
}

upstream atc_visualizer {
    server 127.0.0.1:8002;
    keepalive 64;
}

# Gemini Analyzer - Frontend
server {
    listen 80;
    server_name gemini.yourdomain.com;

    root /var/www/crypto-web/apps/gemini_analyzer/frontend/dist;
    index index.html;

    # Frontend
    location / {
        try_files $uri $uri/ /index.html;
    }

    # Backend API
    location /api/ {
        proxy_pass http://gemini_analyzer;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
    }

    # Static files
    location /static/ {
        alias /var/www/crypto-web/apps/gemini_analyzer/backend/static/;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
}

# ATC Visualizer - Frontend
server {
    listen 80;
    server_name atc.yourdomain.com;

    root /var/www/crypto-web/apps/atc_visualizer/frontend/dist;
    index index.html;

    # Frontend
    location / {
        try_files $uri $uri/ /index.html;
    }

    # Backend API
    location /api/ {
        proxy_pass http://atc_visualizer;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
    }
}
```

**Enable Site:**
```bash
sudo ln -s /etc/nginx/sites-available/crypto-web /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

---

### Option 2: Single Server with Subdomains

Use subdomains for each app:
- `gemini.yourdomain.com` → Gemini Analyzer
- `atc.yourdomain.com` → ATC Visualizer

Follow the Nginx configuration above, but use separate server blocks for each subdomain.

---

### Option 3: Single Server with Path Routing

Use path routing for each app:
- `yourdomain.com/gemini/` → Gemini Analyzer
- `yourdomain.com/atc/` → ATC Visualizer

```nginx
server {
    listen 80;
    server_name yourdomain.com;

    # Gemini Analyzer
    location /gemini/ {
        root /var/www/crypto-web/apps/gemini_analyzer/frontend/dist;
        try_files $uri $uri/ /gemini/index.html;
    }

    location /gemini/api/ {
        rewrite ^/gemini/api/(.*)$ /api/$1 break;
        proxy_pass http://gemini_analyzer;
        # ... proxy settings
    }

    # ATC Visualizer
    location /atc/ {
        root /var/www/crypto-web/apps/atc_visualizer/frontend/dist;
        try_files $uri $uri/ /atc/index.html;
    }

    location /atc/api/ {
        rewrite ^/atc/api/(.*)$ /api/$1 break;
        proxy_pass http://atc_visualizer;
        # ... proxy settings
    }
}
```

Update Vue Router base path in each app:
```javascript
// gemini_analyzer/frontend/src/router/index.js
const router = createRouter({
  history: createWebHistory('/gemini/'),
  routes: [...]
})
```

---

## Docker Deployment

### Docker Compose Setup

Create `web/docker/docker-compose.yml`:

```yaml
version: '3.8'

services:
  # Gemini Analyzer - Backend
  gemini-backend:
    build:
      context: ../../
      dockerfile: web/docker/Dockerfile.gemini-backend
    container_name: gemini-backend
    restart: unless-stopped
    ports:
      - "8001:8001"
    volumes:
      - ./../apps/gemini_analyzer/backend:/app/backend
      - ./../shared:/app/shared
      - ./../../modules:/app/modules
    environment:
      - ENV=production
      - PYTHONUNBUFFERED=1

  # Gemini Analyzer - Frontend
  gemini-frontend:
    build:
      context: ../../
      dockerfile: web/docker/Dockerfile.gemini-frontend
    container_name: gemini-frontend
    restart: unless-stopped
    ports:
      - "5173:80"
    depends_on:
      - gemini-backend

  # ATC Visualizer - Backend
  atc-backend:
    build:
      context: ../../
      dockerfile: web/docker/Dockerfile.atc-backend
    container_name: atc-backend
    restart: unless-stopped
    ports:
      - "8002:8002"
    volumes:
      - ./../apps/atc_visualizer/backend:/app/backend
      - ./../shared:/app/shared
      - ./../../modules:/app/modules
    environment:
      - ENV=production
      - PYTHONUNBUFFERED=1

  # ATC Visualizer - Frontend
  atc-frontend:
    build:
      context: ../../
      dockerfile: web/docker/Dockerfile.atc-frontend
    container_name: atc-frontend
    restart: unless-stopped
    ports:
      - "5174:80"
    depends_on:
      - atc-backend

  # Nginx Reverse Proxy
  nginx:
    image: nginx:alpine
    container_name: nginx-proxy
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - gemini-backend
      - atc-backend
```

### Dockerfile Examples

**Gemini Backend (`web/docker/Dockerfile.gemini-backend`):**
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY web/apps/gemini_analyzer/backend ./backend
COPY web/shared ./shared
COPY modules ./modules

# Expose port
EXPOSE 8001

# Run application
CMD ["gunicorn", "--workers", "4", "--bind", "0.0.0.0:8001", "backend.main:app"]
```

**Gemini Frontend (`web/docker/Dockerfile.gemini-frontend`):**
```dockerfile
FROM node:18-alpine as builder

WORKDIR /app

# Install dependencies
COPY web/apps/gemini_analyzer/frontend/package*.json ./
RUN npm ci

# Build application
COPY web/apps/gemini_analyzer/frontend/ .
RUN npm run build

# Production stage
FROM nginx:alpine

# Copy built files
COPY --from=builder /app/dist /usr/share/nginx/html

# Expose port
EXPOSE 80

# Run nginx
CMD ["nginx", "-g", "daemon off;"]
```

**ATC Backend (`web/docker/Dockerfile.atc-backend`):**
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY web/apps/atc_visualizer/backend ./backend
COPY web/shared ./shared
COPY modules ./modules

EXPOSE 8002

CMD ["gunicorn", "--workers", "4", "--bind", "0.0.0.0:8002", "backend.main:app"]
```

**ATC Frontend (`web/docker/Dockerfile.atc-frontend`):**
```dockerfile
FROM node:18-alpine as builder

WORKDIR /app

COPY web/apps/atc_visualizer/frontend/package*.json ./
RUN npm ci

COPY web/apps/atc_visualizer/frontend/ .
RUN npm run build

FROM nginx:alpine

COPY --from=builder /app/dist /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
```

### Deploy with Docker

```bash
# Build and start all services
cd web/docker
docker-compose up -d

# Check logs
docker-compose logs -f

# Stop services
docker-compose down

# Rebuild services
docker-compose up -d --build

# Scale services
docker-compose up -d --scale gemini-backend=3
```

---

## Cloud Deployment

### AWS Deployment

#### Using AWS Elastic Beanstalk

```bash
# Install EB CLI
pip install awsebcli

# Initialize application
cd web/apps/gemini_analyzer/backend
eb init -p python-3.11

# Create environment
eb create production

# Deploy
eb deploy
```

#### Using AWS ECS + ALB

1. **Create ECR Repository:**
```bash
aws ecr create-repository --repository-name gemini-analyzer
aws ecr create-repository --repository-name atc-visualizer
```

2. **Build and Push Images:**
```bash
# Login to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com

# Build and tag
docker build -t gemini-analyzer -f web/docker/Dockerfile.gemini-backend .
docker tag gemini-analyzer:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/gemini-analyzer:latest

# Push
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/gemini-analyzer:latest
```

3. **Create ECS Task Definition**
4. **Create ECS Service**
5. **Configure Application Load Balancer**

---

### Google Cloud Deployment

#### Using Cloud Run

```bash
# Build and deploy Gemini Analyzer
gcloud builds submit --tag gcr.io/PROJECT-ID/gemini-analyzer

# Deploy
gcloud run deploy gemini-analyzer \
  --image gcr.io/PROJECT-ID/gemini-analyzer \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

---

### Azure Deployment

#### Using App Service

```bash
# Create resource group
az group create --name crypto-web --location eastus

# Create app service plan
az appservice plan create \
  --name crypto-web-plan \
  --resource-group crypto-web \
  --sku B1

# Create web app
az webapp create \
  --name gemini-analyzer \
  --resource-group crypto-web \
  --plan crypto-web-plan

# Deploy
az webapp up --name gemini-analyzer --resource-group crypto-web
```

---

## Environment Configuration

### Environment Variables

Create `.env` file for production:

```bash
# App Environment
ENV=production
DEBUG=false

# API Keys
GEMINI_API_KEY=your_gemini_api_key_here

# Database (future)
DATABASE_URL=postgresql://user:password@localhost:5432/dbname

# Redis (future)
REDIS_URL=redis://localhost:6379

# Logging
LOG_LEVEL=INFO
LOG_FILE=/var/log/crypto-web/app.log
```

### Config File Updates

Update `backend/config.py` for production:

```python
# Gemini Analyzer
CORS_ORIGINS = [
    "https://gemini.yourdomain.com",
    "https://yourdomain.com",
]

# ATC Visualizer
CORS_ORIGINS = [
    "https://atc.yourdomain.com",
    "https://yourdomain.com",
]
```

---

## Security Best Practices

### 1. Enable HTTPS

```bash
# Install Certbot
sudo apt install certbot python3-certbot-nginx

# Get SSL certificate
sudo certbot --nginx -d gemini.yourdomain.com -d atc.yourdomain.com

# Auto-renewal
sudo certbot renew --dry-run
```

### 2. Configure Firewall

```bash
# Allow only necessary ports
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp    # HTTP
sudo ufw allow 443/tcp   # HTTPS
sudo ufw deny 8001/tcp   # Block direct backend access
sudo ufw deny 8002/tcp
sudo ufw enable
```

### 3. Set File Permissions

```bash
# Set proper ownership
sudo chown -R www-data:www-data /var/www/crypto-web

# Set permissions
sudo find /var/www/crypto-web -type d -exec chmod 755 {} \;
sudo find /var/www/crypto-web -type f -exec chmod 644 {} \;
```

### 4. Secure API Keys

```bash
# Never commit .env files
echo ".env" >> .gitignore

# Use secrets manager in production
# AWS: AWS Secrets Manager
# GCP: Secret Manager
# Azure: Key Vault
```

### 5. Rate Limiting

Add rate limiting in Nginx:

```nginx
# Rate limiting
limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;

location /api/ {
    limit_req zone=api_limit burst=20 nodelay;
    proxy_pass http://gemini_analyzer;
}
```

---

## Monitoring & Logging

### Application Logs

```bash
# View systemd logs
sudo journalctl -u gemini-analyzer -f
sudo journalctl -u atc-visualizer -f

# View Nginx logs
sudo tail -f /var/log/nginx/access.log
sudo tail -f /var/log/nginx/error.log
```

### Health Checks

```bash
# Gemini Analyzer
curl http://localhost:8001/health

# ATC Visualizer
curl http://localhost:8002/api/health
```

### Resource Monitoring

```bash
# Check disk usage
df -h

# Check memory usage
free -h

# Check CPU usage
top

# Check running processes
ps aux | grep gunicorn
```

---

## Performance Optimization

### Gunicorn Worker Settings

```bash
# Calculate workers based on CPU cores
# workers = (2 * CPU) + 1
ExecStart=/var/www/crypto-web/venv/bin/gunicorn \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --threads 2 \
    --worker-connections 1000 \
    --timeout 120 \
    --keep-alive 5 \
    --bind 127.0.0.1:8001 \
    main:app
```

### Nginx Caching

```nginx
# Enable caching
proxy_cache_path /var/cache/nginx levels=1:2 keys_zone=api_cache:10m max_size=1g inactive=60m;

location /api/ {
    proxy_cache api_cache;
    proxy_cache_valid 200 10m;
    proxy_cache_valid 404 1m;
    proxy_cache_use_stale error timeout invalid_header updating;
    proxy_cache_background_update on;
    proxy_pass http://gemini_analyzer;
}
```

---

## Backup Strategy

### Database Backups (future)

```bash
# Daily backup script
#!/bin/bash
DATE=$(date +%Y%m%d)
pg_dump -U user dbname > /backup/db_$DATE.sql
```

### Application Backups

```bash
# Backup configuration and static files
tar -czf backup_$(date +%Y%m%d).tar.gz \
    /var/www/crypto-web \
    /etc/nginx/sites-available/crypto-web
```

---

## Troubleshooting

### Port Already in Use

```bash
# Check what's using the port
sudo lsof -i :8001

# Kill process
sudo kill -9 <PID>
```

### Permission Denied

```bash
# Fix permissions
sudo chown -R www-data:www-data /var/www/crypto-web
sudo chmod -R 755 /var/www/crypto-web
```

### Service Won't Start

```bash
# Check service status
sudo systemctl status gemini-analyzer

# View logs
sudo journalctl -u gemini-analyzer -n 50
```

### Frontend Shows 404

```nginx
# Ensure SPA routing is correct
location / {
    try_files $uri $uri/ /index.html;
}
```

---

## Scaling Strategy

### Horizontal Scaling

1. **Load Balancer:** Use Nginx or AWS ALB
2. **Multiple Instances:** Run multiple backend instances
3. **Session Management:** Use Redis for shared sessions

### Vertical Scaling

1. **Increase Resources:** Upgrade server CPU/RAM
2. **Optimize Code:** Profile and optimize bottlenecks
3. **Caching:** Implement response caching

---

## CI/CD Pipeline (Future)

```yaml
# .github/workflows/deploy.yml
name: Deploy

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      - name: Build Frontend
        run: |
          cd web/apps/gemini_analyzer/frontend
          npm install
          npm run build

      - name: Deploy to Server
        run: |
          scp -r dist/* user@server:/var/www/crypto-web/
          ssh user@server 'sudo systemctl restart gemini-analyzer'
```

---

## Support

For deployment issues:
1. Check logs: `journalctl -u <service-name> -f`
2. Verify configuration: `nginx -t`
3. Test endpoints: `curl http://localhost:PORT/health`
4. Review documentation: [API Reference](API_REFERENCE.md), [Architecture](ARCHITECTURE.md)
