# API Gateway (Future)

This document outlines the planned API Gateway implementation for the web applications.

## Overview

The API Gateway will serve as a single entry point for all web applications, providing:
- Request routing and load balancing
- Authentication and authorization
- Rate limiting
- Request/response logging
- API versioning
- CORS management

## Architecture

```
                    ┌─────────────────┐
                    │                 │
                    │  API Gateway    │
                    │   Port 8000    │
                    │                 │
                    └────────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Gemini      │    │   ATC        │    │   Future     │
│  Analyzer    │    │ Visualizer   │    │   Apps       │
│  Port 8001   │    │ Port 8002    │    │   Port 8003+ │
└──────────────┘    └──────────────┘    └──────────────┘
```

## Implementation Options

### Option 1: FastAPI Gateway

**Pros:**
- Same tech stack as existing services
- Easy to implement
- Fast and lightweight

**Cons:**
- Single point of failure
- Limited built-in features

**Implementation:**

```python
# web/gateway/main.py
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from httpx import AsyncClient
import asyncio

app = FastAPI(title="Crypto Probability API Gateway")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Service registry
SERVICES = {
    "gemini": "http://localhost:8001",
    "atc": "http://localhost:8002",
    "portfolio": "http://localhost:8003",
}

@app.api_route("/api/{service:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_request(service: str, request: Request):
    """Proxy request to appropriate service."""
    
    # Determine target service
    service_name = service.split("/")[0]
    base_url = SERVICES.get(service_name)
    
    if not base_url:
        return {"error": "Service not found"}, 404
    
    # Forward request
    target_url = f"{base_url}/api/{service}"
    
    async with AsyncClient() as client:
        # Copy headers
        headers = dict(request.headers)
        headers.pop("host", None)
        
        # Forward request
        response = await client.request(
            method=request.method,
            url=target_url,
            headers=headers,
            content=await request.body(),
            timeout=30.0
        )
        
        return Response(
            content=response.content,
            status_code=response.status_code,
            headers=dict(response.headers)
        )

@app.get("/")
async def root():
    return {
        "message": "Crypto Probability API Gateway",
        "services": list(SERVICES.keys()),
        "endpoints": {
            "gemini": "/api/gemini/...",
            "atc": "/api/atc/...",
            "health": "/health"
        }
    }

@app.get("/health")
async def health():
    """Health check for all services."""
    
    results = {}
    async with AsyncClient() as client:
        for name, url in SERVICES.items():
            try:
                response = await client.get(f"{url}/health", timeout=5.0)
                results[name] = {
                    "status": "healthy" if response.status_code == 200 else "unhealthy",
                    "response_time": response.elapsed.total_seconds()
                }
            except Exception as e:
                results[name] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
    
    return {
        "gateway": "healthy",
        "services": results
    }
```

---

### Option 2: Nginx Reverse Proxy

**Pros:**
- High performance
- Built-in load balancing
- Production-ready
- SSL termination

**Cons:**
- Separate technology stack
- More complex to configure

**Implementation:**

```nginx
# /etc/nginx/nginx.conf

upstream gemini_analyzer {
    server 127.0.0.1:8001;
    server 127.0.0.1:8001;  # Add more instances for HA
    keepalive 64;
}

upstream atc_visualizer {
    server 127.0.0.1:8002;
    keepalive 64;
}

upstream portfolio_app {
    server 127.0.0.1:8003;
    keepalive 64;
}

# Rate limiting
limit_req_zone $binary_remote_addr zone=general:10m rate=10r/s;

# API Gateway
server {
    listen 8000;
    server_name api.yourdomain.com;

    # API endpoints
    location /api/gemini/ {
        limit_req zone=general burst=20 nodelay;
        rewrite ^/api/gemini/(.*)$ /api/$1 break;
        proxy_pass http://gemini_analyzer;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /api/atc/ {
        limit_req zone=general burst=20 nodelay;
        rewrite ^/api/atc/(.*)$ /api/$1 break;
        proxy_pass http://atc_visualizer;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /api/portfolio/ {
        limit_req zone=general burst=20 nodelay;
        rewrite ^/api/portfolio/(.*)$ /api/$1 break;
        proxy_pass http://portfolio_app;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # Health check
    location /health {
        return 200 '{"status":"ok"}';
        add_header Content-Type application/json;
    }
}
```

---

### Option 3: Kong Gateway

**Pros:**
- Full-featured API gateway
- Plugin system
- Excellent monitoring
- Enterprise-ready

**Cons:**
- Requires additional infrastructure
- More complex setup
- Requires PostgreSQL/Redis

**Implementation:**

```yaml
# docker-compose.yml
version: '3.8'

services:
  kong:
    image: kong:latest
    ports:
      - "8000:8000"
      - "8443:8443"
      - "8001:8001"
    environment:
      KONG_DATABASE: postgres
      KONG_PG_HOST: postgres
      KONG_PG_USER: kong
      KONG_PG_PASSWORD: kong
      KONG_ADMIN_LISTEN: "0.0.0.0:8001"
    depends_on:
      - postgres

  postgres:
    image: postgres:13
    environment:
      POSTGRES_USER: kong
      POSTGRES_PASSWORD: kong
      POSTGRES_DB: kong
```

**Configure Services:**

```bash
# Add Gemini Analyzer service
curl -X POST http://localhost:8001/services \
  -d name=gemini-analyzer \
  -d url=http://gemini-analyzer:8001

# Add route
curl -X POST http://localhost:8001/services/gemini-analyzer/routes \
  -d paths[]=/api/gemini \
  -d strip_path=true

# Add ATC Visualizer service
curl -X POST http://localhost:8001/services \
  -d name=atc-visualizer \
  -d url=http://atc-visualizer:8002

# Add route
curl -X POST http://localhost:8001/services/atc-visualizer/routes \
  -d paths[]=/api/atc \
  -d strip_path=true
```

---

## Features

### 1. Service Discovery

**Dynamic Service Registration:**

```python
# web/shared/services/service_registry.py
import redis
from typing import Dict

class ServiceRegistry:
    def __init__(self, redis_url: str):
        self.redis = redis.from_url(redis_url)
    
    def register(self, service_name: str, url: str, ttl: int = 30):
        """Register service."""
        key = f"service:{service_name}"
        self.redis.setex(key, ttl, url)
    
    def discover(self, service_name: str) -> str:
        """Discover service URL."""
        key = f"service:{service_name}"
        return self.redis.get(key)
    
    def heartbeat(self, service_name: str, ttl: int = 30):
        """Send heartbeat."""
        key = f"service:{service_name}"
        self.redis.expire(key, ttl)
```

---

### 2. Rate Limiting

**Token Bucket Algorithm:**

```python
from fastapi import Request, HTTPException
from collections import defaultdict
import time

class RateLimiter:
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.requests = defaultdict(list)
    
    async def check_limit(self, request: Request):
        client_ip = request.client.host
        now = time.time()
        
        # Remove old requests
        minute_ago = now - 60
        self.requests[client_ip] = [
            t for t in self.requests[client_ip] if t > minute_ago
        ]
        
        # Check limit
        if len(self.requests[client_ip]) >= self.requests_per_minute:
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        
        # Add current request
        self.requests[client_ip].append(now)
        return True
```

---

### 3. Authentication

**JWT Middleware:**

```python
from fastapi import Depends, HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt

security = HTTPBearer()

async def verify_token(
    credentials: HTTPAuthorizationCredentials = Security(security)
) -> dict:
    """Verify JWT token."""
    token = credentials.credentials
    
    try:
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM]
        )
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

@app.get("/api/protected")
async def protected_route(user: dict = Depends(verify_token)):
    return {"message": f"Hello, {user['username']}"}
```

---

### 4. Request Logging

**Structured Logging:**

```python
import logging
import json
from fastapi import Request

logger = logging.getLogger(__name__)

async def log_request(request: Request):
    """Log incoming request."""
    start_time = time.time()
    
    # Process request
    
    duration = time.time() - start_time
    
    log_data = {
        "method": request.method,
        "path": request.url.path,
        "client_ip": request.client.host,
        "status_code": response.status_code,
        "duration_ms": round(duration * 1000, 2),
        "timestamp": datetime.utcnow().isoformat()
    }
    
    logger.info(json.dumps(log_data))
```

---

### 5. API Versioning

**Version Routing:**

```python
from fastapi import APIRouter

v1_router = APIRouter(prefix="/v1")
v2_router = APIRouter(prefix="/v2")

# Register versioned routes
app.include_router(v1_router)
app.include_router(v2_router)

# Forward to services
@app.api_route("/v1/gemini/{path:path}", methods=["GET", "POST"])
async def v1_gemini_proxy(path: str, request: Request):
    return await proxy_to_service("gemini", f"api/{path}", request, version="v1")

@app.api_route("/v2/gemini/{path:path}", methods=["GET", "POST"])
async def v2_gemini_proxy(path: str, request: Request):
    return await proxy_to_service("gemini", f"api/{path}", request, version="v2")
```

---

## Migration Path

### Phase 1: Basic Gateway

- [ ] Implement FastAPI-based gateway
- [ ] Set up service registry
- [ ] Configure routing for existing services
- [ ] Add health check endpoint

### Phase 2: Security

- [ ] Implement JWT authentication
- [ ] Add rate limiting
- [ ] Configure CORS
- [ ] Add request logging

### Phase 3: Advanced Features

- [ ] Add caching layer (Redis)
- [ ] Implement API versioning
- [ ] Add request/response transformation
- [ ] Set up monitoring and metrics

### Phase 4: High Availability

- [ ] Deploy gateway to multiple instances
- [ ] Configure load balancer
- [ ] Set up health checks
- [ ] Implement automatic failover

---

## Configuration

### Environment Variables

```bash
# Gateway Configuration
GATEWAY_PORT=8000
GATEWAY_HOST=0.0.0.0

# Service Registry
REDIS_URL=redis://localhost:6379

# Authentication
SECRET_KEY=your-secret-key-here
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Rate Limiting
RATE_LIMIT_REQUESTS_PER_MINUTE=60
RATE_LIMIT_BURST=20

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json
```

### Service Registration

```python
# web/gateway/config.py
SERVICES = {
    "gemini": {
        "url": "http://localhost:8001",
        "health_check": "/health",
        "timeout": 30.0,
        "retries": 3
    },
    "atc": {
        "url": "http://localhost:8002",
        "health_check": "/health",
        "timeout": 30.0,
        "retries": 3
    }
}
```

---

## Monitoring

### Metrics Collection

```python
from prometheus_client import Counter, Histogram, generate_latest

# Metrics
REQUEST_COUNT = Counter(
    'gateway_requests_total',
    'Total requests',
    ['service', 'method', 'status']
)

REQUEST_DURATION = Histogram(
    'gateway_request_duration_seconds',
    'Request duration',
    ['service', 'method']
)

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Collect request metrics."""
    start_time = time.time()
    
    response = await call_next(request)
    
    duration = time.time() - start_time
    service = request.path.split('/')[1]
    
    REQUEST_COUNT.labels(
        service=service,
        method=request.method,
        status=response.status_code
    ).inc()
    
    REQUEST_DURATION.labels(
        service=service,
        method=request.method
    ).observe(duration)
    
    return response

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(
        content=generate_latest(),
        media_type="text/plain"
    )
```

---

## Deployment

### Docker Compose

```yaml
version: '3.8'

services:
  gateway:
    build:
      context: ../../
      dockerfile: web/docker/Dockerfile.gateway
    ports:
      - "8000:8000"
    environment:
      - REDIS_URL=redis://redis:6379
    depends_on:
      - redis

  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

volumes:
  redis_data:
```

### Kubernetes

```yaml
# k8s/gateway-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api-gateway
spec:
  replicas: 3
  selector:
    matchLabels:
      app: api-gateway
  template:
    metadata:
      labels:
        app: api-gateway
    spec:
      containers:
      - name: gateway
        image: crypto-probability/api-gateway:latest
        ports:
        - containerPort: 8000
        env:
        - name: REDIS_URL
          value: "redis://redis-service:6379"
---
apiVersion: v1
kind: Service
metadata:
  name: api-gateway
spec:
  selector:
    app: api-gateway
  ports:
  - port: 8000
    targetPort: 8000
  type: LoadBalancer
```

---

## Testing

### Gateway Tests

```python
import pytest
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_health_check():
    async with AsyncClient(base_url="http://localhost:8000") as client:
        response = await client.get("/health")
        assert response.status_code == 200
        assert "services" in response.json()

@pytest.mark.asyncio
async def test_gemini_route():
    async with AsyncClient(base_url="http://localhost:8000") as client:
        response = await client.get("/api/gemini/health")
        assert response.status_code == 200

@pytest.mark.asyncio
async def test_atc_route():
    async with AsyncClient(base_url="http://localhost:8000") as client:
        response = await client.get("/api/atc/api/health")
        assert response.status_code == 200

@pytest.mark.asyncio
async def test_rate_limiting():
    async with AsyncClient(base_url="http://localhost:8000") as client:
        # Make more requests than allowed
        for _ in range(100):
            response = await client.get("/health")
        
        # Should get 429
        last_response = await client.get("/health")
        assert last_response.status_code == 429
```

---

## Rollback Strategy

If gateway issues arise:

1. **Disable gateway:**
   ```bash
   # Update DNS to point directly to services
   gemini.yourdomain.com → http://localhost:8001
   atc.yourdomain.com → http://localhost:8002
   ```

2. **Check logs:**
   ```bash
   docker logs gateway
   ```

3. **Fix issues and redeploy:**
   ```bash
   docker-compose up -d --build gateway
   ```

4. **Re-enable gateway** after testing

---

## Cost Estimation

| Component | Description | Monthly Cost (AWS) |
|-----------|-------------|-------------------|
| Load Balancer | ALB | ~$20 |
| API Gateway | Amazon API Gateway (pay per use) | ~$5-50 |
| Redis | ElastiCache | ~$15 |
| Monitoring | CloudWatch | ~$10 |
| **Total** | | **~$50-95** |

---

## References

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Nginx Documentation](https://nginx.org/en/docs/)
- [Kong Documentation](https://docs.konghq.com/)
- [Prometheus](https://prometheus.io/)
- [Kubernetes](https://kubernetes.io/)
