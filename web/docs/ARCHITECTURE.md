# Architecture Overview

## System Architecture

The web folder follows a **microservices-lite** architecture with the following layers:

```
┌─────────────────────────────────────────────────────────────────┐
│                         Client Layer                          │
│                   (Browser / Mobile App)                      │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Presentation Layer                       │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │  Gemini     │    │   ATC       │    │  Future     │  │
│  │  Analyzer   │    │ Visualizer  │    │   App       │  │
│  │  (:5173)    │    │  (:5174)    │    │   (:5175)    │  │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘  │
└─────────┼──────────────────┼──────────────────┼──────────────┘
          │                  │                  │
          ▼                  ▼                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                      API Layer (Future)                     │
│                   API Gateway (:8000)                        │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Service Layer                          │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │  Gemini     │    │   ATC       │    │  Future     │  │
│  │  Service    │    │  Service    │    │   Service   │  │
│  │  (:8001)    │    │  (:8002)    │    │   (:8003)    │  │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘  │
└─────────┼──────────────────┼──────────────────┼──────────────┘
          │                  │                  │
          └──────────────────┼──────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Data Layer                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │  Modules    │    │  Exchanges  │    │  External   │  │
│  │  (ATC/HMM) │    │ (Binance)   │    │   APIs      │  │
│  └─────────────┘    └─────────────┘    └─────────────┘  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                      Shared Layer                           │
│  ┌──────────┐  ┌───────────┐  ┌──────────────┐          │
│  │  Utils   │  │ Middleware │  │    Models    │          │
│  └──────────┘  └───────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
```

## Component Responsibilities

### Presentation Layer
- **Frontend Framework:** Vue 3 + Vite
- **State Management:** Composition API (local state)
- **Routing:** Vue Router 4
- **Styling:** Tailwind CSS (Gemini), Custom CSS (ATC)
- **Charts:** ApexCharts (ATC), Mermaid (Gemini diagrams)

### API Layer (Future)
- **Gateway Framework:** FastAPI
- **Responsibility:** Route requests, authentication, rate limiting
- **Load Balancing:** Multiple service instances

### Service Layer
- **Backend Framework:** FastAPI
- **Responsibility:** Business logic, data processing
- **API Documentation:** Swagger/OpenAPI
- **CORS:** Configured per app

### Data Layer
- **Business Logic:** Crypto Probability modules (ATC, HMM, XGBoost, etc.)
- **External APIs:** Binance, Kraken, KuCoin, Gate.io, OKX, Bybit, MEXC, Huobi
- **Data Formats:** OHLCV (Open, High, Low, Close, Volume)

### Shared Layer
- **Utilities:** Task management, logging, error handling
- **Middleware:** CORS, authentication, request logging
- **Models:** Pydantic models for validation
- **Services:** Exchange connection, data fetching (future)

## Communication Patterns

### Synchronous Request/Response
```
Frontend → Backend API → Module → External API
    ↑                    ↑          ↑
  JSON Response      Process    Market Data
```

### Async Background Tasks
```
Frontend → Backend API → TaskManager → Background Thread
    ↑                    ↑
  Poll Status       Long-running Operation
```

### Event Streaming (Future)
```
WebSocket Backend → Frontend
    ↓
Real-time Updates
```

## Port Allocation Strategy

| Type | Range | Purpose |
|------|-------|---------|
| API Gateway | 8000 | Main entry point (future) |
| Backend Services | 8001-8099 | Individual app backends |
| Frontend Dev Servers | 5173-5199 | Individual app frontends |
| Static Services | N/A | Served by backends |

## Data Flow Examples

### 1. Chart Analysis Request
```
1. User submits symbol & timeframe in frontend
2. Frontend POSTs to /api/analyze/single
3. Backend validates request with Pydantic
4. Backend calls DataFetcher to get OHLCV from exchange
5. Backend calls Gemini Analyzer module
6. Module processes data and generates analysis
7. Backend saves chart image to /static/charts/
8. Backend returns JSON response to frontend
9. Frontend displays chart and analysis
```

### 2. Batch Scanner Request
```
1. User configures scan parameters
2. Frontend POSTs to /api/batch/scan
3. Backend creates background task via TaskManager
4. Backend returns session_id to frontend
5. Frontend starts polling /api/batch/scan/{session_id}/status
6. TaskManager processes symbols in batches
7. Each scan result is logged to file
8. Task completes after all symbols scanned
9. Frontend fetches results via /api/batch/results/{filename}
```

## Security Considerations

### Current
- CORS configuration per app
- Input validation with Pydantic
- Error message sanitization (no sensitive paths)
- Source map configuration (hidden in production)

### Future Enhancements
- Authentication/Authorization (JWT tokens)
- API key management
- Rate limiting per endpoint
- Request signing
- HTTPS enforcement

## Scalability Strategy

### Horizontal Scaling
1. **Frontend:** Deploy to CDN (Cloudflare, AWS CloudFront)
2. **Backend:** Multiple instances behind load balancer
3. **Database:** Replication and sharding (future)

### Vertical Scaling
1. Increase server resources (CPU, RAM)
2. Optimize database queries
3. Cache frequently accessed data

### Microservices Transition
```
Monolithic → Modular → Microservices
    │            │             │
 Single app  Multiple apps  Independent services
 (current)    (current)       (future)
```

## Performance Optimization

### Frontend
- Code splitting by route
- Lazy loading components
- Image optimization (vite-plugin-imagemin)
- Minification and tree-shaking
- Debounced input handlers

### Backend
- Background task execution
- Connection pooling (exchanges)
- Response caching (future)
- Database query optimization (future)
- Graceful degradation under load

## Deployment Options

### Development
- Frontend: `npm run dev` (Vite HMR)
- Backend: `uvicorn --reload`
- Proxy: Vite proxy to backend

### Production
- Frontend: `npm run build` + Nginx serve
- Backend: `gunicorn` + `uvicorn workers`
- Orchestration: Docker Compose, Kubernetes

### Cloud Platforms
- AWS: ECS + ALB + CloudFront
- Google Cloud: Cloud Run + Cloud Load Balancer
- Azure: App Service + Application Gateway
- Vercel/Netlify: Frontend only

## Monitoring & Observability (Future)

### Metrics
- Request rate and latency
- Error rates
- Resource usage (CPU, RAM)
- Background task queue length

### Logging
- Structured JSON logs
- Centralized log aggregation (ELK stack)
- Log level filtering per environment

### Tracing
- Distributed tracing (OpenTelemetry)
- Request flow visualization
- Performance bottleneck identification

## Technology Rationale

### Why FastAPI?
- Async/await support for better performance
- Automatic API documentation (Swagger)
- Type hints with Pydantic validation
- Modern Python features

### Why Vue 3?
- Composition API for better code organization
- Lightweight and fast
- Excellent developer experience
- Strong ecosystem (Vite, Pinia)

### Why Tailwind CSS?
- Utility-first approach
- Smaller bundle size
- Customizable design system
- Dark mode support out of the box

## Future Roadmap

### Short Term (1-3 months)
- [ ] API Gateway implementation
- [ ] Authentication system
- [ ] Rate limiting
- [ ] Health monitoring dashboard

### Medium Term (3-6 months)
- [ ] WebSocket support for real-time updates
- [ ] Caching layer (Redis)
- [ ] Database for storing analysis results
- [ ] Automated testing CI/CD

### Long Term (6-12 months)
- [ ] Full microservices architecture
- [ ] Cloud-native deployment
- [ ] Advanced monitoring and alerting
- [ ] Mobile apps (React Native/Flutter)
