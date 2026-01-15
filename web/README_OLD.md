# Gemini Chart Analyzer Web Interface

Giao diện web cho module Gemini Chart Analyzer, hỗ trợ phân tích biểu đồ kỹ thuật và batch scanning thị trường với Google Gemini AI.

## Tính Năng

- **Chart Analyzer**: Phân tích biểu đồ kỹ thuật cho một symbol
  - Single Timeframe: Phân tích trên một khung thời gian
  - Multi-Timeframe: Phân tích trên nhiều khung thời gian với weighted aggregation
  
- **Batch Scanner**: Quét toàn bộ thị trường để tìm signals
  - Single Timeframe: Scan với một timeframe
  - Multi-Timeframe: Scan với nhiều timeframes

## Cấu Trúc

```
web/
├── app.py                    # FastAPI server
├── api/                      # API routes
│   ├── chart_analyzer.py    # Chart Analyzer endpoints
│   └── batch_scanner.py     # Batch Scanner endpoints
├── static/
│   └── vue/                  # Vue 3 frontend
│       ├── src/
│       │   ├── components/   # Vue components
│       │   ├── services/     # API services
│       │   └── router/       # Vue Router
│       └── package.json
```

## Cài Đặt

### 1. Backend (FastAPI)

```bash
# Cài đặt Python dependencies (từ thư mục root của project)
pip install -r requirements.txt
```

### 2. Frontend (Vue 3)

```bash
cd static/vue

# Cài đặt Node.js dependencies
npm install

# Build production
npm run build
```

## Chạy Ứng Dụng

### Development Mode

**Terminal 1 - Backend:**
```bash
# Từ thư mục root của project
python main_web.py

# Hoặc sử dụng uvicorn trực tiếp
uvicorn main_web:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 - Frontend:**
```bash
# Từ thư mục web/static/vue/
npm run dev
```

Truy cập:
- Frontend: http://localhost:5173
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

### Production Mode

**Build frontend:**
```bash
cd static/vue
npm run build
```

**Chạy server:**
```bash
# Từ thư mục root của project
python main_web.py

# Hoặc sử dụng uvicorn
uvicorn main_web:app --host 0.0.0.0 --port 8000
```

Truy cập: http://localhost:8000

## API Endpoints

### Chart Analyzer

- `POST /api/analyze/single` - Single timeframe analysis
- `POST /api/analyze/multi` - Multi-timeframe analysis

### Batch Scanner

- `POST /api/batch/scan` - Start batch scan
- `GET /api/batch/results/{filename}` - Get saved results
- `GET /api/batch/list` - List all results

### Static Files

- `GET /static/charts/*` - Serve chart images
- `GET /static/results/*` - Serve analysis results

## Cấu Hình

### Environment Variables

Tạo file `.env` trong thư mục `web/static/vue/` (optional):

```env
VITE_API_BASE_URL=http://localhost:8000/api
# Sourcemap configuration for production builds
# Options: 'true' (inline sourcemaps), 'false' (no sourcemaps), or unset/empty (hidden sourcemaps - default)
GENERATE_SOURCEMAPS=hidden
```

**Sourcemap Configuration:**
- Không set hoặc `GENERATE_SOURCEMAPS=hidden` (mặc định): Tạo sourcemaps nhưng không expose ra browser, an toàn cho production và cho phép server-side debugging
- `GENERATE_SOURCEMAPS=true`: Tạo inline sourcemaps (có thể expose source code)
- `GENERATE_SOURCEMAPS=false`: Không tạo sourcemaps

Ví dụ cho CI/CD:
```bash
# Production build với hidden sourcemaps (mặc định)
npm run build

# Production build với inline sourcemaps (cho debugging)
GENERATE_SOURCEMAPS=true npm run build

# Production build không có sourcemaps
GENERATE_SOURCEMAPS=false npm run build
```

### Gemini API Key

Đảm bảo đã cấu hình Gemini API key trong `config/config_api.py`:

```python
GEMINI_API_KEY = 'your-api-key-here'
```

## Sử Dụng

### Chart Analyzer

1. Chọn mode: Single hoặc Multi-timeframe
2. Nhập symbol (ví dụ: BTC/USDT)
3. Chọn timeframe(s)
4. Cấu hình indicators (tùy chọn)
5. Click "Bắt Đầu Phân Tích"

### Batch Scanner

1. Chọn mode: Single hoặc Multi-timeframe
2. Cấu hình scan parameters
3. Click "Bắt Đầu Scan"
4. Xem kết quả trong bảng với filtering và sorting

## Troubleshooting

### Lỗi "Module not found"

Đảm bảo đang chạy từ thư mục gốc của project và Python path đã được cấu hình đúng.

### Lỗi "Vue app not built"

Chạy `npm run build` trong thư mục `web/static/vue/` để build frontend.

### Lỗi CORS

Kiểm tra CORS settings trong `app.py`. Trong production, nên chỉ định origins cụ thể thay vì `"*"`.

### Lỗi "GEMINI_API_KEY not found"

Kiểm tra đã cấu hình API key trong `config/config_api.py` hoặc biến môi trường.

## License

Phần của dự án crypto-probability.

