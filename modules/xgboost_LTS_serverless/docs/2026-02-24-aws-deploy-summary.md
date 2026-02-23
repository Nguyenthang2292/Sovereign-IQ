# Hoàn thành xuất sắc Phase 4: Deploy AWS Lambda Trainer thành công & test qua CloudFormation S3

Đây là một quá trình khá gian nan với rất nhiều vấn đề kĩ thuật sâu bên trong hệ thống AWS. Sau đây là chi tiết các nguyên nhân gốc rễ (Root Causes) và giải pháp đã được thực hiện:

### 1. Fix lỗi treo AWS SAM CLI do Pager

- **Nguyên nhân:** Lệnh `sam deploy` hoặc `aws cloudformation` liên tục bị treo console (nuốt output) trên môi trường Windows PowerShell do biến môi trường PAGER của AWS CLI chiếm quyền.
- **Giải pháp:** Bỏ qua AWS CLI và viết hẳn một **Python Boto3 Script (`modules/xgboost_LTS_serverless/scripts/_deploy_trainer_stack.py`)** để tracking trực tiếp quá trình deploy của Stack events theo thời gian thực vòng 5 giây một lần.

### 2. Fix lỗi "AlreadyExists" khi roll back CloudFormation

- **Nguyên nhân:** Từ những lần deploy bị hỏng trước, tên của SQS Queue (`xgboost-predictions`) và S3 Bucket (`xgboost-models-store`) đã bị kẹt trong AWS Account. Quá trình tạo stack mới báo lỗi vì resource bị trùng tên cố định. Secret cho `auto-trade/binance` cũng chứa được tạo trong AWS Secrets Manager khiến việc resolve parameter bị thất bại (`ResourceNotFoundException`).
- **Giải pháp:** Sửa file template:
  - Cho phép CloudFormation **tự động sinh ra tên CloudFormation S3 Bucket và SQS Queue** dựa theo Stack ID để luôn unique.
  - Tách Binance key khỏi Secrets Manager, thay vào đó pass qua Parameter overrides lúc deploy.

### 3. Fix lỗi "Image manifest not supported" (Quan trọng nhất)

- **Nguyên nhân:** Lambda báo lỗi `The image manifest, config or layer media type for the source image is not supported.` Lý do là các bản cập nhật gần đây của Docker Desktop tự động gắn thêm tính năng **Buildx Provenance / SLSA Attestation** (chứng minh nguồn gốc ảnh layer) khiến file image bị chuyển sang định dạng `OCI Image Index` thay vì Docker Manifest thông thường, thứ mà AWS Lambda **không hỗ trợ**.
- **Giải pháp:** Cập nhật script `build_trainer.sh`, giới hạn docker build với flag **`--provenance=false`** để ép chuẩn `v2+json` manifest thuần túy. Image sau đó đã được build lại và đẩy lên ECR thành công.

### 4. Fix lỗi RuntimeError từ Numba (Trong lúc thực thi thật)

- **Nguyên nhân:** Sau khi deploy thành công, call hàm thử qua lệnh `aws lambda invoke`. Hàm chạy nhưng báo lỗi `RuntimeError: cannot cache function 'fibonacci': no locator available`.
  - Thư viện Cốt lõi `pandas-ta` dùng `numba` để tối ưu (njit). Nhưng `numba` chạy lần đầu cần quyền Write để ghi cache file.
  - Trong AWS Lambda, toàn bộ File System đều bị **read-only* trừ vùng nhớ `/tmp`.
- **Giải pháp:** Bổ sung Environment Variable **`NUMBA_CACHE_DIR: /tmp`** thẳng vào file CloudFormation YAML template.

---

## 🚀 Kết quả (End-to-End Test Passed)

Sau khi apply tất cả thay đổi trên, Stack `xgboost-lts-serverless-staging` đã **`UPDATE_COMPLETE`**.

Lệnh `invoke` kích hoạt hàm trực tiếp trên server AWS với Payload Test:

```json
{"symbol": "BTC/USDT", "timeframe": "15m", "model_version": "v1", "s3_bucket": "xgboost-lts-serverless-staging-models-081338828929", "fetch_limit": 1500}
```

**Lambda Response:**

```json
{"status": "ok", "symbol": "BTC/USDT", "s3_key": "BTCUSDT_15m_v1.json", "size_bytes": 627527, "elapsed_s": 6.5}
```

Mô hình (Model) đã hoàn toàn được **train trực tiếp trên AWS Lambda Serverless** (chỉ tốn 6.5s) và **đẩy thành công vào S3 bucket**. Tiến độ trong `lambda-trainer-tasks.md` đã được tracking (Done T4.1, T4.2, T4.3). Toàn bộ Phase 4 Deployment đã hoàn tất viên mãn!
