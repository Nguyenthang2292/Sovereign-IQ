# Cloud Deploy Feature Design

**Date:** 2026-02-28  
**Status:** Design complete, pending implementation

## Summary

Thêm tab "Cloud Deploy" vào GUI cho phép người dùng đóng gói config hiện tại thành một headless bot, deploy lên EC2 AWS, và điều khiển EC2 (start/stop) trực tiếp từ GUI.

---

## Decisions

| Topic | Decision | Reason |
|-------|----------|--------|
| Container runtime | Docker | Dễ CI/CD, isolation tốt |
| EC2 instance | t3.micro | Backend là serverless (Lambda), không cần RAM nhiều |
| Code delivery | S3 artifact (.tar.gz) | S3 đã có sẵn trong project, không phụ thuộc git remote |
| Credentials | AWS Secrets Manager | Key không bao giờ rời khỏi cloud, dễ rotate |

---

## Architecture

```
GUI (local Windows)
├── [BUILD & DEPLOY] button
│   ├── 1. Thu thập settings.yaml hiện tại từ GUI state
│   ├── 2. Tạo headless artifact (Python CLI, không GUI)
│   ├── 3. Đóng gói → bot-{timestamp}.tar.gz (loại trừ __pycache__, .env, artifacts/, logs/)
│   ├── 4. Upload → s3://bucket/deployments/latest.tar.gz
│   ├── 5. Provision EC2 t3.micro nếu chưa có (ap-southeast-1)
│   └── 6. EC2 tự phát hiện artifact mới → docker build → docker run
│
├── [▶ START EC2] button  → boto3: ec2.start_instances()
├── [⏹ STOP EC2] button   → boto3: ec2.stop_instances()
└── Status bar polling 30s: EC2 state (running / stopped / pending)

EC2 t3.micro (ap-southeast-1)
└── Docker container
    ├── Startup: fetch secrets từ AWS Secrets Manager → os.environ
    ├── python headless_bot.py --settings /app/settings.yaml
    │   ├── Scanner (auto_start=True)
    │   └── AutoTrade engine
    └── Logs → stdout → CloudWatch
```

---

## New Files

```
modules/auto_trade/deploy/
├── __init__.py
├── builder.py          # Thu thập config + đóng gói artifact
├── ec2_manager.py      # Provision / start / stop / status / deploy EC2
├── s3_uploader.py      # Upload artifact lên S3
├── secrets_manager.py  # Push Binance keys + AWS creds vào Secrets Manager
└── user_data.sh        # EC2 bootstrap script (chạy 1 lần khi EC2 mới tạo)

modules/auto_trade/gui/tabs/
└── deploy_tab.py       # customtkinter tab "Cloud Deploy"

headless_bot.py         # CLI entry point (project root), chạy trên EC2
Dockerfile              # python:3.11-slim, bỏ customtkinter/darkdetect
```

---

## Module Details

### `builder.py`

Flow khi bấm BUILD:

1. Đọc `settings.yaml` từ GUI state hiện tại
2. Copy source code, exclude: `__pycache__/`, `.env`, `artifacts/`, `logs/`, `charts/`, `outputs/`
3. Inject `settings.yaml` đã export vào root của artifact
4. Tạo `Dockerfile` tối giản vào artifact
5. Đóng gói → `/tmp/bot-YYYY-MM-DDTHHMMSS.tar.gz`
6. Gọi `s3_uploader.upload()` → trả về `s3_uri`
7. Gọi `ec2_manager.deploy(s3_uri)` → EC2 pull về và restart container
8. Stream từng bước ra callback (để hiển thị trong Deploy Log của GUI)

### `ec2_manager.py`

```python
class EC2Manager:
    def provision_if_needed() -> str      # Tạo EC2 mới nếu chưa có, trả về instance_id
    def get_status() -> dict              # {"state": "running", "instance_id": ..., "public_ip": ...}
    def start() -> None                  # ec2.start_instances()
    def stop() -> None                   # ec2.stop_instances()
    def deploy(s3_uri: str) -> None      # SSM RunCommand: pull S3 + docker rebuild
```

EC2 được tag `{"Name": "autotrade-bot"}` để tìm lại qua `describe_instances` filter.

### `s3_uploader.py`

Upload artifact lên hai path:
- `s3://bucket/deployments/bot-{timestamp}.tar.gz` (versioned)
- `s3://bucket/deployments/latest.tar.gz` (EC2 polling target)

### `secrets_manager.py`

First-time setup: GUI hỏi nhập Binance API key/secret → push lên:
- `autotrade/binance` → `{"api_key": "...", "api_secret": "..."}`
- `autotrade/aws` → `{"region": "ap-southeast-1", "dynamodb_table": "AutoTrade", ...}`

EC2 container fetch khi start thông qua IAM role (không cần hardcode credentials).

### `user_data.sh` (EC2 bootstrap, chạy 1 lần)

```bash
#!/bin/bash
apt-get update && apt-get install -y docker.io awscli
systemctl start docker

# Watcher service: poll S3 mỗi 60s, rebuild khi có artifact mới
cat > /opt/autotrade/watcher.sh << 'EOF'
  while true; do
    aws s3 cp s3://bucket/deployments/latest.tar.gz /opt/autotrade/latest.tar.gz
    # So sánh checksum, nếu thay đổi → rebuild
    # docker stop autotrade || true
    # tar xzf latest.tar.gz -C /opt/autotrade/app/
    # docker build -t autotrade /opt/autotrade/app/
    # docker run -d --restart=always --name autotrade autotrade
    sleep 60
  done
EOF

systemctl enable autotrade-watcher
systemctl start autotrade-watcher
```

### `headless_bot.py` (project root)

```python
# 1. Fetch secrets từ AWS Secrets Manager → inject vào os.environ
# 2. Load settings.yaml từ --settings arg
# 3. Khởi động ScannerManager với auto_start=True
# 4. Khởi động AutoTrade engine
# 5. signal.pause() — block forever, log ra stdout
```

Tái dụng hoàn toàn `scanner_manager`, `auto_trade` engine, `data_service` hiện có.  
Không import `customtkinter` hay bất kỳ GUI module nào.

### `Dockerfile`

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip uninstall -y customtkinter darkdetect 2>/dev/null || true

COPY . .

CMD ["python", "headless_bot.py", "--settings", "/app/settings.yaml"]
```

---

## GUI Tab: "Cloud Deploy"

**File:** `modules/auto_trade/gui/tabs/deploy_tab.py`

```
┌─────────────────────────────────────────────────────────┐
│  ☁️  CLOUD DEPLOY                                        │
├─────────────────────────────────────────────────────────┤
│  EC2 Status:  🟢 running  │  i-0abc123  │  3.1.2.4      │
│  Last Deploy: 2026-02-28 12:00  │  bot-20260228T1200     │
├─────────────────────────────────────────────────────────┤
│  [ 🔨 BUILD & DEPLOY ]   [ ▶ START EC2 ]   [ ⏹ STOP ]  │
├─────────────────────────────────────────────────────────┤
│  Deploy Log:                                             │
│  ✅ Collected settings.yaml                              │
│  ✅ Packaged bot-2026-02-28T1200.tar.gz (2.1 MB)        │
│  ✅ Uploaded → s3://bucket/deployments/latest            │
│  ⏳ EC2 pulling artifact...                              │
└─────────────────────────────────────────────────────────┘
```

**Behavior:**
- `BUILD & DEPLOY` chạy trên background thread → stream log vào Deploy Log box, không block GUI
- `START / STOP` gọi boto3 ngay, disable button trong lúc chờ EC2 chuyển trạng thái
- EC2 Status polling mỗi 30s (thêm updater `"ec2_status"` vào `UpdaterManager`)
- Khi EC2 `stopped`: BUILD vẫn chạy + upload S3, nhưng hiện warning `"EC2 stopped — start EC2 to apply"`
- First-time: nếu chưa có EC2 + chưa có secrets → wizard hỏi nhập Binance keys trước khi BUILD

---

## EC2 IAM Role (cần tạo)

```json
{
  "Effect": "Allow",
  "Action": [
    "secretsmanager:GetSecretValue",
    "s3:GetObject",
    "s3:ListBucket",
    "dynamodb:*",
    "logs:CreateLogGroup",
    "logs:CreateLogStream",
    "logs:PutLogEvents"
  ],
  "Resource": "*"
}
```

---

## Open Questions (khi implement)

1. EC2 Key Pair: dùng key pair có sẵn hay tạo mới? (cần cho SSH debug)
2. Security Group: chỉ cần outbound HTTPS (443) cho Binance/AWS APIs — không cần inbound
3. `requirements.txt` trên EC2 có cần tách riêng `requirements-headless.txt` để bỏ bớt GUI deps?
