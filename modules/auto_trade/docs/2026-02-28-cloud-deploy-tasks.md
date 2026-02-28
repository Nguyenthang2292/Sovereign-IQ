# Cloud Deploy — Implementation Tasks

## Goal

Thêm nút BUILD & DEPLOY vào GUI: đóng gói config → upload S3 → chạy headless bot trên EC2 Docker, kèm nút START/STOP EC2.

## Pre-conditions (quyết định trước khi code)

- [ ] EC2 Key Pair: dùng key pair có sẵn hay tạo mới? → cập nhật `aws_config.py`
- [ ] S3 bucket name: xác nhận `s3_bucket` trong `.env` đã có giá trị
- [ ] `requirements-headless.txt`: tách từ `requirements.txt`, bỏ `customtkinter`, `darkdetect`, `pywin32`

---

## Tasks

- [x] **T1: `requirements-headless.txt`**
  Tạo tại project root, copy `requirements.txt` rồi xóa các dòng: `customtkinter`, `darkdetect`, `pywin32`, `pywinpty`.
  → Verify: `grep -v "customtkinter\|darkdetect" requirements-headless.txt`

- [x] **T2: `headless_bot.py`** (project root)

  ```
  1. argparse: --settings (default: modules/auto_trade/settings.yaml)
  2. fetch_secrets_to_env()  ← gọi secrets_manager
  3. load settings.yaml → build config dict
  4. Khởi ScannerManager(config, auto_start=True)
  5. Khởi AutoTradeEngine(config)
  6. signal.pause()
  ```

  Không import bất kỳ GUI module nào.
  → Verify: `python headless_bot.py --help` không raise ImportError

- [x] **T3: `Dockerfile`** (project root)

  ```dockerfile
  FROM python:3.11-slim
  WORKDIR /app
  COPY requirements-headless.txt .
  RUN pip install --no-cache-dir -r requirements-headless.txt
  COPY . .
  CMD ["python", "headless_bot.py", "--settings", "/app/settings.yaml"]
  ```

  → Verify: `docker build -t autotrade-test .` thành công, image < 1.5 GB

- [ ] **T4: `deploy/s3_uploader.py`**

  ```python
  def upload(local_path: str, bucket: str, on_progress=None) -> str:
      # Upload tới deployments/bot-{timestamp}.tar.gz
      # Copy sang deployments/latest.tar.gz
      # Return s3://bucket/deployments/latest.tar.gz
  ```

  → Verify: upload file test, kiểm tra cả 2 key tồn tại trên S3

- [ ] **T5: `deploy/secrets_manager.py`**

  ```python
  def push_binance_secrets(api_key: str, api_secret: str) -> None
  def push_aws_config(region: str, table: str, ...) -> None
  def fetch_secrets_to_env() -> None   # dùng trong headless_bot.py
  def secrets_exist() -> bool          # check trước khi BUILD
  ```

  → Verify: `secrets_exist()` trả True sau khi push; headless_bot đọc được key từ env

- [ ] **T6: `deploy/ec2_manager.py`**

  ```python
  class EC2Manager:
      def provision_if_needed(key_pair, sg_id, iam_profile) -> str  # instance_id
      def get_status() -> dict   # {state, instance_id, public_ip}
      def start() -> None
      def stop() -> None
      def deploy(s3_uri: str) -> None  # SSM RunCommand: pull + docker rebuild
  ```

  EC2 tag: `Name=autotrade-bot`. Tìm lại bằng `describe_instances` filter.
  → Verify: `get_status()` trả dict hợp lệ khi EC2 đang running

- [ ] **T7: `deploy/builder.py`** + `deploy/user_data.sh`
  `builder.py`:

  ```python
  def build_and_deploy(settings_dict: dict, on_log=None) -> str:
      # 1. Export settings_dict → settings.yaml tạm
      # 2. tar.gz source (exclude: __pycache__/ .env artifacts/ logs/ charts/ outputs/ .git/)
      # 3. Inject settings.yaml + Dockerfile vào archive
      # 4. s3_uploader.upload() → s3_uri
      # 5. ec2_manager.provision_if_needed()
      # 6. ec2_manager.deploy(s3_uri)
  ```

  `user_data.sh`: apt install docker + awscli, tạo watcher systemd service poll S3 mỗi 60s, checksum-diff → docker rebuild.
  → Verify: artifact tạo ra có `settings.yaml` + `Dockerfile` bên trong

- [ ] **T8: `deploy/__init__.py`**
  Export: `EC2Manager`, `build_and_deploy`, `secrets_exist`, `push_binance_secrets`.
  → Verify: `from modules.auto_trade.deploy import EC2Manager` không lỗi

- [ ] **T9: `gui/tabs/deploy_tab.py`**
  Layout (customtkinter):
  - Row 1: EC2 status label (state / instance_id / public_ip)
  - Row 2: Last deploy label
  - Row 3: Buttons — `[🔨 BUILD & DEPLOY]` `[▶ START]` `[⏹ STOP]`
  - Row 4: Deploy log textbox (scrollable, read-only)

  Behavior:
  - BUILD chạy `threading.Thread(target=build_and_deploy, ...)` → stream log qua callback
  - START/STOP disable button → gọi `ec2_manager.start/stop()` → re-enable sau khi xong
  - First-time: nếu `not secrets_exist()` → mở dialog nhập Binance keys trước
  → Verify: Build button stream log ra textbox mà không freeze GUI

- [ ] **T10: Wire vào `main_window`**
  - Thêm `deploy_tab.py` vào tab list trong `layout.py`
  - Thêm updater `"ec2_status"` vào `updaters.py` `setup_updaters()`:

    ```python
    self.updaters["ec2_status"] = PeriodicUpdater(self.parent._thread_refresh_ec2_status, interval=30)
    ```

  - Thêm handler `_thread_refresh_ec2_status` trong `main_window.py` → push vào `_update_queue` kind=`"ec2_status"`
  - Xử lý `"ec2_status"` trong `_drain_update_queue` → gọi `deploy_tab.update_status(data)`
  → Verify: Tab "Cloud Deploy" xuất hiện trong GUI, status tự refresh mỗi 30s

---

## Done When

- [ ] Bấm BUILD & DEPLOY từ GUI → bot chạy trên EC2 với đúng settings hiện tại
- [ ] Bấm STOP EC2 → EC2 dừng, bot offline
- [ ] Bấm START EC2 → EC2 khởi động lại, bot tự resume (Docker restart=always)
- [ ] Tắt máy local → bot trên EC2 vẫn chạy

---

## Notes

- Thứ tự implement: T1 → T2 → T3 (verify Docker local) → T4 → T5 → T6 → T7 → T8 → T9 → T10
- T4, T5, T6 độc lập nhau, có thể làm song song
- `deploy/user_data.sh` cần hardcode S3 bucket name → lấy từ `AWSConfig.s3_bucket`
- IAM Role EC2 cần tạo thủ công 1 lần trên AWS Console trước khi chạy T6 (xem design doc phần IAM)
