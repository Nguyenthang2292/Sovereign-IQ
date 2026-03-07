Để lấy danh sách các model chuyên biệt về **"đọc" hình ảnh** (VLM - Vision Language Models) và loại bỏ các model **"tạo" hình ảnh** (Generative), bạn vẫn sử dụng API List Models chuẩn, nhưng cần thêm một bước **lọc (filter)** theo từ khóa trong mã code của mình.

Hiện tại, API của Alibaba Cloud Model Studio chưa hỗ trợ tham số lọc trực tiếp trên URL (ví dụ `?type=vision`), vì vậy cách duy nhất là lấy toàn bộ danh sách và lọc phía client.

### 1. Gọi API để lấy toàn bộ danh sách

Bạn thực hiện yêu cầu **GET** đến endpoint tương thích OpenAI:

* **Endpoint:** `https://dashscope-intl.aliyuncs.com/compatible-mode/v1/models`
* **Header:** `Authorization: Bearer <YOUR_API_KEY>`

### 2. Cách lọc "Model đọc hình ảnh"

Các model có khả năng nhìn và phân tích ảnh của Qwen luôn tuân theo quy tắc đặt tên có chứa hậu tố **`-vl-`** (viết tắt của **Vision-Language**).

Bạn cần lọc các model có ID chứa chữ **"vl"** và **loại bỏ** các model có chứa chữ **"edit"** hoặc **"wan"** (vì đó là model chỉnh sửa và tạo video).

#### Ví dụ mã Python để lọc danh sách:

```python
import requests

api_key = "YOUR_API_KEY"
url = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/models"

headers = {
    "Authorization": f"Bearer {api_key}"
}

response = requests.get(url, headers=headers)
if response.status_code == 200:
    all_models = response.json().get("data", [])
    
    # Lọc các model có chữ 'vl' nhưng không phải là model chỉnh sửa/tạo
    vision_models = [
        m["id"] for m in all_models 
        if "-vl" in m["id"].lower() and "edit" not in m["id"].lower()
    ]
    
    # Sắp xếp từ trên xuống dưới
    vision_models.sort()
    
    print("Danh sách model đọc hình ảnh:")
    for model in vision_models:
        print(f"- {model}")
else:
    print(f"Lỗi: {response.status_code}")

```

### 3. Các model tiêu biểu bạn sẽ nhận được

Sau khi lọc, danh sách của bạn sẽ bao gồm các model mạnh mẽ nhất để trích xuất Text/JSON như:

* **qwen-vl-max**: Model mạnh nhất hiện tại cho các tác vụ hiểu ảnh phức tạp.
* **qwen-vl-plus**: Cân bằng giữa tốc độ và độ chính xác.
* **qwen2.5-vl-72b-instruct**: Model mã nguồn mở tốt nhất cho việc đọc tài liệu.
* **qwen2.5-vl-7b-instruct**: Model nhỏ, tốc độ phản hồi cực nhanh.

### Tại sao không dùng các model khác?

* **Dòng `wan-**`: Đây là model sinh video và ảnh (Text-to-Video/Image).
* **Dòng `qwen-image-edit-**`: Đây là model chỉ nhận ảnh vào để chỉnh sửa (thêm/bớt vật thể), không có khả năng trả về chuỗi Text hay JSON phân tích nội dung.

**Lưu ý:** Nếu bạn đang sử dụng khu vực quốc tế, hãy đổi URL thành `dashscope-intl.aliyuncs.com`.