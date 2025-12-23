"""
I Ching result information extractor using Google Gemini API.
"""

import json
import re
from typing import Optional

from modules.common.ui.logging import log_error, log_info, log_success, log_warn
from modules.gemini_chart_analyzer.core.gemini_analyzer import GeminiAnalyzer
from modules.iching.core.data_models import HaoInfo, IChingResult


class IChingResultExtractor:
    """Trích xuất thông tin từ ảnh kết quả I Ching sử dụng Gemini API."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Khởi tạo IChingResultExtractor.
        
        Args:
            api_key: Google Gemini API key (nếu None, sẽ lấy từ config)
        """
        self.analyzer = GeminiAnalyzer(api_key=api_key)
        log_info("Đã khởi tạo IChingResultExtractor")
    
    def extract_from_image(self, image_path: str) -> Optional[IChingResult]:
        """
        Trích xuất thông tin từ ảnh kết quả I Ching.
        
        Args:
            image_path: Đường dẫn đến file ảnh screenshot
            
        Returns:
            IChingResult object nếu thành công, None nếu thất bại
        """
        log_info(f"Bắt đầu trích xuất thông tin từ ảnh: {image_path}")
        
        try:
            # Tạo prompt chuyên biệt cho I Ching
            prompt = self._create_extraction_prompt()
            
            # Gọi Gemini API để phân tích ảnh
            log_info("Đang gửi ảnh lên Google Gemini để trích xuất thông tin...")
            response_text = self.analyzer.analyze_chart(
                image_path=image_path,
                symbol="I Ching Result",
                timeframe="result",
                prompt_type="custom",
                custom_prompt=prompt
            )
            
            # Parse response
            result = self._parse_response(response_text)
            
            if result:
                log_success("Đã trích xuất thông tin thành công!")
                log_info(f"Nhật thần: {result.nhat_than}")
                log_info(f"Nguyệt lệnh: {result.nguyet_lenh}")
                log_info(f"Thế: Hào {result.the_vi_tri}, Ứng: Hào {result.ung_vi_tri}")
                if result.tk_vi_tri:
                    log_info(f"TK: Hào {', '.join(map(str, result.tk_vi_tri))}")
                
                # Log hào động
                hao_dong_trai = [h.hao for h in result.que_trai if h.is_dong]
                hao_dong_phai = [h.hao for h in result.que_phai if h.is_dong]
                if hao_dong_trai:
                    log_info(f"Hào động quẻ trái: {', '.join(map(str, hao_dong_trai))}")
                if hao_dong_phai:
                    log_info(f"Hào động quẻ phải: {', '.join(map(str, hao_dong_phai))}")
                if not hao_dong_trai and not hao_dong_phai:
                    log_info("Không có hào động")
                
                # Log P.Thần cho các hào có giá trị
                phuc_than_info = [(h.hao, h.phuc_than) for h in result.que_trai if h.phuc_than]
                if phuc_than_info:
                    log_info("P.Thần (quẻ trái): " + ", ".join([f"Hào {h}: {pt}" for h, pt in phuc_than_info]))
            
            return result
            
        except Exception as e:
            log_error(f"Lỗi khi trích xuất thông tin: {e}")
            return None
    
    def _create_extraction_prompt(self) -> str:
        """
        Tạo prompt chuyên biệt để trích xuất thông tin I Ching.
        
        Returns:
            Prompt string
        """
        return """Bạn là chuyên gia phân tích ảnh kết quả Lục Hào (I Ching). Nhiệm vụ của bạn là trích xuất thông tin chi tiết từ ảnh kết quả divination.

Hãy phân tích ảnh và trích xuất các thông tin sau:

1. **Nhật thần**: Tìm trong phần thông tin chung, có dòng "Nhật thần:" (ví dụ: "Sửu-Thổ")

2. **Nguyệt lệnh**: Tìm trong phần thông tin chung, có dòng "Nguyệt lệnh:" (ví dụ: "Tý-Thủy")

3. **Quẻ bên trái**: Tìm bảng đầu tiên (bên trái) với các cột: "Hào", "T/Ư", "Lục Thân", "Can Chi", "P.Thần", "T.K"
   - Đọc 6 hào từ dưới lên (hào 1 ở dưới cùng, hào 6 ở trên cùng)
   - Với mỗi hào, trích xuất:
     * Hào số (1-6)
     * Lục Thân (Thế, Ứng, Phụ Mẫu, Tử Tôn, Thê Tài, Quan Quỷ, Huynh Đệ)
     * Can Chi (ví dụ: "Tý-Thủy", "Dần-Mộc")
     * P.Thần (Phục Thần): Từ cột "P.Thần", lấy giá trị. Nếu là "-" hoặc rỗng, để null. Nếu có giá trị (ví dụ: tên Can Chi), lưu vào phuc_than
   - Xác định vị trí Thế: tìm hào có "Thế" trong cột T/Ư
   - Xác định vị trí Ứng: tìm hào có "Ứng" trong cột T/Ư
   - Xác định vị trí TK: tìm các hào có "K" trong cột T.K
   - **Xác định hào động (RẤT QUAN TRỌNG - ĐỌC KỸ)**: 
     * BƯỚC 1: Trước tiên, hãy QUÉT TOÀN BỘ BẢNG và liệt kê TẤT CẢ các hào có dấu hiệu màu đỏ:
       - Quét từng hàng từ dưới lên (hào 1 đến hào 6)
       - Ghi chú lại: "Hào X có text đỏ" hoặc "Hào Y có vạch đỏ trong hexagram"
       - Ví dụ: "Tìm thấy: Hào 3 có text màu đỏ (Lục Thân và Can Chi), Hào 5 có text màu đỏ (Lục Thân và Can Chi)"
     * BƯỚC 2: Sau khi liệt kê xong, đánh dấu is_dong: true cho TẤT CẢ các hào đã liệt kê
     * Hào động được đánh dấu bằng MỘT TRONG CÁC CÁCH sau:
       - Có text màu đỏ trong các cột của hàng đó (ví dụ: Lục Thân màu đỏ, Can Chi màu đỏ, hoặc cả hai)
       - Được đánh dấu màu đỏ trong biểu đồ hexagram visual (các vạch màu đỏ)
     * VÍ DỤ CỤ THỂ:
       - Nếu thấy hàng thứ 3 từ dưới lên có "Thê Tài" và "Ngọ-Hỏa" màu đỏ → Hào 3 là hào động (is_dong: true)
       - Nếu thấy hàng thứ 5 từ dưới lên có "Huynh Đệ" và "Hợi-Thủy" màu đỏ → Hào 5 là hào động (is_dong: true)
       - Nếu một hàng KHÔNG có text đỏ, KHÔNG có vạch đỏ → Hào đó không động (is_dong: false)
     * QUAN TRỌNG: 
       - Hào số được đếm từ DƯỚI LÊN: Hào 1 = hàng dưới cùng, Hào 6 = hàng trên cùng
       - KHÔNG nhầm lẫn: cột T.K có "K" KHÔNG có nghĩa là hào động
       - CÓ THỂ CÓ NHIỀU HÀO ĐỘNG (ví dụ: cả Hào 3 và Hào 5 đều động)
       - Phải kiểm tra TẤT CẢ 6 hàng, không được bỏ sót
     * Đánh dấu is_dong: true cho các hào động, is_dong: false cho các hào không động

4. **Quẻ bên phải**: Tìm bảng thứ hai (bên phải) với các cột: "Lục Thân", "Can Chi", "Lục Thú", "T.K", "Hào"
   - Đọc 6 hào từ dưới lên
   - Với mỗi hào, trích xuất:
     * Hào số (1-6)
     * Lục Thân
     * Can Chi
     * Lục Thú (Thanh Long, Chu Tước, Câu Trần, Đằng Xà, Bạch Hổ, Huyền Vũ)
   - **LƯU Ý**: Không cần xác định hào động cho quẻ phải. Tất cả hào trong quẻ phải đều có is_dong: false

**QUAN TRỌNG**: Bạn PHẢI trả về kết quả dưới dạng JSON hợp lệ, không có markdown code block, không có text thừa. Chỉ trả về JSON thuần.

Format JSON:
{
  "nhat_than": "Sửu-Thổ",
  "nguyet_lenh": "Tý-Thủy",
  "que_trai": [
    {"hao": 1, "luc_than": "Thê Tài", "can_chi": "Tý-Thủy", "is_dong": false, "phuc_than": null},
    {"hao": 2, "luc_than": "Quan Quỷ", "can_chi": "Dần-Mộc", "is_dong": true, "phuc_than": "Giáp Tý"},
    {"hao": 3, "luc_than": "Huynh Đệ", "can_chi": "Thìn-Thổ", "is_dong": false, "phuc_than": null},
    {"hao": 4, "luc_than": "Phụ Mẫu", "can_chi": "Ngọ-Hỏa", "is_dong": false, "phuc_than": null},
    {"hao": 5, "luc_than": "Tử Tôn", "can_chi": "Thân-Kim", "is_dong": false, "phuc_than": null},
    {"hao": 6, "luc_than": "Huynh Đệ", "can_chi": "Tuất-Thổ", "is_dong": false, "phuc_than": null}
  ],
  "que_phai": [
    {"hao": 1, "luc_than": "Thê Tài", "can_chi": "Tý-Thủy", "luc_thu": "Thanh Long", "is_dong": false},
    {"hao": 2, "luc_than": "Quan Quỷ", "can_chi": "Dần-Mộc", "luc_thu": "Chu Tước", "is_dong": false},
    {"hao": 3, "luc_than": "Huynh Đệ", "can_chi": "Thìn-Thổ", "luc_thu": "Câu Trần", "is_dong": false},
    {"hao": 4, "luc_than": "Phụ Mẫu", "can_chi": "Ngọ-Hỏa", "luc_thu": "Đằng Xà", "is_dong": false},
    {"hao": 5, "luc_than": "Tử Tôn", "can_chi": "Thân-Kim", "luc_thu": "Bạch Hổ", "is_dong": false},
    {"hao": 6, "luc_than": "Huynh Đệ", "can_chi": "Tuất-Thổ", "luc_thu": "Huyền Vũ", "is_dong": false}
  ],
  "the_vi_tri": 4,
  "ung_vi_tri": 1,
  "tk_vi_tri": [6]
}

Lưu ý:
- Hào số được đếm từ dưới lên (hào 1 = dưới cùng, hào 6 = trên cùng)
- Nếu không tìm thấy TK, để "tk_vi_tri": null
- Đảm bảo tất cả 6 hào được liệt kê đầy đủ cho cả quẻ trái và quẻ phải
- Chỉ trả về JSON, không có text giải thích thêm"""
    
    def _parse_response(self, response_text: str) -> Optional[IChingResult]:
        """
        Parse response từ Gemini thành IChingResult object.
        
        Args:
            response_text: Text response từ Gemini
            
        Returns:
            IChingResult object nếu thành công, None nếu thất bại
        """
        try:
            # Tìm JSON trong response (có thể có markdown code block)
            json_text = self._extract_json_from_response(response_text)
            
            if not json_text:
                log_warn("Không tìm thấy JSON trong response")
                return None
            
            # Parse JSON
            data = json.loads(json_text)
            
            # Validate và tạo IChingResult
            result = IChingResult.from_dict(data)
            
            # Validate số lượng hào
            if len(result.que_trai) != 6:
                log_warn(f"Quẻ trái có {len(result.que_trai)} hào, mong đợi 6 hào")
            
            if len(result.que_phai) != 6:
                log_warn(f"Quẻ phải có {len(result.que_phai)} hào, mong đợi 6 hào")
            
            # Kiểm tra tính nhất quán hào động TRƯỚC KHI đồng bộ (để phát hiện lỗi từ Gemini API)
            hao_dong_trai_original = [h.hao for h in result.que_trai if h.is_dong]
            hao_dong_phai_original = [h.hao for h in result.que_phai if h.is_dong]
            if set(hao_dong_trai_original) != set(hao_dong_phai_original):
                log_warn(f"Phát hiện hào động không nhất quán từ Gemini API - Quẻ trái: {hao_dong_trai_original}, Quẻ phải: {hao_dong_phai_original}. "
                        f"Sẽ tự động đồng bộ để quẻ phải tương ứng với quẻ trái.")
            
            # Đồng bộ hào động: Hào động ở quẻ phải tương ứng với quẻ trái
            self._sync_hao_dong(result)
            
            # Validation logic: Kiểm tra tính hợp lý của kết quả
            self._validate_result(result)
            
            return result
            
        except json.JSONDecodeError as e:
            log_error(f"Lỗi parse JSON: {e}")
            log_info(f"Response text: {response_text[:500]}...")
            return None
        except Exception as e:
            log_error(f"Lỗi khi parse response: {e}")
            return None
    
    def _sync_hao_dong(self, result: IChingResult) -> None:
        """
        Đồng bộ hào động từ quẻ trái sang quẻ phải.
        Hào động ở quẻ phải tương ứng với quẻ trái (cùng số hào).
        
        Args:
            result: IChingResult object cần đồng bộ
        """
        # Tìm tất cả hào động ở quẻ trái
        hao_dong_trai = [h.hao for h in result.que_trai if h.is_dong]
        
        if not hao_dong_trai:
            # Không có hào động ở quẻ trái, đảm bảo tất cả hào ở quẻ phải đều không động
            for hao in result.que_phai:
                hao.is_dong = False
            return
        
        # Đồng bộ: Set is_dong: true cho các hào tương ứng ở quẻ phải
        for hao in result.que_phai:
            if hao.hao in hao_dong_trai:
                hao.is_dong = True
            else:
                hao.is_dong = False
        
        log_info(f"Đã đồng bộ hào động: Quẻ trái có Hào {', '.join(map(str, hao_dong_trai))} động → Quẻ phải cũng có các hào này động")
    
    def _validate_result(self, result: IChingResult) -> None:
        """
        Kiểm tra tính hợp lý của kết quả trích xuất.
        
        Args:
            result: IChingResult object cần validate
        """
        # Kiểm tra hào số hợp lệ (1-6)
        for hao in result.que_trai:
            if hao.hao < 1 or hao.hao > 6:
                log_warn(f"Quẻ trái: Hào số không hợp lệ: {hao.hao} (phải từ 1-6)")
        
        for hao in result.que_phai:
            if hao.hao < 1 or hao.hao > 6:
                log_warn(f"Quẻ phải: Hào số không hợp lệ: {hao.hao} (phải từ 1-6)")
        
        # Kiểm tra Thế và Ứng hợp lệ
        if result.the_vi_tri is not None:
            if result.the_vi_tri < 1 or result.the_vi_tri > 6:
                log_warn(f"Vị trí Thế không hợp lệ: {result.the_vi_tri} (phải từ 1-6)")
        
        if result.ung_vi_tri is not None:
            if result.ung_vi_tri < 1 or result.ung_vi_tri > 6:
                log_warn(f"Vị trí Ứng không hợp lệ: {result.ung_vi_tri} (phải từ 1-6)")
        
        # Kiểm tra Thế và Ứng không trùng nhau (chỉ khi cả hai đều không None)
        if result.the_vi_tri is not None and result.ung_vi_tri is not None:
            if result.the_vi_tri == result.ung_vi_tri:
                log_warn(f"Thế và Ứng trùng nhau ở Hào {result.the_vi_tri} (thường không xảy ra)")
        
        # Kiểm tra TK hợp lệ
        if result.tk_vi_tri:
            for tk in result.tk_vi_tri:
                if tk < 1 or tk > 6:
                    log_warn(f"Vị trí TK không hợp lệ: {tk} (phải từ 1-6)")
        
        # Kiểm tra hào động: đếm số lượng và cảnh báo nếu có vấn đề
        hao_dong_trai = [h.hao for h in result.que_trai if h.is_dong]
        hao_dong_phai = [h.hao for h in result.que_phai if h.is_dong]
        
        # Cảnh báo nếu không có hào động nào (có thể đúng, nhưng cần kiểm tra)
        if not hao_dong_trai and not hao_dong_phai:
            log_info("Lưu ý: Không có hào động nào được phát hiện. Hãy kiểm tra ảnh gốc để xác nhận.")
        
        # Kiểm tra số lượng hào động hợp lý (thường từ 0-6)
        # Dùng set union để đếm số hào động duy nhất (tránh double-count vì _sync_hao_dong đã đồng bộ)
        unique_hao_dong = set(hao_dong_trai) | set(hao_dong_phai)
        total_dong = len(unique_hao_dong)
        if total_dong > 6:
            log_warn(f"Số hào động duy nhất ({total_dong}) vượt quá 6. Có thể có lỗi trong trích xuất.")
        
        # Kiểm tra hào số trùng lặp
        hao_numbers_trai = [h.hao for h in result.que_trai]
        hao_numbers_phai = [h.hao for h in result.que_phai]
        
        if len(hao_numbers_trai) != len(set(hao_numbers_trai)):
            log_warn("Quẻ trái: Có hào số trùng lặp!")
        
        if len(hao_numbers_phai) != len(set(hao_numbers_phai)):
            log_warn("Quẻ phải: Có hào số trùng lặp!")
    
    def _extract_json_from_response(self, response_text: str) -> Optional[str]:
        """
        Trích xuất JSON từ response text (có thể có markdown code block).
        
        Args:
            response_text: Response text từ Gemini
            
        Returns:
            JSON string nếu tìm thấy, None nếu không
        """
        # Loại bỏ markdown code block nếu có
        response_text = response_text.strip()
        
        # Tìm JSON object trong text
        # Thử tìm từ { đến }
        start_idx = response_text.find('{')
        if start_idx == -1:
            return None
        
        # Tìm closing brace tương ứng
        brace_count = 0
        end_idx = start_idx
        
        for i in range(start_idx, len(response_text)):
            if response_text[i] == '{':
                brace_count += 1
            elif response_text[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_idx = i + 1
                    break
        
        if brace_count != 0:
            log_warn("Không tìm thấy JSON object đầy đủ")
            return None
        
        json_text = response_text[start_idx:end_idx]
        
        # Loại bỏ markdown code block markers nếu có
        json_text = re.sub(r'^```json\s*', '', json_text, flags=re.IGNORECASE)
        json_text = re.sub(r'^```\s*', '', json_text)
        json_text = re.sub(r'\s*```$', '', json_text)
        json_text = json_text.strip()
        
        return json_text

