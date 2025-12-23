"""
Web automation for I Ching form filling.
"""

import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import traceback
import json

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select, WebDriverWait

try:
    from webdriver_manager.chrome import ChromeDriverManager
    HAS_WEBDRIVER_MANAGER = True
except ImportError:
    HAS_WEBDRIVER_MANAGER = False

from config.iching import (
    CLICK_DELAY,
    IMAGES_DIR,
    ICHING_URL,
    NUM_LINES,
    SCROLL_DELAY_DEFAULT,
    SCROLL_DELAY_LONG,
    SELECT_DELAY,
    SUBMIT_DELAY,
    WAIT_TIMEOUT,
)
from modules.common.ui.logging import log_error, log_info, log_success, log_warn
from modules.iching.core.data_models import IChingResult
from modules.iching.core.result_extractor import IChingResultExtractor


def wait_for_page_load(driver: webdriver.Chrome, wait_timeout: int = 30) -> None:
    """
    Đợi trang web load xong hoàn toàn, bao gồm cả ảnh.
    
    Args:
        driver: Chrome WebDriver instance
        wait_timeout: Thời gian chờ tối đa (giây)
    """
    try:
        log_info("Đang chờ trang web load xong...")
        
        # Đợi document ready state
        wait = WebDriverWait(driver, wait_timeout)
        wait.until(lambda d: d.execute_script("return document.readyState") == "complete")
        
        # Đợi thêm một chút để đảm bảo JavaScript đã chạy xong
        time.sleep(2)
        
        # Scroll xuống để load lazy images
        log_info("Đang scroll để load hết ảnh...")
        last_height = driver.execute_script("return document.body.scrollHeight")
        while True:
            # Scroll xuống cuối trang
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
            
            # Tính chiều cao mới
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height
        
        # Scroll lên đầu trang
        driver.execute_script("window.scrollTo(0, 0);")
        time.sleep(2)
        
        # Đợi tất cả ảnh load xong
        # Wait for all images to be complete using polling
        max_wait = 30
        start_time = time.time()
        while time.time() - start_time < max_wait:
            all_complete = driver.execute_script("""
                var images = document.getElementsByTagName('img');
                for (var i = 0; i < images.length; i++) {
                    if (!images[i].complete) {
                        return false;
                    }
                }
                return true;
            """)
            if all_complete:
                break
            time.sleep(0.5)
        time.sleep(3)  # Đợi thêm một chút để đảm bảo
        
        log_success("Trang web đã load xong hoàn toàn!")
        
    except Exception as e:
        log_warn(f"Không thể đợi trang load hoàn toàn: {e}, tiếp tục chụp screenshot...")


def save_result_screenshot(
    driver: webdriver.Chrome, wait_timeout: int = 30
) -> Tuple[Optional[str], Optional[IChingResult]]:
    """
    Chờ trang web load xong, lưu screenshot và trích xuất thông tin.
    
    Args:
        driver: Chrome WebDriver instance
        wait_timeout: Thời gian chờ tối đa (giây)
        
    Returns:
        Tuple (screenshot_path, iching_result):
        - screenshot_path: Đường dẫn đến file ảnh đã lưu, hoặc None nếu thất bại
        - iching_result: IChingResult object nếu trích xuất thành công, None nếu thất bại
    """
    screenshot_path = None
    iching_result = None
    
    try:
        # Tạo thư mục images nếu chưa có (sử dụng project-configured path)
        images_dir = IMAGES_DIR
        images_dir.mkdir(exist_ok=True, parents=True)
        
        # Đợi trang load xong
        wait_for_page_load(driver, wait_timeout)
        
        # Tạo tên file với timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"iching_result_{timestamp}.png"
        filepath = images_dir / filename
        
        # Chụp screenshot full page
        log_info(f"Đang chụp screenshot và lưu vào: {filepath}")
        
        # Lấy kích thước trang
        total_width = driver.execute_script("return document.body.scrollWidth")
        total_height = driver.execute_script("return document.body.scrollHeight")
        
        # Set window size để chụp full page
        driver.set_window_size(total_width, total_height)
        time.sleep(1)
        
        # Chụp screenshot
        driver.save_screenshot(str(filepath))
        
        if filepath.exists():
            file_size = filepath.stat().st_size
            log_success(f"Đã lưu screenshot thành công: {filepath} ({file_size} bytes)")
            screenshot_path = str(filepath)
            
            # Trích xuất thông tin từ ảnh
            try:
                log_info("Bắt đầu trích xuất thông tin từ ảnh...")
                extractor = IChingResultExtractor()
                iching_result = extractor.extract_from_image(screenshot_path)
                
                if iching_result:
                    log_success("Đã trích xuất thông tin thành công!")
                else:
                    log_warn("Không thể trích xuất thông tin từ ảnh")
            except Exception as e:
                log_error(f"Lỗi khi trích xuất thông tin: {e}")
                # Không fail toàn bộ process nếu extraction thất bại
        else:
            log_error("Không thể tạo file screenshot")
            
    except Exception as e:
        log_error(f"Lỗi khi lưu screenshot: {e}")
        traceback.print_exc()
    
    return screenshot_path, iching_result


def scroll_into_view(
    driver: webdriver.Chrome, element, delay: float = SCROLL_DELAY_DEFAULT
) -> None:
    """Scroll element vào giữa màn hình."""
    try:
        driver.execute_script(
            "arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});",
            element,
        )
        time.sleep(delay)
    except Exception as e:
        log_warn(f"Không thể scroll element: {e}")


def click_with_fallback(driver: webdriver.Chrome, element) -> None:
    """Click element, fallback sang JavaScript nếu cần."""
    try:
        element.click()
    except Exception as e:
        try:
            driver.execute_script("arguments[0].click();", element)
        except Exception as js_e:
            raise RuntimeError(
                f"Không thể click element: normal click failed ({e}), "
                f"JavaScript click failed ({js_e})"
            ) from js_e


def ensure_checkbox_checked(
    driver: webdriver.Chrome, wait: WebDriverWait, checkbox_id: str
) -> None:
    """Đảm bảo checkbox được tick."""
    try:
        checkbox = wait.until(EC.presence_of_element_located((By.ID, checkbox_id)))
        if checkbox.is_selected():
            log_info(f"Checkbox {checkbox_id} đã được tick sẵn.")
            return

        log_info(f"Đang tick vào checkbox {checkbox_id}...")
        scroll_into_view(driver, checkbox, delay=SCROLL_DELAY_LONG)
        wait.until(EC.element_to_be_clickable((By.ID, checkbox_id)))
        click_with_fallback(driver, checkbox)
        log_success(f"Đã tick vào checkbox {checkbox_id} thành công!")
    except Exception as exc:
        log_warn(f"Lỗi khi click checkbox, thử dùng JavaScript: {exc}")
        try:
            driver.execute_script(f"document.getElementById('{checkbox_id}').click();")
            log_success(f"Đã tick vào checkbox {checkbox_id} bằng JavaScript!")
        except Exception as js_exc:
            raise RuntimeError(
                f"Không thể tick checkbox {checkbox_id}: {exc}, "
                f"JavaScript fallback cũng thất bại: {js_exc}"
            ) from js_exc


def clear_all_haodong(driver: webdriver.Chrome) -> None:
    """Bỏ chọn toàn bộ checkbox HaoDong."""
    log_info("Đang kiểm tra và bỏ chọn các checkbox HaoDong...")
    for line_num in range(1, NUM_LINES + 1):
        checkbox_id = f"HaoDong{line_num}"
        try:
            checkbox = driver.find_element(By.ID, checkbox_id)
            if checkbox.is_selected():
                scroll_into_view(driver, checkbox, delay=CLICK_DELAY)
                click_with_fallback(driver, checkbox)
                log_info(f"  Đã bỏ chọn checkbox {checkbox_id}")
            else:
                log_info(f"  Checkbox {checkbox_id} chưa được chọn, bỏ qua")
        except Exception as exc:
            log_warn(f"  Không tìm thấy checkbox {checkbox_id}: {exc}")

    log_success("Đã hoàn thành việc kiểm tra và bỏ chọn các checkbox HaoDong.")


def handle_single_line(
    driver: webdriver.Chrome,
    wait: WebDriverWait,
    info: Dict[str, bool],
    line_num: int,
) -> None:
    """Xử lý một vạch trong form."""
    if line_num < 1 or line_num > NUM_LINES:
        raise ValueError(f"line_num phải trong khoảng 1-{NUM_LINES}, nhận được: {line_num}")
    
    select_id = f"Hao{line_num}"
    log_info(f"Xử lý vạch {line_num}...")

    try:
        select_element = wait.until(EC.presence_of_element_located((By.ID, select_id)))
        select = Select(select_element)

        if info["is_solid"]:
            select.select_by_value("1")
            log_info(f"  Đã chọn vạch liền (———) cho {select_id}")
        else:
            select.select_by_value("0")
            log_info(f"  Đã chọn vạch đứt (—　—) cho {select_id}")

        time.sleep(SELECT_DELAY)

        if not info["is_red"]:
            log_info(f"  Vạch không phải màu đỏ, không tick HaoDong{line_num}")
            return

        checkbox_id = f"HaoDong{line_num}"
        checkbox = wait.until(EC.presence_of_element_located((By.ID, checkbox_id)))
        if checkbox.is_selected():
            log_info(f"  Checkbox {checkbox_id} đã được tick sẵn")
            return

        scroll_into_view(driver, checkbox)
        click_with_fallback(driver, checkbox)
        log_success(f"  Đã tick checkbox {checkbox_id} (vạch màu đỏ)")
    except Exception as e:
        raise RuntimeError(f"Lỗi khi xử lý vạch {line_num}: {e}") from e


def click_submit_button(driver: webdriver.Chrome, wait: WebDriverWait) -> None:
    """Click nút 'Lập quẻ'."""
    log_info("Đang tìm và click button 'Lập quẻ'...")
    try:
        submit_button = wait.until(
            EC.element_to_be_clickable((By.NAME, "submitsearch"))
        )
        scroll_into_view(driver, submit_button, delay=SCROLL_DELAY_LONG)
        click_with_fallback(driver, submit_button)
        time.sleep(SUBMIT_DELAY)
        log_success("Đã submit form!")
    except Exception as exc:
        log_warn(f"Lỗi khi click button 'Lập quẻ' bằng NAME: {exc}")
        try:
            submit_button = driver.find_element(
                By.XPATH, "//button[contains(text(), 'Lập quẻ')]"
            )
            scroll_into_view(driver, submit_button, delay=SCROLL_DELAY_LONG)
            click_with_fallback(driver, submit_button)
            time.sleep(SUBMIT_DELAY)
            log_success("Đã click button 'Lập quẻ' bằng XPath!")
        except Exception as fallback_exc:
            raise RuntimeError(
                f"Không thể tìm thấy button 'Lập quẻ': "
                f"NAME selector failed ({exc}), XPath selector failed ({fallback_exc})"
            ) from fallback_exc


def fill_web_form(
    line_info: List[Dict[str, bool]], 
    auto_close: bool = False,
    stop_on_error: bool = False
) -> None:
    """
    Mở trình duyệt, điền form và submit.
    
    Args:
        line_info: Danh sách 6 dict chứa thông tin vạch
        auto_close: Nếu True, tự động đóng trình duyệt sau khi submit
                   Nếu False, chờ user nhấn Enter
        stop_on_error: Nếu True, dừng xử lý và re-raise exception khi gặp lỗi
                      khi xử lý một vạch. Nếu False (mặc định), tiếp tục xử lý
                      các vạch còn lại khi gặp lỗi. Lưu ý: Khi stop_on_error=False,
                      việc tiếp tục xử lý có thể dẫn đến form submission không hoàn chỉnh
                      nếu một hoặc nhiều vạch không được điền thành công.
        
    Raises:
        ValueError: Nếu line_info không có đủ 6 phần tử
        RuntimeError: Nếu không thể khởi tạo driver hoặc xử lý form.
                     Khi stop_on_error=True, cũng re-raise exception từ handle_single_line
                     nếu xử lý một vạch thất bại.
    """
    if len(line_info) < NUM_LINES:
        raise ValueError(f"line_info phải có ít nhất {NUM_LINES} phần tử, nhận được {len(line_info)}")

    log_info(f"Đang mở trình duyệt với URL: {ICHING_URL}")

    driver: Optional[webdriver.Chrome] = None
    try:
        # Sử dụng webdriver-manager nếu có, nếu không thì dùng Chrome mặc định
        if HAS_WEBDRIVER_MANAGER:
            try:
                service = Service(ChromeDriverManager().install())
                driver = webdriver.Chrome(service=service)
                log_success("Đã sử dụng webdriver-manager để quản lý ChromeDriver.")
            except Exception as exc:
                log_warn(f"Lỗi khi sử dụng webdriver-manager: {exc}")
                log_info("Thử dùng Chrome mặc định...")
                driver = webdriver.Chrome()
        else:
            driver = webdriver.Chrome()

        if driver is None:
            raise RuntimeError("Không thể khởi tạo Chrome driver")

        driver.get(ICHING_URL)
        wait = WebDriverWait(driver, WAIT_TIMEOUT)

        ensure_checkbox_checked(driver, wait, "IsTietKhi")
        clear_all_haodong(driver)

        for line_num in range(1, NUM_LINES + 1):
            line_info_item = line_info[line_num - 1]
            try:
                handle_single_line(driver, wait, line_info_item, line_num)
            except Exception as exc:
                log_error(f"  Lỗi khi xử lý Hao{line_num}: {exc}")
                if stop_on_error:
                    # Re-raise exception để caller có thể quyết định xử lý
                    raise
                # Tiếp tục với các vạch khác thay vì dừng lại

        log_success("Đã hoàn thành việc điền form!")
        click_submit_button(driver, wait)
        
        # Chờ trang load xong, lưu screenshot và trích xuất thông tin
        screenshot_path, iching_result = save_result_screenshot(driver, wait_timeout=WAIT_TIMEOUT)
        if screenshot_path:
            log_info(f"Screenshot đã được lưu tại: {screenshot_path}")
        
        # Log thông tin đã trích xuất nếu có
        if iching_result:
            log_success("=== THÔNG TIN ĐÃ TRÍCH XUẤT ===")
            log_info(f"Nhật thần: {iching_result.nhat_than}")
            log_info(f"Nguyệt lệnh: {iching_result.nguyet_lenh}")
            log_info(f"Thế: Hào {iching_result.the_vi_tri}, Ứng: Hào {iching_result.ung_vi_tri}")
            if iching_result.tk_vi_tri:
                log_info(f"TK: Hào {', '.join(map(str, iching_result.tk_vi_tri))}")
            
            # Log hào động
            hao_dong_trai = [h.hao for h in iching_result.que_trai if h.is_dong]
            hao_dong_phai = [h.hao for h in iching_result.que_phai if h.is_dong]
            if hao_dong_trai:
                log_info(f"Hào động quẻ trái: {', '.join(map(str, hao_dong_trai))}")
            if hao_dong_phai:
                log_info(f"Hào động quẻ phải: {', '.join(map(str, hao_dong_phai))}")
            if not hao_dong_trai and not hao_dong_phai:
                log_info("Không có hào động")
            
            # Log P.Thần cho các hào có giá trị
            phuc_than_info = [(h.hao, h.phuc_than) for h in iching_result.que_trai if h.phuc_than]
            if phuc_than_info:
                log_info("P.Thần (quẻ trái): " + ", ".join([f"Hào {h}: {pt}" for h, pt in phuc_than_info]))
            
            log_info("=== KẾT THÚC THÔNG TIN TRÍCH XUẤT ===")
            
            # Lưu kết quả vào file JSON (optional)
            try:
                json_path = Path(screenshot_path).with_suffix('.json')  
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(iching_result.to_dict(), f, ensure_ascii=False, indent=2)
                log_success(f"Đã lưu kết quả trích xuất vào: {json_path}")
            except Exception as e:
                log_warn(f"Không thể lưu kết quả vào JSON: {e}")

        if not auto_close:
            log_info("Trình duyệt sẽ giữ mở. Nhấn Enter để đóng...")
            try:
                input()
            except (EOFError, KeyboardInterrupt):
                log_info("Đang đóng trình duyệt...")
    finally:
        if driver is not None:
            try:
                driver.quit()
            except Exception as e:
                log_warn(f"Lỗi khi đóng driver: {e}")

