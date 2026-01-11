
from pathlib import Path
import argparse
import sys
import traceback

"""
I Ching to Web - Main Entry Point.

Script để tạo hexagram I Ching và tự động điền form web.
"""



def find_project_root(start_path: Path) -> Path:
    """
    Tìm project root bằng cách đi lên từ start_path và tìm các marker files.

    Tìm kiếm các marker files: pyproject.toml, setup.py, .git, .project_root
    Trả về parent đầu tiên chứa bất kỳ marker nào.
    Nếu không tìm thấy, fallback về behavior cũ (4 levels up).

    Args:
        start_path: Đường dẫn bắt đầu (thường là Path(__file__))

    Returns:
        Path đến project root
    """
    marker_files = ["pyproject.toml", "setup.py", ".git", ".project_root"]

    # Đi lên từ parents và tìm marker
    for parent in start_path.resolve().parents:
        for marker in marker_files:
            marker_path = parent / marker
            if marker_path.exists():
                return parent

    # Fallback về behavior cũ nếu không tìm thấy marker
    return start_path.parent.parent.parent.parent


# Thêm project root vào sys.path để có thể import modules khi chạy trực tiếp
if __name__ == "__main__":
    # Lấy đường dẫn của file hiện tại
    current_file = Path(__file__).resolve()
    # Tìm project root bằng helper function
    project_root = find_project_root(current_file)
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from modules.common.ui.logging import log_error, log_info
from modules.iching.core.hexagram import prepare_hexagram
from modules.iching.core.web_automation import fill_web_form
from modules.iching.utils.helpers import clean_images_folder, ensure_utf8_stdout


def main(auto_close: bool = False) -> None:
    """
    Hàm main để chạy toàn bộ quy trình.

    Quy trình bao gồm:
    0. Làm sạch folder images
    1. Tạo hexagram ngẫu nhiên
    2. Tự động điền form web (bao gồm: lưu screenshot, trích xuất thông tin, và lưu kết quả JSON)

    Args:
        auto_close: Nếu True, tự động đóng trình duyệt sau khi submit
    """
    # Đảm bảo stdout sử dụng UTF-8
    ensure_utf8_stdout()

    log_info("=== BẮT ĐẦU QUY TRÌNH I CHING ===")

    try:
        log_info("Bước 0: Làm sạch folder images...")
        deleted_count = clean_images_folder()
        if deleted_count > 0:
            log_info(f"Đã xóa {deleted_count} file trong folder images")
        else:
            log_info("Folder images đã sạch, không có file nào cần xóa")
    except Exception as exc:
        log_error(f"LỖI khi làm sạch folder images: {exc}")
        traceback.print_exc()
        sys.exit(1)

    try:
        log_info("Bước 1: Tạo hexagram ngẫu nhiên...")
        line_info = prepare_hexagram()
        log_info("Đã tạo hexagram thành công!")
    except (ValueError, FileNotFoundError) as exc:
        log_error(f"LỖI khi tạo hexagram: {exc}")
        traceback.print_exc()
        sys.exit(1)
    except Exception as exc:
        log_error(f"LỖI không mong đợi khi tạo hexagram: {exc}")
        traceback.print_exc()
        sys.exit(1)

    try:
        log_info("Bước 2: Tự động điền form và trích xuất thông tin...")
        fill_web_form(line_info, auto_close=auto_close)
        log_info("=== HOÀN THÀNH QUY TRÌNH ===")
    except (ValueError, RuntimeError) as exc:
        log_error(f"Lỗi khi xử lý form: {exc}")
        log_info("Vui lòng đảm bảo đã cài đặt Selenium và ChromeDriver.")
        sys.exit(1)
    except Exception as exc:
        log_error(f"Lỗi không mong đợi khi xử lý form: {exc}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="I Ching to Web - Tạo hexagram và tự động điền form web")
    parser.add_argument("--auto-close", action="store_true", help="Tự động đóng trình duyệt sau khi submit form")
    args = parser.parse_args()
    main(auto_close=args.auto_close)
