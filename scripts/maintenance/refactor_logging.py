import os
import re


def refactor_logging(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    original_content = content

    # Check if there is anything to refactor
    if "logger." not in content and "logging." not in content and "import logging" not in content:
        return False

    # 1. Replace imports
    # Handle import logging.handlers first
    content = re.sub(r"^import logging\.handlers\n", "", content, flags=re.MULTILINE)

    # Handle import logging
    import_replacement = (
        "from modules.common.ui.logging import log_info, log_error, log_warn, log_debug, log_success, log_system"
    )
    content = re.sub(r"^import logging\n", import_replacement + "\n", content, flags=re.MULTILINE)

    # 2. Remove logger initialization
    content = re.sub(r"^[ \t]*logger\s*=\s*logging\.getLogger\([^)]+\)[ \t]*\n", "", content, flags=re.MULTILINE)
    content = re.sub(r"^[ \t]*self\.logger\s*=\s*logging\.getLogger\([^)]+\)[ \t]*\n", "", content, flags=re.MULTILINE)

    # 3. Remove basicConfig if any
    content = re.sub(r"^[ \t]*logging\.basicConfig\([^)]+\)[ \t]*\n", "", content, flags=re.MULTILINE)

    # 4. Replace logger methods
    # Handle logger.exception separately to add exc_info=True
    # Simple case: logger.exception("msg")
    content = re.sub(r"logger\.exception\(([^)]+)\)", r"log_error(\1, exc_info=True)", content)
    content = re.sub(r"self\.logger\.exception\(([^)]+)\)", r"log_error(\1, exc_info=True)", content)

    # Handle standard logger methods
    content = re.sub(r"logger\.info\(", r"log_info(", content)
    content = re.sub(r"logger\.error\(", r"log_error(", content)
    content = re.sub(r"logger\.warning\(", r"log_warn(", content)
    content = re.sub(r"logger\.warn\(", r"log_warn(", content)
    content = re.sub(r"logger\.debug\(", r"log_debug(", content)

    # Handle self.logger methods
    content = re.sub(r"self\.logger\.info\(", r"log_info(", content)
    content = re.sub(r"self\.logger\.error\(", r"log_error(", content)
    content = re.sub(r"self\.logger\.warning\(", r"log_warn(", content)
    content = re.sub(r"self\.logger\.warn\(", r"log_warn(", content)
    content = re.sub(r"self\.logger\.debug\(", r"log_debug(", content)

    # Some remaining logging.info etc. if they imported logging directly and used it
    content = re.sub(r"logging\.info\(", r"log_info(", content)
    content = re.sub(r"logging\.error\(", r"log_error(", content)
    content = re.sub(r"logging\.warning\(", r"log_warn(", content)
    content = re.sub(r"logging\.warn\(", r"log_warn(", content)
    content = re.sub(r"logging\.debug\(", r"log_debug(", content)
    content = re.sub(r"logging\.exception\(([^)]+)\)", r"log_error(\1, exc_info=True)", content)

    if content != original_content:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Refactored {filepath}")
        return True
    return False


def main():
    target_dir = os.path.join(os.getcwd(), "modules", "auto_trade")
    excluded = [
        os.path.join(target_dir, "monitoring", "logger.py"),
        os.path.join(target_dir, "gui", "utils", "gui_log_handler.py"),
    ]

    count = 0
    for root, _, files in os.walk(target_dir):
        for file in files:
            if file.endswith(".py"):
                filepath = os.path.join(root, file)
                if filepath in excluded:
                    continue
                if refactor_logging(filepath):
                    count += 1

    print(f"Total files refactored: {count}")


if __name__ == "__main__":
    main()
