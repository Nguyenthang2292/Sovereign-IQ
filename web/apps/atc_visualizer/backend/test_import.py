import sys
from pathlib import Path

current_file_path = Path(__file__).resolve()
backend_dir = current_file_path.parent
project_root = backend_dir.parent.parent.parent
project_root_absolute = project_root.resolve()

print(f"[DEBUG] Project root: {project_root_absolute}")
print(f"[DEBUG] Modules exists: {(project_root_absolute / 'modules').exists()}")

sys.path.insert(0, str(project_root_absolute))

try:
    from modules.adaptive_trend.core.analyzer import analyze_symbol
    from modules.adaptive_trend.utils.config import ATCConfig
    from modules.common.core.data_fetcher import DataFetcher
    from modules.common.core.exchange_manager import ExchangeManager

    print("✅ SUCCESS: All modules imported!")
except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback

    traceback.print_exc()
