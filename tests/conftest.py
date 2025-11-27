import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Suppress warnings from pytorch_forecasting about non-writable NumPy arrays
warnings.filterwarnings("ignore", message=".*The given NumPy array is not writable.*")
warnings.filterwarnings("ignore", category=UserWarning, module="pytorch_forecasting")

