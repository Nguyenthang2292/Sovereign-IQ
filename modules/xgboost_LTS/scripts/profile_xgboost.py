import cProfile
import pstats
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
# Script: modules/xgboost_LTS/scripts/profile_xgboost.py
# Root: ../../../
sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))

from config import MODEL_FEATURES
from modules.xgboost_LTS.core.model import train_and_predict


def generate_dummy_data(n_samples=1000):
    """Generate dummy data for profiling."""
    data = {
        "Target": np.random.randint(0, 3, n_samples),
        "close": np.random.uniform(100, 200, n_samples),
        # Add other potential required columns if necessary, mostly MODEL_FEATURES
    }
    for feature in MODEL_FEATURES:
        data[feature] = np.random.randn(n_samples)

    df = pd.DataFrame(data)
    # Ensure float32 if config requires it (handled in train_and_predict usually, but good to be safe)
    return df


def run_profile():
    """Run cProfile on train_and_predict."""
    # Define output path relative to this script
    # Script: modules/xgboost_LTS/scripts/profile_xgboost.py
    # Output: modules/xgboost_LTS/profiles/xgboost_profile.stats
    output_dir = Path(__file__).resolve().parent.parent / "profiles"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "xgboost_profile.stats"

    print("Generating dummy data...")
    df = generate_dummy_data(n_samples=5000)  # Enough samples to get meaningful stats

    print("Starting profiling...")
    profiler = cProfile.Profile()
    profiler.enable()

    try:
        # We assume train_and_predict handles everything given the DF
        train_and_predict(df, use_cache=False)
    except Exception as e:
        print(f"Error during profiling: {e}")
        return

    profiler.disable()

    print(f"Profiling complete. Saving stats to {output_file}")
    stats = pstats.Stats(profiler).sort_stats("cumulative")
    stats.dump_stats(str(output_file))
    stats.print_stats(30)


if __name__ == "__main__":
    run_profile()
