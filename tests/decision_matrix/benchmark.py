import os
import random
import sys
import time

# Add project root to path
sys.path.append(os.getcwd())

from modules.decision_matrix.core.random_forest_core import RandomForestCore
from modules.decision_matrix.utils.training_data import TrainingDataStorage


def run_benchmark():
    print("Preparing benchmark data...")
    training_length = 850
    core = RandomForestCore(training_length=training_length)
    storage = TrainingDataStorage(training_length=training_length)

    # Generate random training data
    for _ in range(training_length):
        x1 = random.uniform(0, 100)
        x2 = random.uniform(0, 100)
        label = random.randint(0, 1)
        storage.add_sample(x1, x2, label)

    x1_matrix = storage.get_x1_matrix()
    x2_matrix = storage.get_x2_matrix()

    iterations = 100
    start_time = time.time()

    print(f"Running classification {iterations} times...")
    for _ in range(iterations):
        core.classify(
            x1_matrix=x1_matrix,
            x2_matrix=x2_matrix,
            current_x1=50.0,
            current_x2=50.0,
            x1_threshold=5.0,
            x2_threshold=5.0,
        )

    end_time = time.time()
    total_time = end_time - start_time
    avg_time = total_time / iterations

    print(f"Total time: {total_time:.4f}s")
    print(f"Average time per call: {avg_time:.6f}s")


if __name__ == "__main__":
    run_benchmark()
