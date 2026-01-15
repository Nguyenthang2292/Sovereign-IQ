from pathlib import Path
import sys

print("=" * 60)
print("PATH DEBUGGING")
print("=" * 60)

current_file = Path(__file__).resolve()
print(f"\n1. Current file: {current_file}")

backend = current_file.parent
print(f"2. Backend dir: {backend}")

atc_visualizer = backend.parent
print(f"3. ATC Visualizer dir: {atc_visualizer}")

web = atc_visualizer.parent
print(f"4. Web dir: {web}")

crypto = web.parent
print(f"5. Crypto-probability dir: {crypto}")

i_ching = crypto.parent
print(f"6. i-ching dir: {i_ching}")

print(f"\nDesired project root: {crypto}")
print(f"Modules path: {crypto / 'modules'}")
print(f"Modules exists: {(crypto / 'modules').exists()}")

print(f"\nPython working dir: {Path.cwd()}")
print(f"Python path[0]: {sys.path[0]}")
print("=" * 60)
