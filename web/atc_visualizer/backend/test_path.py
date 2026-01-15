from pathlib import Path
import sys

p = Path(__file__).resolve()
print("[DEBUG] File:", p)
print("[DEBUG] Parent (x1):", p.parent)
print("[DEBUG] Parent (x2):", p.parent.parent)
print("[DEBUG] Parent (x3):", p.parent.parent.parent)
print("[DEBUG] Parent (x4):", p.parent.parent.parent.parent)

project_root = p.parent.parent.parent
print(f"\n[DEBUG] Project Root: {project_root}")
print(f'[DEBUG] Modules exists: {(project_root / "modules").exists()}')
print(f"[DEBUG] Python path: {sys.path[:3]}")
