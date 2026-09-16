"""Check the environment and pinned sources without downloading model weights."""
import importlib
from pathlib import Path
import subprocess
import sys
import torch


def main():
    print(f"Python: {sys.version.split()[0]}; executable: {sys.executable}")
    print(f"PyTorch: {torch.__version__}; CUDA: {torch.cuda.is_available()}")
    failed = sys.version_info < (3, 11)
    for name in ("scipy", "mapanything", "uniception", "torchvision", "cv2"):
        try:
            module = importlib.import_module(name)
            print(f"{name}: {getattr(module, '__version__', 'installed')}")
        except Exception as exc:
            failed = True
            print(f"{name}: unavailable ({exc})")
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(["git", "submodule", "status"], cwd=root, text=True, capture_output=True)
    print(result.stdout.strip())
    if result.returncode or any(line.startswith(("-", "+", "U")) for line in result.stdout.splitlines()):
        failed = True
        print("Initialize the pinned submodules: git submodule update --init --recursive")
    if not torch.cuda.is_available():
        print("CPU validation is available; full-backbone training needs suitable compute.")
    raise SystemExit(int(failed))


if __name__ == "__main__":
    main()
