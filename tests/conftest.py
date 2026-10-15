import sys
from pathlib import Path

# src/ is a flat module directory rather than an installed package, matching how
# the scripts import each other (`from dataset import ...`).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
