import sys
from pathlib import Path

# Add the src and tests folder to sys.path to be able to import src modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "tests"))
