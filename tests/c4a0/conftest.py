from pathlib import Path
import random
import sys

import numpy as np
import torch

# Add the src and tests folder to sys.path to be able to import src modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "tests"))

# Set the random seed for deterministic behavior
seed = 1337
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
