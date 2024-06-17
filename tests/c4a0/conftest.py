import os
import random
import sys

import numpy as np
import torch

# Add the src and tests folder to sys.path to be able to import src modules
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, os.path.join(root_dir, "src"))
with open("./foo.txt", "w") as f:
    f.write(f"{sys.path}")

# Set the random seed for deterministic behavior
seed = 1337
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
