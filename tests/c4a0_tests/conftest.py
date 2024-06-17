import random

import numpy as np
import torch


# Set the random seed for deterministic behavior
seed = 1337
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
