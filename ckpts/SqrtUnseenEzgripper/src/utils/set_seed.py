import torch
import numpy as np
import random


def set_global_seed(seed=42):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
