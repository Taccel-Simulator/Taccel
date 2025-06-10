import random
import numpy as np
import torch

def set_seed(seed: int=None):
    seed = random.randint(0, 2 ** 32 - 1) if seed is None else seed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)