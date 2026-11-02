import random

import numpy as np
import torch

M = 2**32 - 1


def init_fn(worker_id: int) -> None:
    seed = torch.LongTensor(1).random_().item()
    seed = (seed + worker_id) % M
    np.random.seed(seed)
    random.seed(seed)
