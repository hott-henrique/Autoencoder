import random

import numpy as np
import torch


def ensure(lab_config: dict):
    seed = lab_config["Random"]["Seed"]

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(mode=True)

