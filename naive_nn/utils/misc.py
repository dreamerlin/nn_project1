import numpy as np
import random
import os


def set_random_seed(seed):
    """set random seed for reimplementation."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
