import numpy as np


def normal_init(param, shape):
    """Normalization initialization."""
    return np.random.normal(*param, shape)


def unit_init(shape):
    """Unit initialization."""
    return np.ones(shape)


def zero_init(shape):
    """Zero initialization."""
    return np.zeros(shape)
