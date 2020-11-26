import numpy as np


def normal_init(param, shape):
    return np.random.normal(*param, shape)


def random_init(shape):
    return np.random.rand(*shape)


def unit_init(shape):
    return np.ones(*shape)


def zero_init(shape):
    return np.zeros(*shape)
