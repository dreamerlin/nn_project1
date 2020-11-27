import numpy as np


def sigmoid(x):
    """Implementation of sigmoid."""
    return 1 / (1 + np.exp(-x)), x


def sigmoid_backward(dout, save_x):
    """Backward of sigmoid."""
    dw = np.exp(save_x) / (1 + np.exp(save_x))**2
    dx = dout * dw
    return dx


def relu(x):
    """ReLU activation."""
    out = np.where(x >= 0, x, 0)
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """Backward of ReLU."""
    x = cache
    dx = dout * (x > 0)
    return dx


def leaky_relu(x, leaky_ratio):
    """Implementation of leaky ReLU."""
    out = np.where(x >= 0, x, x * leaky_ratio)
    cache = (x, leaky_ratio)
    return out, cache


def leaky_relu_backward(dout, cache):
    """Backward of leaky ReLU."""
    x, leaky_ratio = cache
    dx = dout.copy() * np.where(x >= 0, 1, leaky_ratio)
    return dx
