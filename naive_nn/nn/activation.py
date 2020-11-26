import numpy as np

from .inference import affine_backward, affine_forward


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


def relu_bachward(dout, cache):
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


def affine_relu(x, w, b):
    """Implementation of affine ReLU."""
    out, affine_cache = affine_forward(x, w, b)
    out, relu_cache = relu(out)
    cache = (affine_cache, relu_cache)
    return out, cache


def affine_relu_backward(dout, cache):
    """Backward of affine ReLU."""
    affine_cache, relu_cache = cache
    dout = relu_bachward(dout, relu_cache)
    dx, dw, db = affine_backward(dout, affine_cache)
    return dx, dw, db


def affine_leaky_relu(x, w, b, leaky_ratio):
    """Implementation of affine leaky ReLU."""
    out, affine_cache = affine_forward(x, w, b)
    out, relu_cache = leaky_relu(out, leaky_ratio)
    cache = (affine_cache, relu_cache)
    return out, cache


def affine_leaky_relu_backward(dout, cache):
    """Backward of affine leaky ReLU."""
    affine_cache, relu_cache = cache
    dout = leaky_relu_backward(dout, relu_cache)
    dx, dw, db = affine_backward(dout, affine_cache)
    return dx, dw, db
