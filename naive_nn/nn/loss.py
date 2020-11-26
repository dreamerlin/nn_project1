import numpy as np


def mse_loss(x, y):
    """Calculate MSE Loss"""
    n = x.shape[0]
    x = x.reshape((n, 1))
    y = y.reshape((n, 1))
    loss = np.sum((x - y)**2) / n
    dx = 2 * (x - y) / n
    return loss, dx
