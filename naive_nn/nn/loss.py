import numpy as np


def mse_loss(x, y):
    """Calculate MSE Loss."""
    n = x.shape[0]
    x = x.reshape((n, 1))
    y = y.reshape((n, 1))
    loss = np.mean((x - y)**2)
    dx = 2 * (x - y) / n
    return loss, dx
