import numpy as np


def batchnorm(x,
              gamma,
              beta,
              eps=1e-5):
    """Implementation of batchnorm."""
    mean = np.mean(x, axis=0)
    var = np.var(x, axis=0)
    x_norm = (x - mean) / np.sqrt(var + eps)
    cache = dict(
        x=x,
        x_norm=x_norm,
        mean=mean,
        var=var,
        gamma=gamma,
        eps=eps)
    out = x_norm * gamma + beta
    return out, cache


def batchnorm_backward(dout, x, x_norm, mean, var, gamma, eps):
    """Backward of batchnorm."""
    n, d = x.shape
    dx_norm = dout * gamma
    temp = np.power(var + eps, -0.5)
    dvar = -0.5 * np.sum(dx_norm * (x - mean) * temp**3, axis=0)
    dmean = np.sum(
        -1 * dx_norm * temp, axis=0) - 2 * dvar * np.sum(
            (x - mean) / n, axis=0)
    dx = dx_norm * temp + dvar * 2 * (x - mean) / n + dmean / n
    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(dout * x_norm, axis=0)
    return dx, dgamma, dbeta
