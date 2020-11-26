import numpy as np


def batchnorm(x, gamma, beta, running_mean=None, running_var=None, momentum=0.7, eps=1e-5, mode='train'):
    n, d = x.shape
    cache = None
    if mode == 'train':
        sample_mean = np.mean(x, axis=0)
        sample_var = np.var(x, axis=0)
        x_hat = (x - sample_mean) / np.sqrt(sample_var + eps)
        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var
        cache = (x, x_hat, sample_mean, sample_var, gamma, eps)
    else:
        x_hat = (x - running_mean) / np.sqrt(running_var + eps)
    out = x_hat * gamma + beta
    bn_params = dict(mode=mode, momentum=0.7, running_mean=running_mean, running_var=running_var,
                     eps=eps)
    return out, cache, bn_params


def batchnorm_backward(dout, cache):
    n, d = x.shape
    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(dout * x_hat, axis=0)
    dx_hat = dout * gamma
    temp = np.power(var + eps, -0.5)
    dvar = -0.5 * np.sum(dx_hat * (x - mean) * temp ** 3, axis=0)
    dmean = np.sum(-1 * dx_hat * temp, axis=0) - 2 * dvar * np.sum((x - mean) / N, axis=0)
    dx = dx_hat * temp + dvar * 2 * (x - mean) / N + dmean / N
    return dx, dgamma, dbeta

