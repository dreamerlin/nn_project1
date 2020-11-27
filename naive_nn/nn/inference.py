import numpy as np


def affine_forward(x, w, b):
    """Implementation of affine forward.

    Args：
        x (np.ndarray): (N, d1, d2, ..., dn) shape.
        w (np.ndarray): (D, W) shape.
        b (np.ndarray): (W, ) shape.

    Returns:
        out, cahed
    """
    n = x.shape[0]
    out = np.dot(x.reshape(n, -1), w) + b
    cache = (x, w)
    return out, cache


def affine_backward(dout, cache):
    """Backward of affine.

    Args:
        dout (np.ndarray): (N, W) shape.
        cache (tuple): x and w.

    Returns:
        Tuple of dx (N, D), dw (D, W), db (W, ).
    """
    x, w = cache
    n = x.shape[0]
    dx = np.dot(dout, w.T).reshape(x.shape)
    dw = np.dot(x.reshape(n, -1).T, dout)
    db = np.sum(dout, axis=0)
    return dx, dw, db
