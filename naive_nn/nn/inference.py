import numpy as np


def affine_forward(x, w, b):
    #  x.shape = (N, d1, d2, ..., dn)
    #  w.shape = (D, W)
    #  b.shape = (W,)
    n, d = x.shape[0], x.size // x.shape[0]
    out = np.dot(x.reshape(n, -1), w) + b
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    # dout.shape = (N, W)
    # dx.shape = (N, D)
    # dw.shape = (D, W)
    # db.shape = (W, )
    # 梯度反向传播
    x, w, _ = cache
    n = x.shape[0]
    dx = np.dot(dout, w.T).reshape(x.shape)
    dw = np.dot(x.reshape(n, -1).T, dout)
    db = np.sum(dout, axis=0)
    return dx, dw, db
