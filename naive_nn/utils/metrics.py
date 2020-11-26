import numpy as np
from .vis import draw3d


def check_accuracy(model):
    number = 1000
    X1 = np.linspace(-5, 5, num=number)
    X2 = np.linspace(-5, 5, num=number)
    x1, x2 = np.meshgrid(X1, X2)
    X = []
    for i, x in enumerate(x1):
        X.append(np.hstack([x.reshape(x.shape[0], 1), x2[i].reshape(x2[i].shape[0], 1)]))
    X = np.vstack(X)
    y = (np.sin(X[:, 0]) - np.cos(X[:, 1]))
    y_ = model.loss(X).reshape(y.shape[0],)
    loss = np.mean((y_.reshape(X.shape[0], 1) - y.reshape(X.shape[0], 1))**2)

    # 画目标函数图
    Y = np.array([y_[i * number:(i + 1) * number] for i in range(number)])
    draw3d(x1, x2, Y)
    return loss