import numpy as np


def check_accuracy(model, number=1000):
    """Check model accuracy using loss.

    Args:
        model (object): Model to be checked.
        number (int): number of grid. Default: 1000.

    Returns:
        tuple[float, dict]: loss and dict for drawing.
    """

    x1 = np.linspace(-5, 5, num=number)
    x2 = np.linspace(-5, 5, num=number)
    x1, x2 = np.meshgrid(x1, x2)
    X = []
    for i, x in enumerate(x1):
        X.append(
            np.hstack(
                [x.reshape(x.shape[0], 1), x2[i].reshape(x2[i].shape[0], 1)]))
    X = np.vstack(X)
    y = (np.sin(X[:, 0]) - np.cos(X[:, 1]))
    y_ = model.loss(X).reshape(y.shape[0], )
    loss = np.mean((y_.reshape(X.shape[0], 1) - y.reshape(X.shape[0], 1))**2)

    Y = np.array([y_[i * number:(i + 1) * number] for i in range(number)])
    grid_dict = dict(x=x1, y=x2, z=Y)

    return loss, grid_dict
