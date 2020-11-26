import matplotlib.pyplot as plt
import numpy as np


def draw2d(y):
    """Draw 2d figure.

    Args:
        y (list): y values.
    """
    x = np.arange(len(y))
    plt.plot(x, y, '-')
    plt.show()


def draw3d(x, y, z):
    """Draw 3d figure.

    Args:
        x (list): x values.
        y (list): y values.
        z (list): z values.
    """
    plt.figure()
    ax3 = plt.axes(projection='3d')
    ax3.plot_surface(x, y, z, cmap='rainbow')
    plt.show()
