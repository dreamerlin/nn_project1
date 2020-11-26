import matplotlib.pyplot as plt
import numpy as np


def draw3d(x, y, z):
    plt.figure()
    ax3 = plt.axes(projection='3d')
    ax3.plot_surface(x, y, z, cmap='rainbow')
    plt.show()
