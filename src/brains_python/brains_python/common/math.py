import numpy as np

__all__ = ["wrapToPi"]


def wrapToPi(x):
    return np.mod(x + np.pi, 2 * np.pi) - np.pi


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x) - np.pi / 2
    return rho, phi
