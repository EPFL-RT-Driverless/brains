import numpy as np

__all__ = ["wrapToPi"]


def wrapToPi(x):
    return np.mod(x + np.pi, 2 * np.pi) - np.pi
