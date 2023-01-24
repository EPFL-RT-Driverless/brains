import numpy as np
import pytest

from brains_python.control import LQRParams


def test_lqr_params_init():
    try:
        LQRParams(
            sampling_time=0.01,
            horizon_size=10,
            Q=np.array([7.0e1, 7.0e1, 3.0e1, 2.0e1]),
            R=np.array([1.0e0, 4.51e0]),
            R_tilde=np.array([1.0e0, 1.0e0]),
        )
    except Exception as e:
        pytest.fail(e)
