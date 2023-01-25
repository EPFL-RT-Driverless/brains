# Copyright (c) 2023. Tudor Oancea EPFL Racing Team Driverless
import numpy as np
import pytest

from brains_python.control import IHMAcadosParams


def test_stanley_params_init():
    try:
        IHMAcadosParams(
            sampling_time=0.05,
            horizon_size=30,
            Q=np.diag([300.0, 300.0, 60.0, 35.0]),
            R=np.array([1.0, 60.0]),
            R_tilde=np.array([100.0, 100.0]),
            Zl=np.array([100.0]),
            Zu=np.array([10.0]),
            zl=np.array([100.0]),
            zu=np.array([10.0]),
        )
    except Exception as e:
        pytest.fail(e)
