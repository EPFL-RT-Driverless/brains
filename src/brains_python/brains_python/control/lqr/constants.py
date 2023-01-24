# Copyright (c) 2023. Tudor Oancea EPFL Racing Team Driverless
import numpy as np

__all__ = ["fsds_lqr_params"]

fsds_lqr_params = {
    "sampling_time": 0.01,
    "horizon_size": 10,
    "Q": np.array([7.0e1, 7.0e1, 3.0e1, 2.0e1]),
    "R": np.array([1.0e0, 4.51e0]),
    "R_tilde": np.array([1.0e0, 1.0e0]),
}
