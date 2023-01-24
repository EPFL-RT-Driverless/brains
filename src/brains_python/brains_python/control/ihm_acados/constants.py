# Copyright (c) 2023. Tudor Oancea EPFL Racing Team Driverless
import numpy as np

__all__ = ["fsds_ihm_acados_params"]

fsds_ihm_acados_params = {
    "sampling_time": 0.05,
    "horizon_size": 30,
    "Q": np.diag([300.0, 300.0, 60.0, 35.0]),
    "R": np.diag([1.0, 60.0]),
    "R_tilde": np.diag([100.0, 100.0]),
    "Zl": np.array([0.0]),
    "Zu": np.array([0.0]),
    "zl": np.array([0.0]),
    "zu": np.array([0.0]),
}
