# Copyright (c) 2023. Tudor Oancea EPFL Racing Team Driverless
import numpy as np

from brains_python.common import Mission
from brains_python.control.controller import ControllerParams
from brains_python.control.stanley.constants import stanley_mercury_params
from brains_python.control.stanley.stanley import StanleyParams
from brains_python.control.utils import calibrate

__all__ = ["calibrate_stanley"]


def calibrate_stanley():
    def mapping(x: np.ndarray, cp: ControllerParams):
        cp.k_P = x[0]
        cp.k_I = x[1]
        cp.k_psi = x[2]
        cp.k_e = x[3]
        cp.k_s = x[4]
        return cp

    calibrate(
        var_to_params_mapping=mapping,
        mission=Mission.SHORT_SKIDPAD,
        track_name="short_skidpad",
        csv_dump_file="stanley_skidpad.csv",
        nvar=5,
        lb=1e-7 * np.ones(5),
        f=lambda metrics: (
            np.mean(metrics[:, 0])
            + np.std(metrics[:, 0])
            + np.mean(metrics[:, 1])
            + np.std(metrics[:, 1])
            + np.mean(metrics[:, 2])
            + np.std(metrics[:, 2])
        ),
        ub=np.array([1.3, 1.3, 3.0, 10.0, 10.0]),
        initial_controller_params=StanleyParams(**stanley_mercury_params),
        load_previous_results=False,
    )


if __name__ == "__main__":
    calibrate_stanley()
