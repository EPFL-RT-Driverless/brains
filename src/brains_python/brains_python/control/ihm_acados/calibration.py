# Copyright (c) 2023. Tudor Oancea EPFL Racing Team Driverless
import numpy as np

from brains_python.common import Mission
from brains_python.control.controller import ControllerParams
from brains_python.control.ihm_acados.constants import fsds_ihm_acados_params
from brains_python.control.ihm_acados.ihm_acados import IHMAcadosParams
from brains_python.control.utils import calibrate

__all__ = ["calibrate_ihm_acados"]


def calibrate_ihm_acados():
    def mapping(x: np.ndarray, cp: ControllerParams):
        cp.Q = np.array([x[0], x[1], x[2], x[3]])
        cp.R = np.array([x[4], x[5]])
        cp.R_tilde = np.array([x[6], x[7]])
        return cp

    calibrate(
        initial_controller_params=IHMAcadosParams(**fsds_ihm_acados_params),
        var_to_params_mapping=mapping,
        mission=Mission.SHORT_SKIDPAD,
        track_name="short_skidpad",
        csv_dump_file="ihm_acados_skidpad.csv",
        nvar=8,
        lb=1e-7 * np.ones(8),
        ub=1e3 * np.ones(8),
        f=lambda metrics: (np.mean(metrics[:, 0]) + np.std(metrics[:, 0])),
        load_previous_results=False,
    )


if __name__ == "__main__":
    calibrate_ihm_acados()
