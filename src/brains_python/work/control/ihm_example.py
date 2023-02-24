import pytest

from brains_python.control import (
    fsds_car_params,
    CarParams,
    IHMAcadosParams,
    IHMAcados,
    fsds_ihm_acados_params,
)
from brains_python.control.utils import *
import sys

sys.path.insert(0, "../../test/control/")
from create_test_config import create_test_config
from data_visualization import PlotMode


def main():
    mission, track_name = AVAILABLE_MISSION_TRACK_TUPLES[4]
    controller_params = IHMAcadosParams(**fsds_ihm_acados_params)
    car_params = CarParams(**fsds_car_params)
    car_params.a_y_max = 10.0
    car_params.v_x_max = 15.0
    run_instance = create_test_config(
        "ihm_acados",
        mission,
        track_name,
        controller_params,
        car_params=car_params,
        plot_mode=PlotMode.STATIC,
        simulation_mode=SimulationMode.SIMIL,
    )
    run_instance.run()


if __name__ == "__main__":
    main()
