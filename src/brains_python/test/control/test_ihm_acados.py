# Copyright (c) 2023. Tudor Oancea EPFL Racing Team Driverless
import pytest

from brains_python.control import (
    fsds_car_params,
    CarParams,
    IHMAcadosParams,
    IHMAcados,
    fsds_ihm_acados_params,
)
from brains_python.control.utils import *
from create_test_config import create_test_config


def test_ihm_acados_init():
    try:
        IHMAcados(
            CarParams(**fsds_car_params),
            IHMAcadosParams(**fsds_ihm_acados_params),
        )
    except Exception as e:
        pytest.fail(e)


@pytest.skip("Skipping test for now")
@pytest.mark.parametrize("args", AVAILABLE_MISSION_TRACK_TUPLES)
def test_ihm_acados_different_tracks(args: tuple):
    mission, track_name = args
    controller_params = IHMAcadosParams(**fsds_ihm_acados_params)
    run_instance = create_test_config(
        "ihm_acados", mission, track_name, controller_params
    )
    run_instance.run()
    assert (
        run_instance.successful_run
    ), "Run was not successful, exit reason: {}".format(run_instance.exit_reason)


if __name__ == "__main__":
    test_ihm_acados_different_tracks(AVAILABLE_MISSION_TRACK_TUPLES[3])
