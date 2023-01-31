#  Copyright (c) 2022. Tudor Oancea EPFL Racing Team Driverless
import pytest

from brains_python.control import (
    StanleyParams,
    stanley_params_from_mission,
    fsds_car_params,
    Stanley,
    CarParams,
)
from brains_python.control.utils import *
from create_test_config import create_test_config


def test_stanley_init():
    try:
        Stanley(
            CarParams(**fsds_car_params),
            StanleyParams(
                k_P=1.3,
                k_I=1.0000e-07,
                k_psi=1.6457e00,
                k_e=10.0,
                k_s=1.0e-7,
                sampling_time=0.01,
            ),
        )
    except Exception as e:
        pytest.fail(e)


# @pytest.skip("Skipping test for now")
@pytest.mark.parametrize("args", AVAILABLE_MISSION_TRACK_TUPLES)
def test_stanley_different_tracks(args: tuple):
    mission, track_name = args
    controller_params = StanleyParams(**stanley_params_from_mission(mission))
    run_instance = create_test_config("stanley", mission, track_name, controller_params)
    run_instance.run()
    assert (
        run_instance.successful_run
    ), "Run was not successful, exit reason: {}".format(run_instance.exit_reason)


if __name__ == "__main__":
    test_stanley_different_tracks(AVAILABLE_MISSION_TRACK_TUPLES[2])
