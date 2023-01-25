#  Copyright (c) 2022. Tudor Oancea EPFL Racing Team Driverless
import numpy as np
import pytest

from brains_python.control import (
    LQRParams,
    fsds_car_params,
    LQR,
    CarParams,
    fsds_lqr_params,
)
from brains_python.control.utils import *
from create_test_config import create_test_config


def test_lqr_init():
    try:
        LQR(CarParams(**fsds_car_params), LQRParams(**fsds_lqr_params))
    except Exception as e:
        pytest.fail(e)


@pytest.skip("Skipping test for now")
@pytest.mark.parametrize("args", AVAILABLE_MISSION_TRACK_TUPLES)
def test_lqr_different_tracks(args: tuple):
    mission, track_name = args
    controller_params = LQRParams(**fsds_lqr_params)
    run_instance = create_test_config("lqr", mission, track_name, controller_params)
    run_instance.run()
    assert (
        run_instance.successful_run
    ), "Run was not successful, exit reason: {}".format(run_instance.exit_reason)


if __name__ == "__main__":
    test_lqr_different_tracks(AVAILABLE_MISSION_TRACK_TUPLES[-1])
