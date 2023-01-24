import pytest
from brains_python.control import ControllerParams


def test_controller_init():
    try:
        ControllerParams(sampling_time=0.01, x=0x01)
    except Exception as e:
        pytest.fail(e)
