import pytest

from brains_python.control import StanleyParams


def test_stanley_params_init():
    try:
        StanleyParams(
            k_P=1.3,
            k_I=1.0000e-07,
            k_psi=1.6457e00,
            k_e=10.0,
            k_s=1.0e-7,
            sampling_time=0.01,
        )
    except Exception as e:
        pytest.fail(e)
