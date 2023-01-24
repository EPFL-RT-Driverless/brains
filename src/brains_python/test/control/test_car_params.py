from typing import Union

from brains_python.control import CarParams
import pytest
import numpy as np


def test_car_params_init():
    try:
        CarParams(
            m=255.0,
            l_r=0.8,
            l_f=0.4,
            a_y_max=7.0,
            v_x_max=10.0,
            delta_max=np.deg2rad(40.0),
            ddelta_max=np.deg2rad(70.0),
            dT_max=4.0,
            L=1.8,
            W=1.0,
            I_z=400.0,
            C_m=3550.0,
            C_r0=0.0,
            C_r2=3.5,
            B_f=17.0,
            C_f=2.0,
            D_f=4.2,
            B_r=20.0,
            C_r=0.8,
            D_r=4.0,
        )
    except Exception as e:
        pytest.fail(e)
