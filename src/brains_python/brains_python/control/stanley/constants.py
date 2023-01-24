# Copyright (c) 2023. Tudor Oancea EPFL Racing Team Driverless
from brains_python.common import Mission

__all__ = [
    "stanley_mercury_params",
    "stanley_minimercury_params",
    "stanley_mercury_params_2",
    "stanley_params_from_mission",
]

stanley_minimercury_params = {
    "k_P": 6.0,
    "k_I": 1.0,
    "k_psi": 0.4295729275792759,
    "k_e": 14.607423294873527,
    "k_s": 20.0,
    "sampling_time": 0.01,
}

stanley_mercury_params_2 = {
    "k_P": 7.0 / 10.0,
    "k_I": 1.0 / 10.0,
    "k_psi": 1.12,
    "k_e": 10.0,
    "k_s": 10.771,
    "sampling_time": 0.01,
}

stanley_mercury_params = {
    "k_P": 1.3,
    "k_I": 1.0000e-07,
    "k_psi": 1.6457e00,
    "k_e": 10.0,
    "k_s": 1.0e-7,
    "sampling_time": 0.01,
}


def stanley_params_from_mission(mission: Mission):
    if mission in {Mission.SKIDPAD, Mission.SHORT_SKIDPAD}:
        return stanley_mercury_params
    else:
        return stanley_mercury_params_2
