#  Copyright (c) 2022. Tudor Oancea EPFL Racing Team Driverless

from control_module.ihm_forces import generate_solver, PhysicalModel

generate_solver(
    horizon_size=20,
    physical_model=PhysicalModel.KIN_4,
    sampling_time=0.03,
    # parallel=4,
    opt=False,
)
