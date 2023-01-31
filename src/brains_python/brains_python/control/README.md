# Control module

<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"/></a>
## Brief description

A collection of all the motion planning and control algorithms developed for the
Driverless cars of the EPFL Racing Team.
These two tasks are accomplished by the classes `MotionPlanner` and `Controller`. However, the user of this module
should almost never have to deal directly with these classes. Instead, the class `MotionPlannerController` should be
used. It contains an instance of `MotionPlanner` and two instances of `Controller` (allowing for the use of separate
control algorithms for racing and stopping the car).

The usage is fairly straight forward. You have to declare an instance of `MotionPlannerController` and then call its
method `compute_control()` with the current state and control inputs of the car. The most important part of the usage
is the definition of the controller parameters and the motion planner parameter (respectively instances of `ControllerParams`
and `MotionPlannerParams`). Please look at the example or ask a professional for help.

```python
from control_module import MotionPlannerController, CarParams, fsds_car_params, MotionPlannerParams
from control_module.stanley import StanleyParams, stanley_mercury_params

# initialization
motion_planner_controller_instance = MotionPlannerController(
    car_params=CarParams(**fsds_car_params),
    motion_planner_params=MotionPlannerParams(**di),
    racing_controller_params=StanleyParams(**stanley_mercury_params),
    stopping_controller_params=StanleyParams(**stanley_mercury_params),
)

# compute control input
res = motion_planner_controller_instance.compute_control(current_state, current_control)
```

## Currently supported controllers
You can find more comprehensive description of these controllers on
[Notion](https://www.notion.so/epflrt/Control-Home-b1f35d2f1ba5473fb86af3b79e5508ec#ccb40bb220aa47319a7b734791eba76a)
.

### Stanley controller
Longitudinal PI controller coupled with a Stanley controller (developed by Stanford
University for the DARPA Grand challenge 2005).

### LQR controller
Linear Quadratic Regulator using a linear time varying model obtained by linearizing the nonlinear model
KIN_4 model around each point of the reference trajectory.

### IHM controller
Implementation of the _Incredibly Honorable MPC_ (aka IHM) algorithm (first functional MPC formulation implemented).
It is basically the fully nonlinear counterpart of the LQR controller, with constraints on the longitudinal velocity and
controls.

## Dev setup

As described on
[Notion](https://www.notion.so/epflrt/How-to-work-at-the-EPFL-Racing-Team-c9d1f06a81854c628b38d4107eac624e)
, you have to run [`scripts/setup_workspace.sh`](scripts/setup_workspace.sh) in a
Unis shell to install all the dependencies in a virtual environment.

To update the dependencies when you change git context, run the magic command in
[`scripts/update_deps.sh`](scripts/update_deps.sh)
