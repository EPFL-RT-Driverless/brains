# v0.1.0

added all the code from [`control_module`](https://github.com/EPFL-RT-Driverless/control_module)
and added the following features:
- simplified `MotionPlannerParams` by removing the closed param (always true iff mission is
  TRACKDRIVE ) and adding default params for `psi_s` and `psi_e`.
  This was possible by the use of a [`strongpods`](https://github.com/tudoroancea/strongpods),
  a dedicated library for strongly typed PODS creation.


# v0.0.0

initial layout for the repo
