# v0.4.1

:sparkles: updated scripts to add checks

# v0.4.0

- added neuromancer to `env.yml` and removed `src/brains_python/requirements.txt`
- first version of Velocity estimation: `VE0`. Still in `work` folder
- added back some CI tests to check the correct version is written in setup.py and CMakeLists.txt files

# v0.3.0

introducing an `env.yml` file for the conda environment.

# v0.2.0

created first nodes that seem to communicate just fine together.
they are empty tho, we need to integrate the code for each module.

# v0.1.1

added first implementations of `EKFLocalization` and `EKFSLAM` (not the more robust yet)

# v0.1.0

added all the code from [`control_module`](https://github.com/EPFL-RT-Driverless/control_module)
and added the following features:
- simplified `MotionPlannerParams` by removing the closed param (always true iff mission is
  TRACKDRIVE ) and adding default params for `psi_s` and `psi_e`.
  This was possible by the use of a [`strongpods`](https://github.com/tudoroancea/strongpods),
  a dedicated library for strongly typed PODS creation.


# v0.0.0

initial layout for the repo
