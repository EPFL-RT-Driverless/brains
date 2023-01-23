Two important env vars to set manually: `BRAINS_CONFIG_FILE` and `BRAINS_LAUNCH_FILE`. The first one specifies the
name of the JSON file used to configure the brains. It is located in the `config` directory and each node can read
anything in it. Conventions to be specified in the `README.md` as there can be no comments in the JSON file. The second
one specifies the file called in `ros2 launch ...` to launch all the necessary nodes.

# project structure


# initial setup
Create a virtual environment either using `python3 -m venv` or `conda env create` and install the following
dependencies manually:
- `torch`: the installation changes a lot depending on whether you have cuda or not, or if you have to install a
  custom weird version on the Jetson.
- `casadi`: there is still no official lib for arm, so you may have to install it from source or use the following
  develop [nightly builds](https://github.com/casadi/casadi/releases/tag/nightly-develop).
- `acados`: You have to install it from source so ...
- `forcespro`: same

# launch the project
make sure you have sourced the base ros2 environment
1) set the env vars `BRAINS_CONFIG_FILE` and `BRAINS_LAUNCH_FILE` (see above)
2) launch the project via `scripts/launch.sh`

example:
```bash
BRAINS_CONFIG_FILE=default.json BRAINS_LAUNCH_FILE=default_launch.py zsh scripts/launch.sh
```

# create pycharm run config for a python node
1) if it is not yet done, install the Envfile plugin
2) create a new run config with the appropriate script in the `src/python/nodes` dir
3) set the env vars `BRAINS_CONFIG_FILE` and `BRAINS_LAUNCH_FILE` (see above)
4) select the `Envfile` tab and add the `build.env` file at the root of the project (generated during the build phase)

# other useful scripts:
- `scripts/clean.sh` to clean the build and install directories (via `colcon build --cmake-target clean`)
- `scripts/purge.sh` to completely remove any colcon output
