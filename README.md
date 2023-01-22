Two important env vars to set manually: `BRAINS_CONFIG_FILE` and `BRAINS_LAUNCH_FILE`. The first one specifies the
name of the JSON file used to configure the brains. It is located in the `config` directory and each node can read
anything in it. Conventions to be specified in the `README.md` as there can be no comments in the JSON file. The second
one specifies the file called in `ros2 launch ...` to launch all the necessary nodes.

# project structure


# initial setup
We have to:
1) install the dependencies of the python package
2) install the python package in editable mode in the venv (either pip or conda) via `pip install -e src/python`
3) if there is a venv dir, add a `COLCON_IGNORE` file to it to avoid colcon trying to build it
4) build colcon workspace via `scripts/build.sh`
5) install extras: `black`, `pre-commit`, etc.
6) manually add `<path_to_project>/install/custom_interfaces/lib/python3.10/site-packages` and
   `~/miniforge3/envs/ros_humble/lib/python3.10/site-packages` to the interpreter paths
   of pycharm. Note that you may have to adjust the python version in the path.

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
