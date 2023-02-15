# BRAINS (Basically, Racing Autonomously Is Now Simple)

## project structure


## initial setup
See [the Notion page](https://www.notion.so/epflrt/BRAINS-bd87e134b39e4b3bbff7b356c2e9a43d?pvs=4).

## build the project
The whole build process is handled by the [`scripts/build.sh`](scripts/build.sh) script.
It calls colcon with the right flags and arguments, sources the FSDS installation if there is one, and generates the
`build.env` file that is used to set the env vars for the project in PyCharm configurations.

## launch the project
Two important env vars to set manually: `BRAINS_CONFIG_FILE` and `BRAINS_LAUNCH_FILE`. The first one specifies the
name of the JSON file used to configure the brains. It is located in the `config` directory and each node can read
anything in it. Conventions to be specified in the `README.md` as there can be no comments in the JSON file. The second
one specifies the file called in `ros2 launch ...` to launch all the necessary nodes.

Make sure you have sourced the brains environment (see [initial setup](#initial-setup)).
1. set the env vars `BRAINS_CONFIG_FILE` and `BRAINS_LAUNCH_FILE` (see above)
2. launch the project via `scripts/launch.sh`

example:
```bash
BRAINS_CONFIG_FILE=default.json BRAINS_LAUNCH_FILE=default_launch.py zsh scripts/launch.sh
```

You can also manually launch the nodes via `ros2 launch ...`.

## create pycharm run config for a python node
1. if it is not yet done, install the Envfile plugin
2. create a new run config with the appropriate script in the `src/python/nodes` dir
3. set the env vars `BRAINS_CONFIG_FILE` and `BRAINS_LAUNCH_FILE` (see above)
4. select the `Envfile` tab and add the `build.env` file at the root of the project (generated during the build phase)

## other useful scripts:
- `scripts/clean.sh` to clean the build and install directories (via `colcon build --cmake-target clean`)
- `scripts/purge.sh` to completely remove any colcon output
