PYTHON_EXE=$(which python3)
echo "Using python executable: $PYTHON_EXE"
read -p "Continue? [y/n] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    exit 1
fi

# install python dependencies
pip3 install -r src/brains_python/requirements.txt
pip3 install -e src/brains_python
pip3 install -U black pre-commit
pre-commit install

# if there is a directory venv or venv*, create an empty COLCON_IGNORE file inside
# this is to avoid colcon from trying to build the venv
for d in venv*; do
    if [ -d "$d" ]; then
        touch "$d/COLCON_IGNORE"
    fi
done

# if JETSON var is not defined, source the FSDS ros2 bridge
if [ -z "$JETSON" ]; then
  FSDS_ROS_ROOT=$HOME/Formula-Student-Driverless-Simulator/ros2
  if [ -f "$FSDS_ROS_ROOT/install/setup.sh" ]; then

      if [ -z "$ZSH_VERSION" ]; then
          source "$FSDS_ROS_ROOT/install/setup.zsh"
      else
          source "$FSDS_ROS_ROOT/install/setup.bash"
      fi
  fi
fi

# build the workspace with colcon (use bash or zsh depending on the current shell)
if [ -z "$ZSH_VERSION" ]; then
    zsh scrips/build.sh
else
    bash scrips/build.sh
fi
