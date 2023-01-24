PYTHON_EXE=$(which python3)
echo "Using python executable: $PYTHON_EXE"

# install python dependencies
pip3 install -r src/brains_python/requirements.txt
pip3 install -e src/brains_python
pip3 install -U black pre-commit
pre-commit install

# if JETSON var is not defined, source the FSDS ros2 bridge
if [ -z "$NOFSDS" ]; then
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
    zsh scripts/build.sh
else
    bash scripts/build.sh
fi
