#  Copyright (c) 2023. Tudor Oancea EPFL Racing Team Driverless
if [ -z "$BRAINS_LAUNCH_FILE" ]; then
  if [ -z "$1" ]; then
    echo "BRAINS_LAUNCH_FILE is empty, using launch_default.json"
    BRAINS_LAUNCH_FILE="launch_default.json"
  else
    BRAINS_LAUNCH_FILE=$1
  fi
fi
if [ -z "$BRAINS_CONFIG_FILE" ]; then
  if [ -z "$2" ]; then
    echo "BRAINS_CONFIG_FILE is empty, using default.json"
    export BRAINS_CONFIG_FILE="default.json"
  else
    export BRAINS_CONFIG_FILE=$2
  fi
fi

source install/setup.zsh
ros2 pylaunch pylaunch "$BRAINS_LAUNCH_FILE"
