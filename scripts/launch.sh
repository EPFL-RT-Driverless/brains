#  Copyright (c) 2023. Tudor Oancea EPFL Racing Team Driverless
if [ -z "$BRAINS_LAUNCH_FILE" ]; then
  if [ -z "$1" ]; then
    echo "BRAINS_LAUNCH_FILE is empty, using default_launch.py"
    BRAINS_LAUNCH_FILE="default_launch.py"
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

source install/setup.sh
ros2 launch brains_launch "$BRAINS_LAUNCH_FILE"
