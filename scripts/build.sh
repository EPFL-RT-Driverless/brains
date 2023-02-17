#  Copyright (c) 2023. Tudor Oancea EPFL Racing Team Driverless
if [[ $(basename $PWD) != "brains" ]]; then
    echo "Please run this script from the root of the brains repository"
    exit 1
fi

if [ -n "$BASH_VERSION" ]; then
  source $HOME/Formula-Student-Driverless-Simulator/ros2/install/setup.bash || true
elif [ -n "$ZSH_VERSION" ]; then
  source $HOME/Formula-Student-Driverless-Simulator/ros2/install/setup.zsh || true
fi
PYTHON_EXE=$(which python3)  # makes sure that the correct python interpreter is called in CMake
PYTHONWARNINGS=ignore:::setuptools.command.install,ignore:::setuptools.command.easy_install,ignore:::pkg_resources colcon build --symlink-install --cmake-args "-DPython3_EXECUTABLE=$PYTHON_EXE"
# check current shell (bash or zsh) and source the apprioriate setup file
if [ -n "$BASH_VERSION" ]; then
  source install/setup.bash
elif [ -n "$ZSH_VERSION" ]; then
  source install/setup.zsh
fi
env > build.env
echo "DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH" >> build.env  # for some reason the env and printenv commands do not show this variable so we have to add it manually
