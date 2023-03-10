#  Copyright (c) 2023. Tudor Oancea EPFL Racing Team Driverless
# check that the script is run from the root of the brains repository
if [[ $(basename "$PWD") != "brains" ]]; then
    echo "Please run this script from the root of the brains repository"
    exit 1
fi
# makes sure that the correct python interpreter is called in CMake
PYTHON_EXE=$(which python3)
# colcon build with all the right options
PYTHONWARNINGS=ignore:::setuptools.command.install,ignore:::setuptools.command.easy_install,ignore:::pkg_resources colcon build --symlink-install --cmake-args -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -G Ninja -DPython3_EXECUTABLE=$PYTHON_EXE
# check current shell (bash or zsh) and source the apprioriate setup file
if [ -n "$BASH_VERSION" ]; then
  source install/setup.bash
elif [ -n "$ZSH_VERSION" ]; then
  source install/setup.zsh
fi
env > build.env
echo "DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH" >> build.env  # for some reason the env and printenv commands do not show this variable so we have to add it manually
