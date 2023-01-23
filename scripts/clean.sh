#  Copyright (c) 2023. Tudor Oancea EPFL Racing Team Driverless
PYTHONWARNINGS=ignore:::setuptools.command.install,ignore:::setuptools.command.easy_install,ignore:::pkg_resources \
  colcon build --cmake-target clean
