#  Copyright (c) 2023. Tudor Oancea EPFL Racing Team Driverless
pip3 install -e src/brains_python
pre-commit install
# build the workspace with colcon (use bash or zsh depending on the current shell)
if [ -z "$ZSH_VERSION" ]; then
    zsh scripts/build.sh
else
    bash scripts/build.sh
fi
