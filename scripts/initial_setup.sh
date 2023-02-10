PYTHON_EXE=$(which python3)
echo "Using python executable: $PYTHON_EXE"

# install python dependencies
conda install black[d] pre-commit numpy scipy=1.8.1 matplotlib pytest requests opencv
pip3 install -r src/brains_python/requirements.txt
pip3 install -e src/brains_python
pre-commit install

# build the workspace with colcon (use bash or zsh depending on the current shell)
if [ -z "$ZSH_VERSION" ]; then
    zsh scripts/build.sh
else
    bash scripts/build.sh
fi
