# Copyright (c) Tudor Oancea, 2022
# create virtual environment (optional)
if [ -d "venv*" ]
then
  echo "venv already exists"
else
  python3 -m venv venv
fi
source venv/bin/activate

# install dependencies through pip
pip3 install --upgrade pip
pip3 install -U -r requirements.txt
pip3 install -U -r requirements_dev.txt
pip3 install -U black "black[d]" pre-commit  # install black and pre-commit if they were not listed in requirements-dev.txt
pre-commit install

# intsall workspace package
pip3 install -e .
