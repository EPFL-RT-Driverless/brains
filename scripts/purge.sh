#  Copyright (c) 2023. Tudor Oancea EPFL Racing Team Driverless
if [[ $(basename $PWD) != "brains" ]]; then
    echo "Please run this script from the root of the brains repository"
    exit 1
fi

rm -fr build install log build.env
