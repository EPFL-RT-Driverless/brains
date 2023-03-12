#  Copyright (c) 2023. Tudor Oancea EPFL Racing Team Driverless
import os
from glob import glob

from setuptools import setup

package_name = "brains_launch"

setup(
    name=package_name,
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name), glob("launch_files/*launch.[pxy][yma]*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    author="Tudor Oancea",
    author_email="oancea.tudor@icloud.com",
    maintainer="Tudor Oancea",
    maintainer_email="oancea.tudor@icloud.com",
    description="launch configurations of the BRAINS project",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [],
    },
)
