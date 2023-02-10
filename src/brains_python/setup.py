#  Copyright (c) 2023. Tudor Oancea EPFL Racing Team Driverless
from setuptools import setup

package_name = "brains_python"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    author="Tudor Oancea",
    author_email="oancea.tudor@icloud.com",
    maintainer="Tudor Oancea",
    maintainer_email="oancea.tudor@icloud.com",
    description="python code of the BRAINS project",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "velocity_estimation = brains_python.nodes.velocity_estimation:main",
            "fsds_car_sensors = brains_python.nodes.fsds_car_sensors:main",
        ],
    },
)
