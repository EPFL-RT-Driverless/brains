#  Copyright (c) 2023. Tudor Oancea EPFL Racing Team Driverless
from setuptools import setup

package_name = "brains_python"

setup(
    name=package_name,
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
            "numpy_logger = brains_python.nodes.numpy_logger:main",
            "control_only = brains_python.nodes.control_only:main",
        ],
    },
)
