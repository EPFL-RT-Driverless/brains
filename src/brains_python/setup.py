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
            "fsds_velocity_estimation = brains_python.nodes.fsds_velocity_estimation:main",
            "fsds_car_controller = brains_python.nodes.fsds_car_controller:main",
            "ekf_slam = brains_python.nodes.ekf_slam:main",
            "control_known_track = brains_python.nodes.control_known_track:main",
            "point_cloud_file = brains_python.nodes.point_cloud_file:main",
            "video_file_and_yolo = brains_python.nodes.video_file_and_yolo:main",
            "vision_fusion = brains_python.nodes.vision_fusion:main",
        ],
    },
)
