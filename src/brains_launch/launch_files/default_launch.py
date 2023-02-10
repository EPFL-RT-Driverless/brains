#  Copyright (c) 2023. Tudor Oancea EPFL Racing Team Driverless
import launch
import launch_ros.actions


def generate_launch_description():
    return launch.LaunchDescription(
        [
            launch_ros.actions.Node(
                name="velocity_estimation",
                executable="velocity_estimation",
                package="brains_python",
            ),
            launch_ros.actions.Node(
                name="fsds_car_sensors",
                executable="fsds_car_sensors",
                package="brains_python",
            ),
        ]
    )
