#  Copyright (c) 2023. Tudor Oancea EPFL Racing Team Driverless
import launch
import launch_ros.actions


def generate_launch_description():
    return launch.LaunchDescription(
        [
            launch_ros.actions.Node(
                name="fsds_car_sensors",
                executable="fsds_car_sensors",
                package="brains_python",
            ),
            launch_ros.actions.Node(
                name="fsds_car_controller",
                executable="fsds_car_controller",
                package="brains_python",
            ),
            launch_ros.actions.Node(
                name="control_known_track",
                executable="control_known_track",
                package="brains_python",
            ),
            launch_ros.actions.Node(
                name="point_cloud_file",
                executable="point_cloud_file",
                package="brains_python",
            ),
            launch_ros.actions.Node(
                name="video_file_and_yolo",
                executable="video_file_and_yolo",
                package="brains_python",
            ),
            launch_ros.actions.Node(
                name="vision_fusion",
                executable="vision_fusion",
                package="brains_python",
            ),
            launch_ros.actions.Node(
                name="velocity_estimation",
                executable="velocity_estimation",
                package="brains_python",
            ),
            launch_ros.actions.Node(
                name="ekf_slam",
                executable="ekf_slam",
                package="brains_python",
            ),
        ]
    )
