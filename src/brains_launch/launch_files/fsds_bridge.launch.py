import launch
import launch_ros.actions

from os.path import expanduser
import json


def generate_launch_description():
    with open(
        expanduser("~") + "/Formula-Student-Driverless-Simulator/settings.json", "r"
    ) as file:
        settings = json.load(file)

    vehicle_name = list(settings["Vehicles"].keys())[0]
    print("Vehicle name: ", vehicle_name)
    lidar_names = [
        k
        for k, v in settings["Vehicles"][vehicle_name]["Sensors"].items()
        if v["SensorType"] == 6
    ]
    camera_names = [
        k
        for k, v in settings["Vehicles"][vehicle_name]["Cameras"].items()
        if v["CaptureSettings"][0]["ImageType"] == 0
    ]
    print("Camera names: ", camera_names)
    print("Lidar names: ", lidar_names)

    ld = launch.LaunchDescription(
        [
            launch.actions.DeclareLaunchArgument(
                name="host_ip", default_value="localhost"
            ),
            launch.actions.DeclareLaunchArgument(name="timeout", default_value="1.0"),
            launch.actions.DeclareLaunchArgument(
                name="manual_mode", default_value="false"
            ),
            launch.actions.DeclareLaunchArgument(
                name="camera_framerate", default_value="20.0"
            ),
            launch.actions.DeclareLaunchArgument(
                name="statistics_freq", default_value="1.0"
            ),
            launch.actions.DeclareLaunchArgument(
                name="velocity_estimation_covariance", default_value="[0.04, 0.04]"
            ),
            launch.actions.DeclareLaunchArgument(
                name="state_freq", default_value="100.0"
            ),
            launch.actions.DeclareLaunchArgument(
                name="imu_freqs", default_value="[100.0]"
            ),
            launch.actions.DeclareLaunchArgument(
                name="wss_freq", default_value="100.0"
            ),
            launch.actions.DeclareLaunchArgument(
                name="gss_freq", default_value="100.0"
            ),
            launch.actions.DeclareLaunchArgument(
                name="gps_freqs", default_value="[10.0]"
            ),
            *[
                launch_ros.actions.Node(
                    package="brains_cpp",
                    executable="fsds_camera_node",
                    name="fsds_camera_node_" + camera_name,
                    output="screen",
                    on_exit=launch.actions.Shutdown(),
                    parameters=[
                        {
                            "host_ip": launch.substitutions.LaunchConfiguration(
                                "host_ip"
                            )
                        },
                        {
                            "timeout": launch.substitutions.LaunchConfiguration(
                                "timeout"
                            )
                        },
                        {
                            "statistics_freq": launch.substitutions.LaunchConfiguration(
                                "statistics_freq"
                            )
                        },
                        {"camera_name": camera_name},
                        {
                            "framerate": launch.substitutions.LaunchConfiguration(
                                "camera_framerate"
                            )
                        },
                    ],
                )
                for i, camera_name in enumerate(camera_names)
            ],
            *[
                launch_ros.actions.Node(
                    package="brains_cpp",
                    executable="fsds_lidar_node",
                    name="fsds_lidar_node_" + lidar_name,
                    output="screen",
                    on_exit=launch.actions.Shutdown(),
                    parameters=[
                        {
                            "host_ip": launch.substitutions.LaunchConfiguration(
                                "host_ip"
                            )
                        },
                        {
                            "timeout": launch.substitutions.LaunchConfiguration(
                                "timeout"
                            )
                        },
                        {
                            "statistics_freq": launch.substitutions.LaunchConfiguration(
                                "statistics_freq"
                            )
                        },
                        {"lidar_name": lidar_name},
                    ],
                )
                for i, lidar_name in enumerate(lidar_names)
            ],
            launch_ros.actions.Node(
                package="brains_cpp",
                executable="fsds_main_node",
                output="screen",
                on_exit=launch.actions.Shutdown(),
                parameters=[
                    {"host_ip": launch.substitutions.LaunchConfiguration("host_ip")},
                    {"timeout": launch.substitutions.LaunchConfiguration("timeout")},
                    {
                        "statistics_freq": launch.substitutions.LaunchConfiguration(
                            "statistics_freq"
                        )
                    },
                    {
                        "manual_mode": launch.substitutions.LaunchConfiguration(
                            "manual_mode"
                        )
                    },
                    {
                        "car_state_freq": launch.substitutions.LaunchConfiguration(
                            "state_freq"
                        )
                    },
                    {
                        "imu_freqs": launch.substitutions.LaunchConfiguration(
                            "imu_freqs"
                        )
                    },
                    {"wss_freq": launch.substitutions.LaunchConfiguration("wss_freq")},
                    {"gss_freq": launch.substitutions.LaunchConfiguration("gss_freq")},
                    {
                        "gps_freqs": launch.substitutions.LaunchConfiguration(
                            "gps_freqs"
                        )
                    },
                ],
            ),
        ]
    )
    return ld


if __name__ == "__main__":
    generate_launch_description()
