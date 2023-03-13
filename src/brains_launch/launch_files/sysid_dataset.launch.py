import launch
import launch_ros.actions


def generate_launch_description():
    ld = launch.LaunchDescription(
        [
            launch.actions.DeclareLaunchArgument(
                name="host_ip", default_value="localhost"
            ),
            launch.actions.DeclareLaunchArgument(name="timeout", default_value="1.0"),
            launch.actions.DeclareLaunchArgument(
                name="statistics_freq", default_value="1.0"
            ),
            launch.actions.DeclareLaunchArgument(
                name="track_name", default_value="fsds_competition_2"
            ),
            launch.actions.DeclareLaunchArgument(name="lap_count", default_value="10"),
            launch.actions.DeclareLaunchArgument(name="v_x_max", default_value="10.0"),
            launch.actions.DeclareLaunchArgument(name="a_y_max", default_value="7.0"),
            launch.actions.DeclareLaunchArgument(name="W", default_value="1.5"),
            launch.actions.DeclareLaunchArgument(
                name="log_path", default_value="state_control_log.csv"
            ),
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
                    {"manual_mode": False},
                    {"car_state_freq": 100.0},
                    {"wss_freq": 0.0},
                ],
            ),
            launch_ros.actions.Node(
                package="brains_python",
                executable="control_only",
                output="screen",
                on_exit=launch.actions.Shutdown(),
                parameters=[
                    {
                        "track_name": launch.substitutions.LaunchConfiguration(
                            "track_name"
                        )
                    },
                    {
                        "lap_count": launch.substitutions.LaunchConfiguration(
                            "lap_count"
                        )
                    },
                    {"v_x_max": launch.substitutions.LaunchConfiguration("v_x_max")},
                ],
            ),
            launch_ros.actions.Node(
                package="brains_python",
                executable="csv_logger",
                output="screen",
                on_exit=launch.actions.Shutdown(),
                parameters=[
                    {"log_path": launch.substitutions.LaunchConfiguration("log_path")},
                ],
            ),
        ]
    )
    return ld


if __name__ == "__main__":
    generate_launch_description()
