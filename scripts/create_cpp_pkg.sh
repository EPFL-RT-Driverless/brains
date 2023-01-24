#  Copyright (c) 2023. Tudor Oancea EPFL Racing Team Driverless
ros2 pkg create --build-type ament_cmake "$1" --dependencies \
  rclcpp action_msgs actionlib_msgs diagnostic_msgs geometry_msgs \
  lifecycle_msgs map_msgs nav_msgs pcl_msgs pendulum_msgs rosgraph_msgs \
  sensor_msgs shape_msgs statistics_msgs std_msgs stereo_msgs \
  tf2_geometry_msgs tf2_msgs tf2_sensor_msgs trajectory_msgs unique_identifier_msgs \
  visualization_msgs
