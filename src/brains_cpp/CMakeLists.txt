# Copyright (c) 2023. Tudor Oancea EPFL Racing Team Driverless
cmake_minimum_required(VERSION 3.8)
project(brains_cpp VERSION 0.3.0)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(action_msgs REQUIRED)
find_package(actionlib_msgs REQUIRED)
find_package(diagnostic_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(lifecycle_msgs REQUIRED)
find_package(map_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(pcl_msgs REQUIRED)
find_package(pendulum_msgs REQUIRED)
find_package(rosgraph_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(shape_msgs REQUIRED)
find_package(statistics_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(stereo_msgs REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2_msgs REQUIRED)
find_package(tf2_sensor_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(unique_identifier_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)

  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)

  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
