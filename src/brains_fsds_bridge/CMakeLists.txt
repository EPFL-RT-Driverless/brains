cmake_minimum_required(VERSION 3.18)
project(brains_fsds_bridge)

# set this to path to AirSim root folder if you want your catkin workspace in a custom directory
set(AIRSIM_ROOT $ENV{HOME}/Formula-Student-Driverless-Simulator/AirSim)

add_subdirectory("${AIRSIM_ROOT}/cmake/rpclib_wrapper" rpclib_wrapper)
add_subdirectory("${AIRSIM_ROOT}/cmake/AirLib" AirLib)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -pthread -Wno-deprecated")

# set(CXX_EXP_LIB "-nostdinc++ -I/usr/include/c++/8 -I/usr/include/x86_64-linux-gnu/c++/8 -nodefaultlibs -l/usr/lib/x86_64-linux-gnu/libc++.so -l/usr/lib/x86_64-linux-gnu/libc++abi.so -lm -lc -lgcc_s -lgcc  -lstdc++fs -fmax-errors=10 -Wnoexcept -Wstrict-null-sentinel")
set(RPC_LIB_INCLUDES " ${AIRSIM_ROOT}/external/rpclib/include")
set(RPC_LIB rpc) # name of .a file with lib prefix
message(STATUS "found RPC_LIB_INCLUDES=${RPC_LIB_INCLUDES}")

# find dependencies
find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

find_package(OpenCV REQUIRED)
include_directories(
  SYSTEM
  ${OpenCV_INCLUDE_DIRS}
  ${AIRSIM_ROOT}/AirLib/deps/eigen3
  ${AIRSIM_ROOT}/AirLib/include
  ${RPC_LIB_INCLUDES}
)

include_directories(include)

find_package(PkgConfig REQUIRED)
pkg_check_modules(libcurl REQUIRED libcurl)

# package libs and executables
ament_auto_add_executable(main_node
  src/main_node.cpp
  include/brains_fsds_bridge/common.hpp
)
target_link_libraries(main_node AirLib ${OpenCV_LIBS} ${libcurl_LIBRARIES} ${CXX_EXP_LIB})

ament_auto_add_executable(camera_node
  src/camera_node.cpp
  include/brains_fsds_bridge/common.hpp
)
target_link_libraries(camera_node AirLib ${OpenCV_LIBS} ${libcurl_LIBRARIES} ${CXX_EXP_LIB})

ament_auto_add_executable(lidar_node
  src/lidar_node.cpp
  include/brains_fsds_bridge/common.hpp
)
target_link_libraries(lidar_node AirLib ${OpenCV_LIBS} ${libcurl_LIBRARIES} ${CXX_EXP_LIB})

ament_auto_package(INSTALL_TO_SHARE launch)
