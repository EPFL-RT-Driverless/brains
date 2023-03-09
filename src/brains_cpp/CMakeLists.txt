# Copyright (c) 2023. Tudor Oancea EPFL Racing Team Driverless
cmake_minimum_required(VERSION 3.8)
project(brains_cpp)

set(CMAKE_CXX_STANDARD 17)
if (CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
	add_compile_options(-Wall -Wextra -Wpedantic -std=c++17 -pthread -Wno-deprecated -Wno-deprecated-declarations)
	# maybe replace that by set(CMAKE_CXX_FLAGS ...)
endif ()

# find AirSim source code
set(AIRSIM_ROOT $ENV{HOME}/Formula-Student-Driverless-Simulator/AirSim)
add_subdirectory("${AIRSIM_ROOT}/cmake/rpclib_wrapper" rpclib_wrapper)
add_subdirectory("${AIRSIM_ROOT}/cmake/AirLib" AirLib)

# find rpclib
set(RPC_LIB_INCLUDES " ${AIRSIM_ROOT}/external/rpclib/include")
set(RPC_LIB rpc) # name of .a file with lib prefix
message(STATUS "found RPC_LIB_INCLUDES=${RPC_LIB_INCLUDES}")

# find ament
find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

# find OpenCV
find_package(OpenCV REQUIRED)

# other dependencies
find_package(PkgConfig REQUIRED)
pkg_check_modules(libcurl REQUIRED libcurl)

# include directories
include_directories(
		SYSTEM
		${OpenCV_INCLUDE_DIRS}
		${AIRSIM_ROOT}/AirLib/deps/eigen3
		${AIRSIM_ROOT}/AirLib/include
		${RPC_LIB_INCLUDES}
)

include_directories(
		include
)

# package libs and executables =================================================
ament_auto_add_executable(main_node
						  src/fsds/fsds_main_node.cpp
						  include/brains_cpp/common.hpp
						  )
target_link_libraries(main_node AirLib ${OpenCV_LIBS} ${libcurl_LIBRARIES} ${CXX_EXP_LIB})

ament_auto_add_executable(camera_node
						  src/fsds/fsds_camera_node.cpp
						  include/brains_cpp/common.hpp
						  )
target_link_libraries(camera_node AirLib ${OpenCV_LIBS} ${libcurl_LIBRARIES} ${CXX_EXP_LIB})

ament_auto_add_executable(lidar_node
						  src/fsds/fsds_lidar_node.cpp
						  include/brains_cpp/common.hpp
						  )
target_link_libraries(lidar_node AirLib ${OpenCV_LIBS} ${libcurl_LIBRARIES} ${CXX_EXP_LIB})

ament_auto_package(INSTALL_TO_SHARE)
