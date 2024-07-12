cmake_minimum_required(VERSION 3.16)
project(mrs_octomap_mapping_planning)
add_definitions(-std=c++17)

add_definitions(-DOCTOMAP_NODEBUGOUT)

set(CATKIN_DEPENDENCIES
  geometry_msgs
  octomap_msgs
  octomap_ros
  visualization_msgs
  roscpp
  dynamic_reconfigure
  mrs_lib
  mrs_msgs
)

# include Eigen3
find_package(Eigen3 REQUIRED)
find_package(catkin REQUIRED COMPONENTS
  ${CATKIN_DEPENDENCIES}
)
find_package(octomap REQUIRED)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${OCTOMAP_INCLUDE_DIRS}
)

set(LINK_LIBRARIES
  ${OCTOMAP_LIBRARIES}
  ${EIGEN3_LIBRARIES}
  ${catkin_LIBRARIES}
)

generate_dynamic_reconfigure_options(cfg/OctomapServer.cfg)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES mrs_octomap_mapping_planning ${LINK_LIBRARIES}
  CATKIN_DEPENDS ${CATKIN_DEPENDENCIES}
  DEPENDS OCTOMAP
)

add_executable(octomap_ray_casting src/octomap_ray_casting.cpp)
target_link_libraries(octomap_ray_casting ${LINK_LIBRARIES})

add_executable(goal_service_call src/goal_service_call.cpp)
target_link_libraries(goal_service_call ${LINK_LIBRARIES})

## --------------------------------------------------------------
## |                           Install                          |
## --------------------------------------------------------------

install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(DIRECTORY tmux/
  USE_SOURCE_PERMISSIONS
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/tmux
)
