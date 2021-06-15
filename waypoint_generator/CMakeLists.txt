cmake_minimum_required(VERSION 2.8.3)
project(waypoint_generator)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  asv_msgs
  asv_path_trackers
  asv_simulator
  roscpp
  rospy
)

catkin_package()

###########
## Build ##
###########
include_directories(
  ${catkin_INCLUDE_DIRS}
)