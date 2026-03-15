cmake_minimum_required(VERSION 3.0.2)
project(road_segmentation)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  nav_msgs
  cv_bridge
  std_msgs
  geometry_msgs
  tf2_ros
  tf2_geometry_msgs
)

catkin_package()

catkin_install_python(PROGRAMS
  src/road_segmentation_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
