cmake_minimum_required(VERSION 3.0.2)
project(mapping_node)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  nav_msgs
  tf2_ros
  tf2_geometry_msgs
  std_msgs
)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)

add_executable(mapping_node src/mapping_node.cpp)
target_link_libraries(mapping_node ${catkin_LIBRARIES})
