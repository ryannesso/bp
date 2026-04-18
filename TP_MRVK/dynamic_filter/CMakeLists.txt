cmake_minimum_required(VERSION 3.0.2)
project(dynamic_filter)

# Dynamic mask layer for costmap
find_package(catkin REQUIRED COMPONENTS
  laser_geometry
  message_filters
  roscpp
  sensor_msgs
  tf
  pcl_ros
  pcl_conversions
  obstacle_detector
  tracked_obstacle_msgs
)
find_package(Eigen3 REQUIRED)


# ШАГ 3: ОБЪЯВЛЯЕМ ПАКЕТ
catkin_package(
  CATKIN_DEPENDS 
    laser_geometry 
    message_filters 
    roscpp 
    sensor_msgs 
    tf 
    pcl_ros 
    pcl_conversions 
    obstacle_detector
    tracked_obstacle_msgs
)

# ШАГ 4: ПУТИ К ЗАГОЛОВКАМ
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)

# Новый узел (очистка скана)
add_executable(scan_cleaner_node 
  src/dynamic_filter.cpp
)
target_link_libraries(scan_cleaner_node
  ${catkin_LIBRARIES}
)

add_dependencies(scan_cleaner_node ${obstacle_detector_EXPORTED_TARGETS})