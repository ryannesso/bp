cmake_minimum_required(VERSION 3.0.2)
project(improved_dwa_local_planner)

# === ШАГ 1: Добавляем 'obstacle_detector' в список поиска ===
find_package(catkin REQUIRED COMPONENTS
  roscpp
  pluginlib
  nav_core
  nav_msgs
  base_local_planner
  costmap_2d
  geometry_msgs
  tf2_ros
  obstacle_detector
)

catkin_package(
  INCLUDE_DIRS include 
  LIBRARIES improved_dwa_local_planner
  CATKIN_DEPENDS roscpp pluginlib nav_core nav_msgs base_local_planner costmap_2d geometry_msgs tf2_ros obstacle_detector
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

# === ИЗМЕНЕНИЕ: Название библиотеки теперь соответствует проекту ===
add_library(${PROJECT_NAME}
  src/improved_dwa_local_planner.cpp
)

target_link_libraries(${PROJECT_NAME}
  ${catkin_LIBRARIES}
)

# Секция install - оставляем без изменений
install(TARGETS ${PROJECT_NAME}
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(FILES
  planner_plugins.xml
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)