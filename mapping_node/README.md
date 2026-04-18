# mapping_node

Node for building occupancy grid, ESDF, and ΔESDF from PointCloud2 and odometry.

## Topics
- Subscribes:
  - `/cloud` (`sensor_msgs/PointCloud2`)
  - `/odom` (`nav_msgs/Odometry`)
- Publishes:
  - `/occupancy_grid` (`nav_msgs/OccupancyGrid`)
  - `/esdf` (`nav_msgs/OccupancyGrid`)
  - `/delta_esdf` (`nav_msgs/OccupancyGrid`)

## TODO
- Реализовать построение occupancy grid
- Реализовать расчет ESDF
- Реализовать расчет ΔESDF
- Интеграция с improved_dwa_local_planner
