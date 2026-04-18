#pragma once
#include <costmap_2d/layer.h>
#include <costmap_2d/layered_costmap.h>
#include <costmap_2d/costmap_layer.h>
#include <ros/ros.h>
#include <tracked_obstacle_msgs/TrackedCircleArray.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <geometry_msgs/PointStamped.h>
#include <mutex>
#include <vector>

namespace costmap_2d {

struct DynObstacleInfo {
  double x, y, radius;
  bool   is_ghost;
  double time_since_seen;
};

class DynamicObstacleLayer : public CostmapLayer {
public:
  DynamicObstacleLayer();
  ~DynamicObstacleLayer() override = default;

  void onInitialize() override;
  void updateBounds(double rx, double ry, double ryaw,
                    double* min_x, double* min_y,
                    double* max_x, double* max_y) override;
  void updateCosts(costmap_2d::Costmap2D& master_grid,
                   int min_i, int min_j, int max_i, int max_j) override;
  void reset() override { onInitialize(); }

private:
  void trackedCallback(const tracked_obstacle_msgs::TrackedCircleArray::ConstPtr& msg);

  std::string tracked_topic_;
  double      clear_radius_buffer_;
  bool        clear_ghost_objects_;
  double      ghost_ttl_;
  bool        static_lethal_only_;

  ros::Subscriber tracked_sub_;
  ros::NodeHandle nh_;

  std::mutex                   data_mutex_;
  std::vector<DynObstacleInfo> obstacles_;
  ros::Time                    last_msg_time_;

  tf2_ros::Buffer            tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
  std::string                global_frame_;

  double dirty_min_x_, dirty_min_y_, dirty_max_x_, dirty_max_y_;
  bool   has_dirty_;
};

}  // namespace costmap_2d
