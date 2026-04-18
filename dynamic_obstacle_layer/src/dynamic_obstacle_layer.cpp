#include <dynamic_obstacle_layer/dynamic_obstacle_layer.h>
#include <pluginlib/class_list_macros.h>
#include <costmap_2d/costmap_2d.h>
#include <cmath>
#include <algorithm>

PLUGINLIB_EXPORT_CLASS(costmap_2d::DynamicObstacleLayer, costmap_2d::Layer)

namespace costmap_2d {

// ─────────────────────────────────────────────────────────────────────────────
DynamicObstacleLayer::DynamicObstacleLayer()
  : tf_listener_(tf_buffer_)
  , has_dirty_(false)
  , dirty_min_x_(0), dirty_min_y_(0), dirty_max_x_(0), dirty_max_y_(0)
{}

// ─────────────────────────────────────────────────────────────────────────────
void DynamicObstacleLayer::onInitialize()
{
  ros::NodeHandle nh("~/" + name_);

  nh.param<std::string>("tracked_topic",       tracked_topic_,       "/obstacles_tracked");
  nh.param("clear_radius_buffer",  clear_radius_buffer_,  0.35);
  nh.param("clear_ghost_objects",  clear_ghost_objects_,  true);
  nh.param("ghost_ttl",            ghost_ttl_,            2.0);
  nh.param("static_lethal_only",   static_lethal_only_,   false);
  nh.param("enabled",              enabled_,              true);

  global_frame_ = layered_costmap_->getGlobalFrameID();

  tracked_sub_ = nh_.subscribe(tracked_topic_, 1,
      &DynamicObstacleLayer::trackedCallback, this);

  has_dirty_ = false;
  current_ = true;  // VERY IMPORTANT: without this, move_base blocks forever!
  ROS_INFO("[DynamicObstacleLayer] init: topic=%s frame=%s buf=%.2f ghost=%d",
           tracked_topic_.c_str(), global_frame_.c_str(),
           clear_radius_buffer_, (int)clear_ghost_objects_);
}

// ─────────────────────────────────────────────────────────────────────────────
void DynamicObstacleLayer::trackedCallback(
    const tracked_obstacle_msgs::TrackedCircleArray::ConstPtr& msg)
{
  if (!enabled_) return;

  std::string src_frame = msg->header.frame_id;
  ros::Time   stamp     = msg->header.stamp;
  if (stamp.isZero()) stamp = ros::Time::now();

  // Build new snapshot in the costmap global frame
  std::vector<DynObstacleInfo> new_obs;
  new_obs.reserve(msg->circles.size());

  double new_min_x = std::numeric_limits<double>::max();
  double new_min_y = std::numeric_limits<double>::max();
  double new_max_x = std::numeric_limits<double>::lowest();
  double new_max_y = std::numeric_limits<double>::lowest();
  bool   any       = false;

  for (const auto& c : msg->circles) {
    // Skip ghost objects if not wanted, or if too old
    if (c.is_ghost) {
      if (!clear_ghost_objects_) continue;
      if (c.time_since_seen > ghost_ttl_) continue;
    }

    // Transform centre to global costmap frame
    geometry_msgs::PointStamped in_pt, out_pt;
    in_pt.header.frame_id = src_frame;
    in_pt.header.stamp    = stamp;
    in_pt.point.x = c.center.x;
    in_pt.point.y = c.center.y;
    in_pt.point.z = 0.0;

    try {
      tf_buffer_.transform(in_pt, out_pt, global_frame_,
                           ros::Duration(0.05));
    } catch (tf2::TransformException& ex) {
      ROS_WARN_THROTTLE(2.0, "[DynamicObstacleLayer] TF error: %s", ex.what());
      continue;
    }

    double r = std::max(0.1, (double)c.radius) + clear_radius_buffer_;

    DynObstacleInfo info;
    info.x              = out_pt.point.x;
    info.y              = out_pt.point.y;
    info.radius         = r;
    info.is_ghost       = c.is_ghost;
    info.time_since_seen = c.time_since_seen;
    new_obs.push_back(info);

    // Expand dirty bounding box
    new_min_x = std::min(new_min_x, info.x - r);
    new_min_y = std::min(new_min_y, info.y - r);
    new_max_x = std::max(new_max_x, info.x + r);
    new_max_y = std::max(new_max_y, info.y + r);
    any = true;
  }

  std::lock_guard<std::mutex> lk(data_mutex_);
  obstacles_     = std::move(new_obs);
  last_msg_time_ = stamp;

  if (any) {
    dirty_min_x_ = new_min_x;
    dirty_min_y_ = new_min_y;
    dirty_max_x_ = new_max_x;
    dirty_max_y_ = new_max_y;
    has_dirty_   = true;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
void DynamicObstacleLayer::updateBounds(
    double /*rx*/, double /*ry*/, double /*ryaw*/,
    double* min_x, double* min_y,
    double* max_x, double* max_y)
{
  if (!enabled_) return;
  current_ = true;

  std::lock_guard<std::mutex> lk(data_mutex_);
  if (!has_dirty_) return;

  // Expand the update region to include all dynamic obstacle circles
  *min_x = std::min(*min_x, dirty_min_x_);
  *min_y = std::min(*min_y, dirty_min_y_);
  *max_x = std::max(*max_x, dirty_max_x_);
  *max_y = std::max(*max_y, dirty_max_y_);
}

// ─────────────────────────────────────────────────────────────────────────────
void DynamicObstacleLayer::updateCosts(
    costmap_2d::Costmap2D& master_grid,
    int min_i, int min_j, int max_i, int max_j)
{
  if (!enabled_) return;

  std::lock_guard<std::mutex> lk(data_mutex_);
  if (obstacles_.empty()) return;

  // Check message staleness — do not clear if data is too old (>3 s)
  double data_age = (ros::Time::now() - last_msg_time_).toSec();
  if (data_age > 3.0) return;

  double resolution  = master_grid.getResolution();
  double origin_x    = master_grid.getOriginX();
  double origin_y    = master_grid.getOriginY();
  unsigned int width  = master_grid.getSizeInCellsX();
  unsigned int height = master_grid.getSizeInCellsY();

  for (const auto& obs : obstacles_) {
    // worldToMap requires unsigned int — use proper types to avoid UB
    unsigned int ucx, ucy;
    if (!master_grid.worldToMap(obs.x, obs.y, ucx, ucy))
      continue;

    int cx_cell = static_cast<int>(ucx);
    int cy_cell = static_cast<int>(ucy);
    int radius_cells = static_cast<int>(std::ceil(obs.radius / resolution)) + 1;

    int i_min = std::max(min_i, cx_cell - radius_cells);
    int i_max = std::min(max_i, cx_cell + radius_cells);
    int j_min = std::max(min_j, cy_cell - radius_cells);
    int j_max = std::min(max_j, cy_cell + radius_cells);

    double r2 = obs.radius * obs.radius;

    for (int j = j_min; j <= j_max; ++j) {
      for (int i = i_min; i <= i_max; ++i) {
        if (i < 0 || j < 0 ||
            static_cast<unsigned int>(i) >= width ||
            static_cast<unsigned int>(j) >= height)
          continue;

        // Check if cell is within the circle
        double wx = origin_x + (i + 0.5) * resolution;
        double wy = origin_y + (j + 0.5) * resolution;
        double dx = wx - obs.x;
        double dy = wy - obs.y;
        if (dx*dx + dy*dy > r2) continue;

        // Optionally skip static-map LETHAL cells (real walls)
        if (static_lethal_only_ &&
            master_grid.getCost(static_cast<unsigned int>(i),
                                static_cast<unsigned int>(j)) == costmap_2d::LETHAL_OBSTACLE)
          continue;

        // Stamp FREE_SPACE to erase the dynamic obstacle footprint
        master_grid.setCost(static_cast<unsigned int>(i),
                            static_cast<unsigned int>(j),
                            costmap_2d::FREE_SPACE);
      }
    }
  }
}

}  // namespace costmap_2d
