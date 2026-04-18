#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/PointField.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <algorithm>
#include <cmath>
#include <geometry_msgs/PointStamped.h>
#include <limits>
#include <obstacle_detector/Obstacles.h>
#include <ros/ros.h>
#include <sensor_msgs/LaserScan.h>
#include <string>
#include <tf/transform_listener.h>
#include <tracked_obstacle_msgs/TrackedCircle.h>
#include <tracked_obstacle_msgs/TrackedCircleArray.h>
#include <vector>
#include <visualization_msgs/MarkerArray.h>

// ─────────────────────────────────────────────────────────────────────────────
// DATA STRUCTURES
// ─────────────────────────────────────────────────────────────────────────────

struct Cluster {
  std::vector<int>   indices;
  std::vector<float> xs;
  std::vector<float> ys;
};

struct SpatialMemory {
  ros::Time last_seen;
  float x, y;
  float vx, vy;
  float radius;
  ros::Time unstable_until;
  ros::Time quarantine_until;   // NEW: первые N секунд объект фильтруется с удвоенным радиусом
  ros::Time first_seen;         // NEW: время первого появления
  bool  updated_this_frame;
  int   id;
  int   tracker_id;
};

struct FilterTarget {
  float x1, y1;
  float x2, y2;
  float radius;
};

// ─────────────────────────────────────────────────────────────────────────────
// SCAN CLEANER
// ─────────────────────────────────────────────────────────────────────────────

class ScanCleaner {
public:
  ros::Publisher dynamic_mask_pub_;

  ScanCleaner() {
    ros::NodeHandle pnh("~");

    pnh.param("speed_threshold",    speed_threshold_,    0.08);
    pnh.param("ttl_duration",       ttl_duration_,       3.5);
    pnh.param("radius_buffer",      radius_buffer_,      0.4);
    pnh.param("proximity_threshold",proximity_threshold_, 0.3);
    pnh.param("max_tilt_angle",     max_tilt_angle_,     0.08);
    pnh.param("gap_threshold",      gap_threshold_,      0.25);
    pnh.param("max_cluster_span",   max_cluster_span_,   1.5);
    pnh.param("dilation_rays",      dilation_rays_,      3);

    // NEW: карантинные параметры
    pnh.param("quarantine_duration", quarantine_duration_,  0.6);  // сек после появления
    pnh.param("quarantine_radius_mult", quarantine_radius_mult_, 2.0); // множитель радиуса в карантине

    scan_sub_      = nh_.subscribe("/scan",      1, &ScanCleaner::scanCallback,      this);
    obstacles_sub_ = nh_.subscribe("/obstacles", 1, &ScanCleaner::obstaclesCallback, this);

    filtered_pub_   = nh_.advertise<sensor_msgs::LaserScan>("/scan_cleaned", 1);
    viz_pub_        = nh_.advertise<visualization_msgs::MarkerArray>("/scan_cleaner_viz", 1);
    tracked_pub_    = nh_.advertise<tracked_obstacle_msgs::TrackedCircleArray>("/obstacles_tracked", 1);
    dynamic_mask_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/dynamic_mask", 1);

    has_obstacles_     = false;
    memory_id_counter_ = 1000;
  }

private:
  ros::NodeHandle nh_;
  ros::Subscriber scan_sub_;
  ros::Subscriber obstacles_sub_;
  ros::Publisher  filtered_pub_;
  ros::Publisher  viz_pub_;
  ros::Publisher  tracked_pub_;
  tf::TransformListener tf_listener_;

  double speed_threshold_;
  double ttl_duration_;
  double radius_buffer_;
  double proximity_threshold_;
  double max_tilt_angle_;
  double gap_threshold_;
  double max_cluster_span_;
  int    dilation_rays_;
  double quarantine_duration_;
  double quarantine_radius_mult_;

  obstacle_detector::Obstacles       latest_obs_msg_;
  std::vector<SpatialMemory>         memories_;
  int                                memory_id_counter_;
  bool                               has_obstacles_;

  // ───────────────────────────────────────────────────────────────── helpers ──

  float distToSegment(float px, float py,
                      float x1, float y1, float x2, float y2) {
    float l2 = (x1-x2)*(x1-x2) + (y1-y2)*(y1-y2);
    if (l2 == 0.0f) return std::hypot(px-x1, py-y1);
    float t = std::max(0.0f, std::min(1.0f,
        ((px-x1)*(x2-x1) + (py-y1)*(y2-y1)) / l2));
    return std::hypot(px - (x1 + t*(x2-x1)), py - (y1 + t*(y2-y1)));
  }

  // ──────────────────────────────────────────────────────── obstacles callback ──

  void obstaclesCallback(const obstacle_detector::Obstacles::ConstPtr &msg) {
    latest_obs_msg_ = *msg;
    has_obstacles_  = true;
  }

  // ──────────────────────────────────────────────────────────── scan callback ──

  void scanCallback(const sensor_msgs::LaserScan::ConstPtr &scan_in) {
    if (!has_obstacles_) {
      filtered_pub_.publish(scan_in);
      return;
    }

    ros::Time   scan_time   = scan_in->header.stamp;
    std::string laser_frame = scan_in->header.frame_id;
    std::string obs_frame   = latest_obs_msg_.header.frame_id;

    double dt_msg = std::max(0.0,
        std::min((scan_time - latest_obs_msg_.header.stamp).toSec(), 0.5));

    std::vector<FilterTarget>               targets;
    std::vector<std::pair<float,float>>     dynamic_points;

    for (auto &m : memories_)
      m.updated_this_frame = false;

    // ── ЭТАП 1: СОПОСТАВЛЕНИЕ ─────────────────────────────────────────────────

    for (const auto &circle : latest_obs_msg_.circles) {
      float vx = std::isnan(circle.velocity.x) ? 0.0f : (float)circle.velocity.x;
      float vy = std::isnan(circle.velocity.y) ? 0.0f : (float)circle.velocity.y;
      float r  = std::isnan(circle.radius)     ? 0.3f : (float)circle.radius;
      float cx = (float)circle.center.x;
      float cy = (float)circle.center.y;
      float speed    = std::hypot(vx, vy);
      bool  is_moving = (speed > (float)speed_threshold_);

      SpatialMemory *best_match = nullptr;
      float          best_dist  = -1.f;

      for (auto &m : memories_) {
        if (m.updated_this_frame) continue;
        double dt_mem = std::max(0.0, (scan_time - m.last_seen).toSec());
        float gx = m.x + m.vx*(float)dt_mem;
        float gy = m.y + m.vy*(float)dt_mem;
        float dist_sq = std::min(
            std::pow(cx - m.x, 2) + std::pow(cy - m.y, 2),
            std::pow(cx - gx, 2) + std::pow(cy - gy, 2));
        float match_limit = (float)proximity_threshold_ + r + m.radius;
        if (dist_sq < match_limit * match_limit) {
          if (best_match == nullptr || dist_sq < best_dist) {
            best_match = &m;
            best_dist  = dist_sq;
          }
        }
      }

      if (best_match != nullptr || is_moving) {
        ros::Time new_unstable = ros::Time(0);

        if (best_match != nullptr) {
          // Обновление существующей памяти
          new_unstable = best_match->unstable_until;
          float old_speed = std::hypot(best_match->vx, best_match->vy);
          bool stopped    = !is_moving;
          bool reversed   = (old_speed > 0.05f && speed > 0.05f &&
                             (best_match->vx*vx + best_match->vy*vy < 0));
          bool accelerated = (old_speed <= (float)speed_threshold_ && is_moving);

          if (stopped || reversed || accelerated)
            new_unstable = scan_time + ros::Duration(1.0);

          best_match->last_seen         = scan_time;
          best_match->x                 = cx;
          best_match->y                 = cy;
          best_match->vx                = vx;
          best_match->vy                = vy;
          best_match->radius            = r;
          best_match->unstable_until    = new_unstable;
          best_match->tracker_id        = circle.id;
          best_match->updated_this_frame = true;
        } else {
          // ── НОВЫЙ ОБЪЕКТ: ставим карантин ─────────────────────────────────
          // Первые quarantine_duration_ секунд объект фильтруется с увеличенным
          // радиусом — это покрывает "pop-in" момент когда объект только появился
          // из-за препятствия и detector ещё не успел его распознать корректно.
          SpatialMemory nm;
          nm.last_seen         = scan_time;
          nm.first_seen        = scan_time;
          nm.x                 = cx;
          nm.y                 = cy;
          nm.vx                = vx;
          nm.vy                = vy;
          nm.radius            = r;
          nm.unstable_until    = scan_time + ros::Duration(1.0); // новый — всегда нестабилен
          nm.quarantine_until  = scan_time + ros::Duration(quarantine_duration_);
          nm.updated_this_frame = true;
          nm.id                = memory_id_counter_++;
          nm.tracker_id        = circle.id;
          memories_.push_back(nm);
          best_match = &memories_.back();
        }

        // Эффективный радиус фильтрации
        float effective_radius = r + std::min(speed * 0.3f, 0.4f);

        // Если объект в нестабильном состоянии — расширяем
        if (scan_time < new_unstable)
          effective_radius += 0.15f;

        // NEW: Если объект в карантине — УДВАИВАЕМ радиус фильтрации.
        // Это критично для pop-in случая: объект появился из-за стены,
        // а первые несколько сканов его позиция ещё неустойчива.
        if (best_match != nullptr && scan_time < best_match->quarantine_until)
          effective_radius *= (float)quarantine_radius_mult_;

        geometry_msgs::Point p_start = circle.center;
        geometry_msgs::Point p_end   = circle.center;
        if (is_moving) {
          float forward_time = (float)dt_msg + 0.15f;
          p_end.x += vx * forward_time;
          p_end.y += vy * forward_time;
        }

        float x1, y1, x2, y2;
        if (transformPointToLaser(p_start, obs_frame, laser_frame, scan_time, x1, y1) &&
            transformPointToLaser(p_end,   obs_frame, laser_frame, scan_time, x2, y2)) {
          targets.push_back({x1, y1, x2, y2, effective_radius});
          dynamic_points.emplace_back(x1, y1);
          if (is_moving) dynamic_points.emplace_back(x2, y2);
        }
      }
    }

    // ── Призраки ──────────────────────────────────────────────────────────────
    // NEW: когда объект пропадает из детектора (за препятствием), мы продолжаем
    // фильтровать его РАСШИРЕННУЮ зону — это предотвращает запись в costmap
    // при медленном исчезновении или ID-reassignment в трекере.
    for (auto it = memories_.begin(); it != memories_.end(); ) {
      if (!it->updated_this_frame) {
        double elapsed = std::max(0.0, (scan_time - it->last_seen).toSec());
        if (elapsed > ttl_duration_) {
          it = memories_.erase(it);
          continue;
        } else {
          geometry_msgs::Point p;
          p.x = it->x + it->vx * elapsed;
          p.y = it->y + it->vy * elapsed;
          float x1, y1;
          if (transformPointToLaser(p, obs_frame, laser_frame, scan_time, x1, y1)) {
            // NEW: Призраки фильтруются с raduis_buffer * 1.5 — чуть шире чем живые
            // объекты, чтобы перекрыть неточность экстраполяции
            float ghost_radius = it->radius + (float)radius_buffer_ * 1.5f;
            targets.push_back({x1, y1, x1, y1, ghost_radius});
            dynamic_points.emplace_back(x1, y1);
          }
        }
      }
      ++it;
    }

    // ── ЭТАП 2: ФИЛЬТРАЦИЯ СКАНА ──────────────────────────────────────────────

    sensor_msgs::LaserScan scan_out = *scan_in;
    float clear_val = scan_in->range_max + 0.5f;

    if (!targets.empty()) {
      std::vector<Cluster> clusters;
      Cluster current;
      float prev_x = 0, prev_y = 0;
      bool  prev_valid = false;

      auto finishCluster = [&]() {
        if (!current.indices.empty()) {
          clusters.push_back(current);
          current.indices.clear();
          current.xs.clear();
          current.ys.clear();
        }
      };

      for (size_t i = 0; i < scan_in->ranges.size(); ++i) {
        float r = scan_in->ranges[i];
        if (std::isnan(r) || std::isinf(r) ||
            r < scan_in->range_min || r > scan_in->range_max) {
          finishCluster();
          prev_valid = false;
          continue;
        }

        float angle = scan_in->angle_min + (float)i * scan_in->angle_increment;
        float x = r * cosf(angle);
        float y = r * sinf(angle);

        float adaptive_gap = (float)gap_threshold_ +
                             r * std::abs(scan_in->angle_increment) * 1.5f;
        if (prev_valid && std::hypot(x - prev_x, y - prev_y) > adaptive_gap)
          finishCluster();

        current.indices.push_back((int)i);
        current.xs.push_back(x);
        current.ys.push_back(y);
        prev_x = x;
        prev_y = y;
        prev_valid = true;
      }
      finishCluster();

      std::vector<bool> global_delete_mask(scan_out.ranges.size(), false);

      for (const auto &cluster : clusters) {
        float min_x = cluster.xs[0], max_x = cluster.xs[0];
        float min_y = cluster.ys[0], max_y = cluster.ys[0];
        std::vector<bool> local_mask(cluster.xs.size(), false);
        bool has_dynamic_points = false;

        for (size_t i = 0; i < cluster.xs.size(); ++i) {
          min_x = std::min(min_x, cluster.xs[i]);
          max_x = std::max(max_x, cluster.xs[i]);
          min_y = std::min(min_y, cluster.ys[i]);
          max_y = std::max(max_y, cluster.ys[i]);

          for (const auto &t : targets) {
            float dist = distToSegment(cluster.xs[i], cluster.ys[i],
                                       t.x1, t.y1, t.x2, t.y2);
            if (dist < t.radius + (float)radius_buffer_) {
              local_mask[i]     = true;
              has_dynamic_points = true;
              break;
            }
          }
        }

        if (!has_dynamic_points) continue;

        float cluster_span   = std::max(max_x - min_x, max_y - min_y);
        bool  is_huge_cluster = (cluster_span > (float)max_cluster_span_);

        if (is_huge_cluster) {
          // Большой кластер (стена + человек рядом): удаляем только маркированные
          // точки + небольшая дилатация по краям чтобы убрать фантомные пиксели
          for (size_t i = 0; i < cluster.xs.size(); ++i) {
            if (local_mask[i]) {
              int start_idx = std::max(0, (int)i - dilation_rays_);
              int end_idx   = std::min((int)cluster.xs.size()-1, (int)i + dilation_rays_);
              for (int j = start_idx; j <= end_idx; ++j)
                global_delete_mask[cluster.indices[j]] = true;
            }
          }
        } else {
          // Маленький кластер (человек отдельно): удаляем ВЕСЬ кластер
          for (int idx : cluster.indices)
            global_delete_mask[idx] = true;
        }
      }

      for (size_t i = 0; i < scan_out.ranges.size(); ++i) {
        if (global_delete_mask[i])
          scan_out.ranges[i] = clear_val;
      }
    }

    filtered_pub_.publish(scan_out);
    publishTrackedObstacles(scan_time, obs_frame);
    publishVisualization(targets, memories_, laser_frame, obs_frame, scan_time);
    publishDynamicMask(dynamic_points, scan_in, laser_frame, scan_time);
  }

  // ──────────────────────────────────────────────────────────── publishers ────

  void publishDynamicMask(const std::vector<std::pair<float,float>> &points,
                          const sensor_msgs::LaserScan::ConstPtr &scan_in,
                          const std::string &frame, const ros::Time &stamp) {
    if (points.empty()) return;
    sensor_msgs::PointCloud2 cloud;
    cloud.header.stamp   = stamp;
    cloud.header.frame_id = frame;
    cloud.height = 1;
    cloud.width  = points.size();
    cloud.is_dense     = false;
    cloud.is_bigendian = false;
    cloud.fields.resize(3);
    for (int i = 0; i < 3; ++i) {
      cloud.fields[i].datatype = sensor_msgs::PointField::FLOAT32;
      cloud.fields[i].count    = 1;
      cloud.fields[i].offset   = i * 4;
    }
    cloud.fields[0].name = "x";
    cloud.fields[1].name = "y";
    cloud.fields[2].name = "z";
    cloud.point_step = 12;
    cloud.row_step   = cloud.point_step * cloud.width;
    cloud.data.resize(cloud.row_step);
    for (size_t i = 0; i < points.size(); ++i) {
      float x = points[i].first, y = points[i].second, z = 0.0f;
      memcpy(&cloud.data[i*12+0], &x, 4);
      memcpy(&cloud.data[i*12+4], &y, 4);
      memcpy(&cloud.data[i*12+8], &z, 4);
    }
    dynamic_mask_pub_.publish(cloud);
  }

  void publishTrackedObstacles(const ros::Time &stamp, const std::string &frame) {
    tracked_obstacle_msgs::TrackedCircleArray msg;
    msg.header.stamp    = stamp;
    msg.header.frame_id = frame;

    for (const auto &m : memories_) {
      bool   is_ghost = !m.updated_this_frame;
      double elapsed  = is_ghost ? std::max(0.0, (stamp - m.last_seen).toSec()) : 0.0;

      tracked_obstacle_msgs::TrackedCircle tc;
      tc.center.x      = m.x + m.vx * (float)elapsed;
      tc.center.y      = m.y + m.vy * (float)elapsed;
      tc.center.z      = 0;
      tc.velocity.x    = m.vx;
      tc.velocity.y    = m.vy;
      tc.velocity.z    = 0;
      tc.radius        = m.radius;
      tc.id            = m.tracker_id;
      tc.is_ghost      = is_ghost;
      tc.is_unstable   = (stamp < m.unstable_until);
      tc.memory_id     = m.id;
      tc.time_since_seen = (float)elapsed;
      msg.circles.push_back(tc);
    }
    tracked_pub_.publish(msg);
  }

  bool transformPointToLaser(const geometry_msgs::Point &pt,
                             const std::string &from, const std::string &to,
                             const ros::Time &time, float &out_x, float &out_y) {
    geometry_msgs::PointStamped in, out;
    in.header.frame_id = from;
    in.header.stamp    = time;
    in.point = pt;
    try {
      if (tf_listener_.waitForTransform(to, from, time, ros::Duration(0.04))) {
        tf_listener_.transformPoint(to, in, out);
        out_x = (float)out.point.x;
        out_y = (float)out.point.y;
        return true;
      }
    } catch (tf::TransformException &) {}
    return false;
  }

  void publishVisualization(const std::vector<FilterTarget>   &targets,
                            const std::vector<SpatialMemory>  &memories,
                            const std::string &laser_frame,
                            const std::string &obs_frame,
                            ros::Time          stamp) {
    visualization_msgs::MarkerArray msg;
    visualization_msgs::Marker clear;
    clear.action = 3; // DELETEALL
    msg.markers.push_back(clear);

    int marker_id = 0;

    for (const auto &t : targets) {
      visualization_msgs::Marker m;
      m.header.frame_id = laser_frame;
      m.header.stamp    = stamp;
      m.ns              = "masks";
      m.id              = marker_id++;
      m.type            = visualization_msgs::Marker::SPHERE_LIST;
      m.action          = visualization_msgs::Marker::ADD;

      geometry_msgs::Point p1, p2;
      p1.x = t.x1; p1.y = t.y1; p1.z = 0;
      p2.x = t.x2; p2.y = t.y2; p2.z = 0;
      m.points.push_back(p1);
      m.points.push_back(p2);

      double limit = t.radius + radius_buffer_;
      m.scale.x = limit * 2.0;
      m.scale.y = limit * 2.0;
      m.scale.z = 0.1;
      m.color.r = 0.0; m.color.g = 1.0; m.color.b = 1.0; m.color.a = 0.3;
      msg.markers.push_back(m);
    }

    for (const auto &mem : memories) {
      visualization_msgs::Marker text;
      text.header.frame_id = obs_frame;
      text.header.stamp    = stamp;
      text.ns              = "db_state";
      text.id              = mem.id;
      text.type            = visualization_msgs::Marker::TEXT_VIEW_FACING;
      text.pose.position.x = mem.x;
      text.pose.position.y = mem.y;
      text.pose.position.z = 1.2;
      text.scale.z         = 0.4;
      text.color.a         = 1.0;

      bool in_quarantine = (stamp < mem.quarantine_until);

      if (!mem.updated_this_frame) {
        text.color.r = 0.6; text.color.g = 0.6; text.color.b = 0.6;
        text.text = "[GHOST] ID:" + std::to_string(mem.id);
      } else if (in_quarantine) {
        text.color.r = 1.0; text.color.g = 0.0; text.color.b = 1.0; // фиолетовый
        text.text = "[QUARANTINE] ID:" + std::to_string(mem.id);
      } else if (stamp < mem.unstable_until) {
        text.color.r = 1.0; text.color.g = 0.3; text.color.b = 0.0;
        text.text = "[UNSTABLE] ID:" + std::to_string(mem.id);
      } else {
        text.color.r = 0.0; text.color.g = 1.0; text.color.b = 0.0;
        text.text = "[OK] ID:" + std::to_string(mem.id);
      }
      msg.markers.push_back(text);
    }
    viz_pub_.publish(msg);
  }
};

int main(int argc, char **argv) {
  ros::init(argc, argv, "scan_cleaner_node");
  ScanCleaner cleaner;
  ros::spin();
  return 0;
}