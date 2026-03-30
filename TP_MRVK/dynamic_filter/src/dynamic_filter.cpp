#include <algorithm>
#include <cmath>
#include <geometry_msgs/PointStamped.h>
#include <limits>
#include <obstacle_detector/Obstacles.h>
#include <ros/ros.h>
#include <sensor_msgs/LaserScan.h>
#include <string>
#include <tf/transform_listener.h>
#include <vector>
#include <visualization_msgs/MarkerArray.h>

// НОВАЯ СТРУКТУРА: Пространственная память (не зависит от ID трекера)
struct SpatialMemory {
  ros::Time last_seen;
  float x;
  float y;
  float vx;
  float vy;
  float radius;
  ros::Time unstable_until; // Таймер брони
  bool updated_this_frame;  // Флаг обновления
  int id;                   // Внутренний ID только для визуализации
};

struct FilterTarget {
  float x;
  float y;
  float radius;
};

class ScanCleaner {
public:
  ScanCleaner() {
    ros::NodeHandle pnh("~");

    pnh.param("speed_threshold", speed_threshold_, 0.08);
    pnh.param("ttl_duration", ttl_duration_, 1.5);
    pnh.param("radius_buffer", radius_buffer_, 0.4);
    pnh.param("proximity_threshold", proximity_threshold_, 0.3);
    pnh.param("max_tilt_angle", max_tilt_angle_, 0.08);

    scan_sub_ = nh_.subscribe("/scan", 1, &ScanCleaner::scanCallback, this);
    obstacles_sub_ =
        nh_.subscribe("/obstacles", 1, &ScanCleaner::obstaclesCallback, this);
    filtered_pub_ = nh_.advertise<sensor_msgs::LaserScan>("/scan_cleaned", 1);
    viz_pub_ =
        nh_.advertise<visualization_msgs::MarkerArray>("/scan_cleaner_viz", 1);

    has_obstacles_ = false;
    memory_id_counter_ = 1000;
    ROS_INFO("ScanCleaner: Spatial Memory tracker initialized.");
  }

private:
  ros::NodeHandle nh_;
  ros::Subscriber scan_sub_;
  ros::Subscriber obstacles_sub_;
  ros::Publisher filtered_pub_;
  ros::Publisher viz_pub_;
  tf::TransformListener tf_listener_;

  double speed_threshold_;
  double ttl_duration_;
  double radius_buffer_;
  double proximity_threshold_;
  double max_tilt_angle_;

  obstacle_detector::Obstacles latest_obs_msg_;
  std::vector<SpatialMemory> memories_; // База данных памяти
  int memory_id_counter_;
  bool has_obstacles_;

  void obstaclesCallback(const obstacle_detector::Obstacles::ConstPtr &msg) {
    latest_obs_msg_ = *msg;
    has_obstacles_ = true;
  }

  void scanCallback(const sensor_msgs::LaserScan::ConstPtr &scan_in) {
    if (!has_obstacles_) {
      filtered_pub_.publish(scan_in);
      return;
    }

    ros::Time scan_time = scan_in->header.stamp;
    std::string laser_frame = scan_in->header.frame_id;
    std::string obs_frame = latest_obs_msg_.header.frame_id;

    double dt_msg = (scan_time - latest_obs_msg_.header.stamp).toSec();
    if (dt_msg < 0)
      dt_msg = 0;
    if (dt_msg > 0.5)
      dt_msg = 0.5;

    std::vector<FilterTarget> targets;

    // Сбрасываем флаги
    for (auto &m : memories_)
      m.updated_this_frame = false;

    // ==========================================================
    // ЭТАП 1: СОПОСТАВЛЕНИЕ КРУГОВ С НАШЕЙ ПАМЯТЬЮ
    // ==========================================================
    for (const auto &circle : latest_obs_msg_.circles) {
      // Защита от битых NaN данных из трекера
      float vx = std::isnan(circle.velocity.x) ? 0.0f : circle.velocity.x;
      float vy = std::isnan(circle.velocity.y) ? 0.0f : circle.velocity.y;
      float r = std::isnan(circle.radius) ? 0.3f : circle.radius;
      float cx = circle.center.x;
      float cy = circle.center.y;

      float speed = std::hypot(vx, vy);
      bool is_moving = (speed > speed_threshold_);

      SpatialMemory *best_match = nullptr;
      float best_dist = -1;

      // Ищем, не было ли тут динамики недавно
      for (auto &m : memories_) {
        if (m.updated_this_frame)
          continue;

        float dx = cx - m.x;
        float dy = cy - m.y;

        // Также проверяем дистанцию до призрака (куда он мог улететь)
        double dt = (scan_time - m.last_seen).toSec();
        if (dt < 0)
          dt = 0;
        float gx = m.x + m.vx * dt;
        float gy = m.y + m.vy * dt;
        float dx_g = cx - gx;
        float dy_g = cy - gy;

        float dist_sq = std::min(dx * dx + dy * dy, dx_g * dx_g + dy_g * dy_g);
        float match_limit = proximity_threshold_ + r + m.radius;

        if (dist_sq < match_limit * match_limit) {
          if (best_match == nullptr || dist_sq < best_dist) {
            best_match = &m;
            best_dist = dist_sq;
          }
        }
      }

      // Если объект движется сейчас, ИЛИ он стоит, но мы помним его динамику
      if (best_match != nullptr || is_moving) {
        ros::Time new_unstable = ros::Time(0);

        if (best_match != nullptr) {
          new_unstable = best_match->unstable_until;
          float old_speed = std::hypot(best_match->vx, best_match->vy);

          // ДЕТЕКТОРЫ ПРОБЛЕМ (Остановка, разворот, старт)
          bool stopped = !is_moving;
          bool reversed = false;
          if (old_speed > 0.05 && speed > 0.05) {
            if (best_match->vx * vx + best_match->vy * vy < 0)
              reversed = true;
          }
          bool accelerated =
              (old_speed <= speed_threshold_ && speed > speed_threshold_);

          // Если что-то из этого случилось - даем броню на 1 секунду!
          if (stopped || reversed || accelerated) {
            new_unstable = scan_time + ros::Duration(1.0);
          }

          // Обновляем память
          best_match->last_seen = scan_time;
          best_match->x = cx;
          best_match->y = cy;
          best_match->vx = vx;
          best_match->vy = vy;
          best_match->radius = r;
          best_match->unstable_until = new_unstable;
          best_match->updated_this_frame = true;
        } else {
          // Записываем новый объект в память
          SpatialMemory nm;
          nm.last_seen = scan_time;
          nm.x = cx;
          nm.y = cy;
          nm.vx = vx;
          nm.vy = vy;
          nm.radius = r;
          nm.unstable_until = ros::Time(0);
          nm.updated_this_frame = true;
          nm.id = memory_id_counter_++;
          memories_.push_back(nm);
        }

        // --- РАСЧЕТ РАДИУСА И ДОБАВЛЕНИЕ МАСОК ---
        float effective_radius = r;
        if (scan_time < new_unstable) {
          effective_radius += 0.1; // Включаем жирную маску
        }

        // Центр
        targets.push_back(transformToLaser(circle.center, obs_frame,
                                           laser_frame, scan_time,
                                           effective_radius));

        // Капсула
        if (is_moving) {
          geometry_msgs::Point p;
          p.x = cx + vx * (dt_msg + 0.15);
          p.y = cy + vy * (dt_msg + 0.15);
          p.z = 0;
          targets.push_back(transformToLaser(p, obs_frame, laser_frame,
                                             scan_time, effective_radius));
        }
      }
    }

    // ==========================================================
    // ЭТАП 2: РАБОТА С ПРИЗРАКАМИ
    // ==========================================================
    for (auto it = memories_.begin(); it != memories_.end();) {
      if (!it->updated_this_frame) {
        double elapsed = (scan_time - it->last_seen).toSec();
        if (elapsed < 0)
          elapsed = 0;

        if (elapsed > ttl_duration_) {
          it = memories_.erase(it);
          continue;
        } else {
          geometry_msgs::Point p;
          p.x = it->x + it->vx * elapsed;
          p.y = it->y + it->vy * elapsed;
          p.z = 0;
          targets.push_back(transformToLaser(p, obs_frame, laser_frame,
                                             scan_time, it->radius));
        }
      }
      ++it;
    }

    // ВИЗУАЛИЗАЦИЯ
    publishVisualization(targets, memories_, laser_frame, obs_frame, scan_time);

    // ==========================================================
    // ЭТАП 3: ФИЛЬТР НАКЛОНА И ОЧИСТКА СКАНА
    // ==========================================================
    try {
      tf::StampedTransform odom_to_base;
      tf_listener_.lookupTransform("odom", "base_link", ros::Time(0),
                                   odom_to_base);
      double roll, pitch, yaw;
      odom_to_base.getBasis().getRPY(roll, pitch, yaw);

      if (std::abs(roll) > max_tilt_angle_ ||
          std::abs(pitch) > max_tilt_angle_) {
        sensor_msgs::LaserScan empty_scan = *scan_in;
        for (size_t i = 0; i < empty_scan.ranges.size(); ++i) {
          empty_scan.ranges[i] = std::numeric_limits<float>::quiet_NaN();
        }
        filtered_pub_.publish(empty_scan);
        return;
      }
    } catch (tf::TransformException &ex) {
    }

    sensor_msgs::LaserScan scan_out = *scan_in;
    float clear_val = scan_in->range_max + 0.5;

    if (!targets.empty()) {
      for (size_t i = 0; i < scan_out.ranges.size(); ++i) {
        float r = scan_out.ranges[i];
        if (std::isnan(r) || std::isinf(r))
          continue;

        float angle = scan_out.angle_min + i * scan_out.angle_increment;
        float px = r * std::cos(angle);
        float py = r * std::sin(angle);

        for (const auto &t : targets) {
          if (t.x == 999.0f)
            continue;
          float dx = px - t.x;
          float dy = py - t.y;
          float limit = t.radius + radius_buffer_;
          if (dx * dx + dy * dy < limit * limit) {
            scan_out.ranges[i] = clear_val;
            break;
          }
        }
      }
    }
    filtered_pub_.publish(scan_out);
  }

  // СУПЕР БЫСТРАЯ ТРАНСФОРМАЦИЯ (БЕЗ ЗАВИСАНИЙ И МОЛЬБЫ О TF)
  FilterTarget transformToLaser(const geometry_msgs::Point &pt,
                                const std::string &from, const std::string &to,
                                const ros::Time &time, float radius) {
    geometry_msgs::PointStamped in, out;
    in.header.frame_id = from;
    in.header.stamp = ros::Time(
        0); // МГНОВЕННО БЕРЕМ ПОСЛЕДНЮЮ ПОЗИЦИЮ (убирает потерю сканов)
    in.point = pt;

    try {
      tf_listener_.transformPoint(to, in, out);
      return {(float)out.point.x, (float)out.point.y, radius};
    } catch (tf::TransformException &ex) {
      return {999.0f, 999.0f, radius};
    }
  }

  void publishVisualization(const std::vector<FilterTarget> &targets,
                            const std::vector<SpatialMemory> &memories,
                            const std::string &laser_frame,
                            const std::string &obs_frame, ros::Time stamp) {
    visualization_msgs::MarkerArray msg;

    visualization_msgs::Marker clear;
    clear.action = 3;
    msg.markers.push_back(clear);

    int marker_id = 0;
    for (const auto &t : targets) {
      if (t.x == 999.0f)
        continue;

      visualization_msgs::Marker m;
      m.header.frame_id = laser_frame;
      m.header.stamp = stamp;
      m.ns = "masks";
      m.id = marker_id++;
      m.type = visualization_msgs::Marker::CYLINDER;
      m.pose.position.x = t.x;
      m.pose.position.y = t.y;
      m.pose.position.z = 0.0;
      m.pose.orientation.w = 1.0;

      double limit = t.radius + radius_buffer_;
      m.scale.x = limit * 2.0;
      m.scale.y = limit * 2.0;
      m.scale.z = 0.1;

      m.color.r = 0.0;
      m.color.g = 1.0;
      m.color.b = 1.0;
      m.color.a = 0.4;
      msg.markers.push_back(m);
    }

    for (const auto &m : memories) {
      bool is_ghost = !m.updated_this_frame;
      bool is_unstable = (stamp < m.unstable_until);

      visualization_msgs::Marker text;
      text.header.frame_id = obs_frame;
      text.header.stamp = stamp;
      text.ns = "db_state";
      text.id = m.id;
      text.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
      text.pose.position.x = m.x;
      text.pose.position.y = m.y;
      text.pose.position.z = 1.2;
      text.scale.z = 0.4;
      text.color.a = 1.0;

      std::string state_str;
      if (is_ghost) {
        text.color.r = 0.6;
        text.color.g = 0.6;
        text.color.b = 0.6;
        state_str = "[GHOST] ";
      } else if (is_unstable) {
        text.color.r = 1.0;
        text.color.g = 0.3;
        text.color.b = 0.0;
        state_str = "[UNSTABLE] ";
      } else {
        text.color.r = 0.0;
        text.color.g = 1.0;
        text.color.b = 0.0;
        state_str = "[OK] ";
      }
      text.text = state_str + "MemID: " + std::to_string(m.id);
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