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

// =============================================================================
// SPATIAL MEMORY: Пространственная память (независима от ID трекера)
// Каждый объект живёт здесь, пока не истечёт TTL.
// Состояния:
//   ACTIVE    — объект виден в текущем кадре, скорость стабильна
//   UNSTABLE  — объект резко изменил поведение (остановка/разворот/старт)
//               выдаётся "броня" на 1 секунду с расширенным радиусом маски
//   GHOST     — объект пропал из скана, но позиция предсказывается по скорости
//               живёт до ttl_duration_ секунд, затем удаляется
// =============================================================================
struct SpatialMemory {
  ros::Time last_seen;
  float x;
  float y;
  float vx;
  float vy;
  float radius;
  ros::Time unstable_until; // До этого времени — UNSTABLE
  bool updated_this_frame;  // Служебный флаг сопоставления за кадр
  int id;                   // Внутренний ID (только для визуализации)
  int tracker_id;           // Последний известный ID из obstacle_detector
};

struct FilterTarget {
  float x;
  float y;
  float radius;
};

// =============================================================================
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

    // -------------------------------------------------------------------------
    // НОВЫЙ ПАБЛИШЕР: Обогащённые данные об объектах для DWA и других
    // потребителей Топик: /obstacles_tracked Тип:
    // tracked_obstacle_msgs/TrackedCircleArray Содержит: предсказанные позиции,
    // флаги is_ghost / is_unstable,
    //           time_since_seen, memory_id
    // -------------------------------------------------------------------------
    tracked_pub_ = nh_.advertise<tracked_obstacle_msgs::TrackedCircleArray>(
        "/obstacles_tracked", 1);

    has_obstacles_ = false;
    memory_id_counter_ = 1000;
    ROS_INFO("ScanCleaner: Spatial Memory tracker initialized. "
             "Publishing enriched obstacles on /obstacles_tracked");
  }

private:
  ros::NodeHandle nh_;
  ros::Subscriber scan_sub_;
  ros::Subscriber obstacles_sub_;
  ros::Publisher filtered_pub_;
  ros::Publisher viz_pub_;
  ros::Publisher tracked_pub_; // ← НОВЫЙ
  tf::TransformListener tf_listener_;

  double speed_threshold_;
  double ttl_duration_;
  double radius_buffer_;
  double proximity_threshold_;
  double max_tilt_angle_;

  obstacle_detector::Obstacles latest_obs_msg_;
  std::vector<SpatialMemory> memories_;
  int memory_id_counter_;
  bool has_obstacles_;

  // ---------------------------------------------------------------------------
  void obstaclesCallback(const obstacle_detector::Obstacles::ConstPtr &msg) {
    latest_obs_msg_ = *msg;
    has_obstacles_ = true;
  }

  // ---------------------------------------------------------------------------
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

    // Сбрасываем флаги перед сопоставлением
    for (auto &m : memories_)
      m.updated_this_frame = false;

    // =========================================================================
    // ЭТАП 1: СОПОСТАВЛЕНИЕ КРУГОВ С ПАМЯТЬЮ
    // =========================================================================
    for (const auto &circle : latest_obs_msg_.circles) {
      // Защита от NaN из трекера
      float vx =
          std::isnan(circle.velocity.x) ? 0.0f : (float)circle.velocity.x;
      float vy =
          std::isnan(circle.velocity.y) ? 0.0f : (float)circle.velocity.y;
      float r = std::isnan(circle.radius) ? 0.3f : (float)circle.radius;
      float cx = (float)circle.center.x;
      float cy = (float)circle.center.y;

      float speed = std::hypot(vx, vy);
      bool is_moving = (speed > (float)speed_threshold_);

      SpatialMemory *best_match = nullptr;
      float best_dist = -1.f;

      // Ищем ближайшую запись в памяти (с учётом "призрака" — предсказанной
      // позиции)
      for (auto &m : memories_) {
        if (m.updated_this_frame)
          continue;

        double dt_mem = (scan_time - m.last_seen).toSec();
        if (dt_mem < 0)
          dt_mem = 0;

        // Проверяем и текущую позицию из трекера, и предсказанную позицию
        // "призрака"
        float gx = m.x + m.vx * (float)dt_mem;
        float gy = m.y + m.vy * (float)dt_mem;

        float dx1 = cx - m.x, dy1 = cy - m.y;
        float dx2 = cx - gx, dy2 = cy - gy;
        float dist_sq = std::min(dx1 * dx1 + dy1 * dy1, dx2 * dx2 + dy2 * dy2);

        float match_limit = (float)proximity_threshold_ + r + m.radius;

        if (dist_sq < match_limit * match_limit) {
          if (best_match == nullptr || dist_sq < best_dist) {
            best_match = &m;
            best_dist = dist_sq;
          }
        }
      }

      if (best_match != nullptr || is_moving) {
        ros::Time new_unstable = ros::Time(0);

        if (best_match != nullptr) {
          new_unstable = best_match->unstable_until;
          float old_speed = std::hypot(best_match->vx, best_match->vy);

          // Детекция нестабильного поведения
          bool stopped = !is_moving;
          bool reversed = false;
          bool accelerated =
              (old_speed <= (float)speed_threshold_ && is_moving);

          if (old_speed > 0.05f && speed > 0.05f) {
            if (best_match->vx * vx + best_match->vy * vy < 0)
              reversed = true;
          }

          if (stopped || reversed || accelerated) {
            new_unstable = scan_time + ros::Duration(1.0);
            ROS_DEBUG_THROTTLE(0.5,
                               "ScanCleaner: MemID=%d -> UNSTABLE "
                               "(stopped=%d reversed=%d accel=%d)",
                               best_match->id, stopped, reversed, accelerated);
          }

          // Обновляем запись
          best_match->last_seen = scan_time;
          best_match->x = cx;
          best_match->y = cy;
          best_match->vx = vx;
          best_match->vy = vy;
          best_match->radius = r;
          best_match->unstable_until = new_unstable;
          best_match->tracker_id = circle.id;
          best_match->updated_this_frame = true;

        } else {
          // Новый объект — добавляем в память
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
          nm.tracker_id = circle.id;
          memories_.push_back(nm);
        }

        // Радиус маски: расширяем во время нестабильности
        float effective_radius = r;
        if (scan_time < new_unstable)
          effective_radius += 0.1f;

        // Маска на текущую позицию объекта
        targets.push_back(transformToLaser(circle.center, obs_frame,
                                           laser_frame, scan_time,
                                           effective_radius));

        // Капсула вперёд по скорости (компенсация задержки)
        if (is_moving) {
          geometry_msgs::Point p;
          p.x = cx + vx * (float)(dt_msg + 0.15);
          p.y = cy + vy * (float)(dt_msg + 0.15);
          p.z = 0;
          targets.push_back(transformToLaser(p, obs_frame, laser_frame,
                                             scan_time, effective_radius));
        }
      }
    }

    // =========================================================================
    // ЭТАП 2: РАБОТА С ПРИЗРАКАМИ (объект исчез из скана, предсказываем
    // позицию)
    // =========================================================================
    for (auto it = memories_.begin(); it != memories_.end();) {
      if (!it->updated_this_frame) {
        double elapsed = (scan_time - it->last_seen).toSec();
        if (elapsed < 0)
          elapsed = 0;

        if (elapsed > ttl_duration_) {
          it = memories_.erase(it);
          continue;
        } else {
          // Предсказываем куда ушёл объект
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

    // Визуализация масок и состояний памяти
    publishVisualization(targets, memories_, laser_frame, obs_frame, scan_time);

    // =========================================================================
    // ЭТАП 2.5: ПУБЛИКАЦИЯ /obstacles_tracked
    // Все живые записи памяти → TrackedCircleArray для DWA и других
    // потребителей
    // =========================================================================
    publishTrackedObstacles(scan_time, obs_frame);

    // =========================================================================
    // ЭТАП 3: ФИЛЬТР НАКЛОНА РОБОТА
    // Если робот сильно наклонён — очищаем весь скан, чтобы не писать мусор
    // =========================================================================
    try {
      tf::StampedTransform odom_to_base;
      tf_listener_.lookupTransform("odom", "base_link", ros::Time(0),
                                   odom_to_base);
      double roll, pitch, yaw;
      odom_to_base.getBasis().getRPY(roll, pitch, yaw);

      if (std::abs(roll) > max_tilt_angle_ ||
          std::abs(pitch) > max_tilt_angle_) {
        sensor_msgs::LaserScan empty_scan = *scan_in;
        for (size_t i = 0; i < empty_scan.ranges.size(); ++i)
          empty_scan.ranges[i] = std::numeric_limits<float>::quiet_NaN();
        filtered_pub_.publish(empty_scan);
        return;
      }
    } catch (tf::TransformException &) {
    }

    // =========================================================================
    // ЭТАП 4: УДАЛЕНИЕ ТОЧЕК СКАНА ВНУТРИ МАСОК
    // =========================================================================
    sensor_msgs::LaserScan scan_out = *scan_in;
    float clear_val = scan_in->range_max + 0.5f;

    if (!targets.empty()) {
      for (size_t i = 0; i < scan_out.ranges.size(); ++i) {
        float range = scan_out.ranges[i];
        if (std::isnan(range) || std::isinf(range))
          continue;

        float angle = scan_out.angle_min + (float)i * scan_out.angle_increment;
        float px = range * std::cos(angle);
        float py = range * std::sin(angle);

        for (const auto &t : targets) {
          if (t.x == 999.0f)
            continue;
          float dx = px - t.x;
          float dy = py - t.y;
          float limit = t.radius + (float)radius_buffer_;
          if (dx * dx + dy * dy < limit * limit) {
            scan_out.ranges[i] = clear_val;
            break;
          }
        }
      }
    }

    filtered_pub_.publish(scan_out);
  }

  // ---------------------------------------------------------------------------
  // Публикует TrackedCircleArray для всех подписчиков (DWA, отладка и т.д.)
  // ---------------------------------------------------------------------------
  void publishTrackedObstacles(const ros::Time &stamp,
                               const std::string &frame) {
    tracked_obstacle_msgs::TrackedCircleArray msg;
    msg.header.stamp = stamp;
    msg.header.frame_id = frame;

    for (const auto &m : memories_) {
      bool is_ghost = !m.updated_this_frame;
      bool is_unstable = (stamp < m.unstable_until);

      double elapsed = is_ghost ? (stamp - m.last_seen).toSec() : 0.0;
      if (elapsed < 0)
        elapsed = 0;

      tracked_obstacle_msgs::TrackedCircle tc;
      // Позиция: если призрак — предсказываем по последней скорости
      tc.center.x = m.x + m.vx * (float)elapsed;
      tc.center.y = m.y + m.vy * (float)elapsed;
      tc.center.z = 0;
      tc.velocity.x = m.vx;
      tc.velocity.y = m.vy;
      tc.velocity.z = 0;
      tc.radius = m.radius;
      tc.id = m.tracker_id;
      tc.is_ghost = is_ghost;
      tc.is_unstable = is_unstable;
      tc.memory_id = m.id;
      tc.time_since_seen = (float)elapsed;

      msg.circles.push_back(tc);
    }

    tracked_pub_.publish(msg);
  }

  // ---------------------------------------------------------------------------
  // Быстрая трансформация точки из системы препятствий в систему лазера
  // ---------------------------------------------------------------------------
  FilterTarget transformToLaser(const geometry_msgs::Point &pt,
                                const std::string &from, const std::string &to,
                                const ros::Time &time, float radius) {
    geometry_msgs::PointStamped in, out;
    in.header.frame_id = from;
    in.header.stamp = ros::Time(0); // Берём последнюю доступную трансформацию
    in.point = pt;

    try {
      tf_listener_.transformPoint(to, in, out);
      return {(float)out.point.x, (float)out.point.y, radius};
    } catch (tf::TransformException &) {
      return {999.0f, 999.0f, radius};
    }
  }

  // ---------------------------------------------------------------------------
  // Визуализация масок и состояний памяти в RViz
  // ---------------------------------------------------------------------------
  void publishVisualization(const std::vector<FilterTarget> &targets,
                            const std::vector<SpatialMemory> &memories,
                            const std::string &laser_frame,
                            const std::string &obs_frame, ros::Time stamp) {
    visualization_msgs::MarkerArray msg;
    visualization_msgs::Marker clear;
    clear.action = 3;
    msg.markers.push_back(clear);

    int marker_id = 0;

    // Маски (цилиндры в системе лазера)
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

    // Текстовые метки состояний памяти
    for (const auto &mem : memories) {
      bool is_ghost = !mem.updated_this_frame;
      bool is_unstable = (stamp < mem.unstable_until);

      visualization_msgs::Marker text;
      text.header.frame_id = obs_frame;
      text.header.stamp = stamp;
      text.ns = "db_state";
      text.id = mem.id;
      text.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
      text.pose.position.x = mem.x;
      text.pose.position.y = mem.y;
      text.pose.position.z = 1.2;
      text.scale.z = 0.4;
      text.color.a = 1.0;

      if (is_ghost) {
        text.color.r = 0.6;
        text.color.g = 0.6;
        text.color.b = 0.6;
        double elapsed = (stamp - mem.last_seen).toSec();
        text.text = "[GHOST " + std::to_string((int)(elapsed * 10) / 10.0) +
                    "s] "
                    "MemID:" +
                    std::to_string(mem.id);
      } else if (is_unstable) {
        text.color.r = 1.0;
        text.color.g = 0.3;
        text.color.b = 0.0;
        text.text = "[UNSTABLE] MemID:" + std::to_string(mem.id);
      } else {
        text.color.r = 0.0;
        text.color.g = 1.0;
        text.color.b = 0.0;
        text.text = "[OK] MemID:" + std::to_string(mem.id);
      }
      msg.markers.push_back(text);
    }

    viz_pub_.publish(msg);
  }
};

// =============================================================================
int main(int argc, char **argv) {
  ros::init(argc, argv, "scan_cleaner_node");
  ScanCleaner cleaner;
  ros::spin();
  return 0;
}
