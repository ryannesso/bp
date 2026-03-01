#include <algorithm>
#include <cmath>
#include <geometry_msgs/PointStamped.h>
#include <limits>
#include <map>
#include <obstacle_detector/Obstacles.h>
#include <ros/ros.h>
#include <sensor_msgs/LaserScan.h>
#include <set>
#include <string>
#include <tf/transform_listener.h>
#include <vector>

// Хранение истории объектов с учетом их скорости
struct DynamicObject {
  ros::Time last_seen;
  float x; // Координата в odom/map
  float y;
  float vx; // Скорость м/с
  float vy;
  float radius;
};

struct FilterTarget {
  float x; // Координата в системе координат лазера
  float y;
  float radius;
};

class ScanCleaner {
public:
  ScanCleaner() {
    ros::NodeHandle pnh("~");

    // Параметры - ОПТИМИЗИРОВАНЫ для стабильности без статики
    pnh.param("speed_threshold", speed_threshold_, 0.08); // м/с - НИЖЕ порог, чтобы не пропускать медленные объекты
    pnh.param("ttl_duration", ttl_duration_, 0.5);        // сек - МЕНЬШЕ время жизни призраков
    pnh.param("radius_buffer", radius_buffer_, 0.35);     // МЕНЬШЕ буффер для более точной очистки
    pnh.param("proximity_threshold", proximity_threshold_, 0.3); // расстояние для определения "соседства"

    // Подписки
    scan_sub_ = nh_.subscribe("/scan", 1, &ScanCleaner::scanCallback, this);
    obstacles_sub_ =
        nh_.subscribe("/obstacles", 1, &ScanCleaner::obstaclesCallback, this);
    filtered_pub_ = nh_.advertise<sensor_msgs::LaserScan>("/scan_cleaned", 1);

    has_obstacles_ = false;
    ROS_INFO("ScanCleaner: Node initialized. Speed Thresh: %.2f, Buffer: %.2f, TTL: %.2f",
             speed_threshold_, radius_buffer_, ttl_duration_);
  }

private:
  ros::NodeHandle nh_;
  ros::Subscriber scan_sub_;
  ros::Subscriber obstacles_sub_;
  ros::Publisher filtered_pub_;
  tf::TransformListener tf_listener_;

  double speed_threshold_;
  double ttl_duration_;
  double radius_buffer_;
  double proximity_threshold_; // Новый параметр для проверки соседства

  obstacle_detector::Obstacles latest_obs_msg_;
  std::map<int32_t, DynamicObject> object_db_;
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

    // Вычисляем задержку данных трекера относительно скана
    double dt_msg = (scan_time - latest_obs_msg_.header.stamp).toSec();
    if (dt_msg < 0)
      dt_msg = 0; // На случай если скан старее (маловероятно в Gazebo)

    std::set<int32_t> seen_now;
    std::vector<FilterTarget> targets;

    // ЭТАП 1: Обновление базы данных и предсказание положения
    for (const auto &circle : latest_obs_msg_.circles) {
      float speed = std::hypot(circle.velocity.x, circle.velocity.y);
      int32_t id = circle.id;

      bool is_moving = (speed > speed_threshold_);
      bool is_known = (id != 0 && object_db_.count(id));
      bool is_nearby_dynamic = false;

      // Если объект стоит и новый (id сменился?), проверяем, не был ли он
      // динамическим недавно
      if (!is_moving && !is_known) {
        for (const auto &entry : object_db_) {
          float dx = circle.center.x - entry.second.x;
          float dy = circle.center.y - entry.second.y;
          // Если объект в пределах 0.5м от известного динамического - считаем
          // его тем же
          if (dx * dx + dy * dy < 0.25) {
            is_nearby_dynamic = true;
            break;
          }
        }
      }

      // Если объект движется, ИЛИ он уже был динамическим, ИЛИ он появился на
      // месте динамического
      if (is_moving || is_known || is_nearby_dynamic) {

        if (id != 0) {
          seen_now.insert(id);
          // Обновляем/Добавляем состояние в базе
          // Даже если он стоит (v=0), мы обновляем запись, чтобы продлить ему
          // жизнь (TTL)
          object_db_[id] = {scan_time,
                            (float)circle.center.x,
                            (float)circle.center.y,
                            (float)circle.velocity.x,
                            (float)circle.velocity.y,
                            (float)circle.radius};
        }

        // Линейная экстраполяция
        geometry_msgs::Point p_pred;
        p_pred.x = circle.center.x + circle.velocity.x * dt_msg;
        p_pred.y = circle.center.y + circle.velocity.y * dt_msg;
        p_pred.z = 0;

        targets.push_back(transformToLaser(p_pred, obs_frame, laser_frame,
                                           scan_time, circle.radius));
      }
    }

    // ЭТАП 2: Работа с "призраками" (объекты, которые трекер потерял на
    // кадр-два)
    for (auto it = object_db_.begin(); it != object_db_.end();) {
      if (seen_now.count(it->first)) {
        ++it;
        continue;
      }

      double elapsed = (scan_time - it->second.last_seen).toSec();
      if (elapsed > ttl_duration_) {
        it = object_db_.erase(it);
      } else {
        // Двигаем "призрак" по инерции
        geometry_msgs::Point p_ghost;
        p_ghost.x = it->second.x + it->second.vx * elapsed;
        p_ghost.y = it->second.y + it->second.vy * elapsed;

        targets.push_back(transformToLaser(p_ghost, obs_frame, laser_frame,
                                           scan_time, it->second.radius));
        ++it;
      }
    }

    // ЭТАП 3: Очистка лучей скана
    sensor_msgs::LaserScan scan_out = *scan_in;
    // Значение, которое гарантированно больше range_max, чтобы SLAM "очистил"
    // клетку
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
          float dx = px - t.x;
          float dy = py - t.y;
          float dist_sq = dx * dx + dy * dy;
          float limit = t.radius + radius_buffer_;

          if (dist_sq < limit * limit) {
            scan_out.ranges[i] = clear_val;
            break;
          }
        }
      }
    }
    filtered_pub_.publish(scan_out);
  }

  // Трансформация точки с учетом времени скана (Time Travel Transform)
  FilterTarget transformToLaser(const geometry_msgs::Point &pt,
                                const std::string &from, const std::string &to,
                                const ros::Time &time, float radius) {

    geometry_msgs::PointStamped in, out;
    in.header.frame_id = from;
    in.header.stamp = time; // Ищем положение именно в момент скана
    in.point = pt;

    try {
      // Ждем трансформацию 50мс (важно для Noetic/Gazebo)
      if (tf_listener_.waitForTransform(to, from, time, ros::Duration(0.05))) {
        tf_listener_.transformPoint(to, in, out);
        return {(float)out.point.x, (float)out.point.y, radius};
      } else {
        // Если не успела, пробуем по Time(0)
        tf_listener_.transformPoint(to, in, out);
        return {(float)out.point.x, (float)out.point.y, radius};
      }
    } catch (tf::TransformException &ex) {
      ROS_WARN_THROTTLE(2, "TF Error: %s", ex.what());
      return {999.0f, 999.0f,
              radius}; // Уносим маску далеко, чтобы не вырезать лишнего
    }
  }
};

int main(int argc, char **argv) {
  ros::init(argc, argv, "scan_cleaner_node");
  ScanCleaner cleaner;
  ros::spin();
  return 0;
}