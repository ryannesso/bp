#include <cv_bridge/cv_bridge.h>
#include <image_geometry/pinhole_camera_model.h>
#include <limits>
#include <ros/ros.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/LaserScan.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_ros/transform_listener.h>

class ScanMaskFilter {
  ros::NodeHandle nh_;
  ros::Subscriber scan_sub_;
  ros::Subscriber mask_sub_;
  ros::Subscriber info_sub_;
  ros::Publisher scan_pub_;

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
  image_geometry::PinholeCameraModel cam_model_;

  cv::Mat latest_mask_;
  bool has_cam_info_ = false;

public:
  ScanMaskFilter() : tf_listener_(tf_buffer_) {
    // Подписки: скан с лидара, маска с твоей детекции, инфо о камере
    scan_sub_ =
        nh_.subscribe("/scan_filtered", 10, &ScanMaskFilter::scanCb, this);
    mask_sub_ = nh_.subscribe("/path_detector/mask", 1, &ScanMaskFilter::maskCb, this);
    info_sub_ =
        nh_.subscribe("/camera/camera_info", 1, &ScanMaskFilter::infoCb, this);

    // Паблишер очищенного скана для SLAM
    scan_pub_ = nh_.advertise<sensor_msgs::LaserScan>("/scan_for_slam", 10);
  }

  // Получаем параметры камеры (чтобы знать фокусное расстояние и искажения)
  void infoCb(const sensor_msgs::CameraInfoConstPtr &msg) {
    cam_model_.fromCameraInfo(msg);
    has_cam_info_ = true;
  }

  // Получаем твою черно-белую маску (Белое - дорога, Черное - трава)
  void maskCb(const sensor_msgs::ImageConstPtr &msg) {
    try {
      latest_mask_ = cv_bridge::toCvCopy(msg, "mono8")->image;
    } catch (cv_bridge::Exception &e) {
      ROS_ERROR("cv_bridge exception: %s", e.what());
    }
  }

  void scanCb(const sensor_msgs::LaserScanConstPtr &scan_msg) {
    // Если нет данных с камеры, просто пропускаем скан дальше
    if (!has_cam_info_ || latest_mask_.empty()) {
      scan_pub_.publish(scan_msg);
      return;
    }

    sensor_msgs::LaserScan filtered_scan = *scan_msg;
    geometry_msgs::TransformStamped transformStamped;

    try {
      // Ищем трансформацию от лидара до камеры (tf_tree должен быть настроен)
      transformStamped = tf_buffer_.lookupTransform(
          cam_model_.tfFrame(), scan_msg->header.frame_id,
          scan_msg->header.stamp, ros::Duration(0.1));
    } catch (tf2::TransformException &ex) {
      ROS_WARN_THROTTLE(2.0, "TF Error: %s", ex.what());
      scan_pub_.publish(filtered_scan);
      return;
    }

    // Проходим по всем лучам лидара
    for (size_t i = 0; i < scan_msg->ranges.size(); ++i) {
      float range = scan_msg->ranges[i];

      // Если луч бьет в бесконечность или слишком близко - игнорируем
      if (!std::isfinite(range) || range < scan_msg->range_min ||
          range > scan_msg->range_max) {
        continue;
      }

      // 1. Превращаем дальность луча в 3D точку (X, Y) в координатах лидара
      float angle = scan_msg->angle_min + i * scan_msg->angle_increment;
      geometry_msgs::PointStamped pt_laser;
      pt_laser.header.frame_id = scan_msg->header.frame_id;
      pt_laser.point.x = range * cos(angle);
      pt_laser.point.y = range * sin(angle);
      pt_laser.point.z = 0.0; // 2D лидар плоский

      // 2. Переводим эту точку в координаты объектива камеры
      geometry_msgs::PointStamped pt_cam;
      tf2::doTransform(pt_laser, pt_cam, transformStamped);

      // 3. Если точка находится ПОЗАДИ камеры (Z <= 0) или вне кадра, 
      // стираем её, так как мы не можем проверить её по маске.
      if (pt_cam.point.z <= 0.0) {
        filtered_scan.ranges[i] = std::numeric_limits<float>::infinity();
        continue;
      }

      // 4. Проецируем 3D точку на 2D пиксели картинки
      cv::Point2d uv = cam_model_.project3dToPixel(
          cv::Point3d(pt_cam.point.x, pt_cam.point.y, pt_cam.point.z));

      // 5. Проверяем, попал ли пиксель внутрь кадра
      if (uv.x >= 0 && uv.x < latest_mask_.cols && uv.y >= 0 &&
          uv.y < latest_mask_.rows) {

        // 6. Узнаем цвет пикселя маски (0 = черное/трава, 255 = белое/дорога)
        uint8_t pixel_val = latest_mask_.at<uint8_t>(uv.y, uv.x);

        if (pixel_val == 0) {
          // БИНГО! Лидар ударил в траву/горку.
          filtered_scan.ranges[i] = std::numeric_limits<float>::infinity();
        }
      } else {
        // Точка впереди камеры, но не попала в кадр (сбоку)
        filtered_scan.ranges[i] = std::numeric_limits<float>::infinity();
      }
    }

    // Публикуем очищенный скан
    scan_pub_.publish(filtered_scan);
  }
};

int main(int argc, char **argv) {
  ros::init(argc, argv, "lidar_camera_fusion");
  ScanMaskFilter node;
  ros::spin();
  return 0;
}