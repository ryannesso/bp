#include <algorithm>
#include <detected_path_layer/detected_path_layer.h>
#include <iostream>
#include <limits>
#include <pluginlib/class_list_macros.h>

PLUGINLIB_EXPORT_CLASS(costmap_2d::DetectedPathLayer, costmap_2d::Layer)

namespace costmap_2d {

DetectedPathLayer::DetectedPathLayer() {}

void DetectedPathLayer::onInitialize() {
  ros::NodeHandle nh("~/" + name_);
  current_ = true;
  dsrv_ = new dynamic_reconfigure::Server<costmap_2d::GenericPluginConfig>(nh);
  dsrv_->setCallback(
      boost::bind(&DetectedPathLayer::reconfigureCB, this, _1, _2));

  sub = nh.subscribe("/shoddy/detected_path", 10,
                     &DetectedPathLayer::PathDetectionCallback, this);
  point_pub_ = nh.advertise<sensor_msgs::PointCloud>("debug_points", 1);
  image_pub_ = nh.advertise<sensor_msgs::Image>("debug_image", 1);
}

void DetectedPathLayer::reconfigureCB(costmap_2d::GenericPluginConfig &config,
                                      uint32_t level) {
  enabled_ = config.enabled;
}

void DetectedPathLayer::PathDetectionCallback(
    const path_detector::DetectedPath::ConstPtr &msg) {
  frame_to_costmap(msg->height, msg->width, msg->frame);
}

void DetectedPathLayer::frame_to_costmap(
    unsigned int height, unsigned int width,
    const std::vector<unsigned char> &frame_data) {
  std::vector<double> temp_x, temp_y;

  for (unsigned int h = 0; h < height; h += 4) {
    for (unsigned int w = 0; w < width; w += 4) {
      if (frame_data[h * width + w] == 255) { // 255 = препятствие

        double dx = (double)w - (double)width / 2.0;
        double dy = (double)h - (double)height / 2.0;

        // Угол луча от оптической оси (вертикальный)
        double alpha = atan2(dy, CAMERA_FOCAL_LENGTH);
        double beta = CAMERA_PITCH + alpha;

        // Если beta <= 0, точка за горизонтом или выше — игнорируем
        if (beta > 0.05) {
          // Дистанция по земле от точки под камерой до точки проекции
          double r = CAMERA_HEIGHT / tan(beta);

          // Координаты в системе base_link
          double x_proj = CAMERA_X + r;
          
          // Боковое смещение (Y)
          // t = CAMERA_HEIGHT / (v_z_camera_frame_rotated)
          // v_z_rotated = -(F*sin(p) + dy*cos(p))
          double denominator = CAMERA_FOCAL_LENGTH * SIN_CAMERA_PITCH + dy * COS_CAMERA_PITCH;
          double y_proj = -dx * CAMERA_HEIGHT / denominator;

          double dist = sqrt((x_proj - CAMERA_X) * (x_proj - CAMERA_X) + y_proj * y_proj);

          // Ограничиваем дистанцию и проверяем, что точка перед роботом
          if (dist > 0.3 && dist < 5.0 && x_proj > 0.1) {
            temp_x.push_back(x_proj);
            temp_y.push_back(y_proj);
          }
        }
      }
    }
  }

  {
    boost::mutex::scoped_lock lock(lock_);
    relative_x = temp_x;
    relative_y = temp_y;
  }

  // --- Визуализация (только если нужно) ---
  if (point_pub_.getNumSubscribers() > 0) {
    sensor_msgs::PointCloud cloud;
    cloud.header.frame_id = "base_link";
    cloud.header.stamp = ros::Time::now();
    cloud.points.resize(temp_x.size());
    for (size_t i = 0; i < temp_x.size(); i++) {
      cloud.points[i].x = temp_x[i];
      cloud.points[i].y = temp_y[i];
      cloud.points[i].z = 0.0;
    }
    point_pub_.publish(cloud);
  }
}

void DetectedPathLayer::updateBounds(double origin_x, double origin_y,
                                     double origin_yaw, double *min_x,
                                     double *min_y, double *max_x,
                                     double *max_y) {
  if (!enabled_)
    return;

  global_mark_x.clear();
  global_mark_y.clear();

  double min_bx = std::numeric_limits<double>::max();
  double min_by = std::numeric_limits<double>::max();
  double max_bx = -std::numeric_limits<double>::max();
  double max_by = -std::numeric_limits<double>::max();
  bool has_points = false;

  {
    boost::mutex::scoped_lock lock(lock_);
    for (size_t i = 0; i < relative_x.size(); i++) {
      double gx = origin_x + relative_x[i] * cos(origin_yaw) -
                  relative_y[i] * sin(origin_yaw);
      double gy = origin_y + relative_x[i] * sin(origin_yaw) +
                  relative_y[i] * cos(origin_yaw);
      global_mark_x.push_back(gx);
      global_mark_y.push_back(gy);
      min_bx = std::min(min_bx, gx);
      min_by = std::min(min_by, gy);
      max_bx = std::max(max_bx, gx);
      max_by = std::max(max_by, gy);
      has_points = true;
    }
  }

  if (has_points) {
    *min_x = std::min(*min_x, min_bx);
    *min_y = std::min(*min_y, min_by);
    *max_x = std::max(*max_x, max_bx);
    *max_y = std::max(*max_y, max_by);
  }
}

void DetectedPathLayer::updateCosts(costmap_2d::Costmap2D &master_grid,
                                    int min_i, int min_j, int max_i,
                                    int max_j) {
  if (!enabled_)
    return;
  for (size_t i = 0; i < global_mark_x.size(); i++) {
    unsigned int x, y;
    if (master_grid.worldToMap(global_mark_x[i], global_mark_y[i], x, y)) {
      master_grid.setCost(x, y, LETHAL_OBSTACLE);
    }
  }
}
} // namespace costmap_2d