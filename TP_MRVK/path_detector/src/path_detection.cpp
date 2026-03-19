#include <algorithm>
#include <cmath>
#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/Twist.h>
#include <image_transport/image_transport.h>
#include <iostream>
#include <nav_msgs/OccupancyGrid.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/LaserScan.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <std_msgs/Float32.h>
#include <std_msgs/Header.h>
#include <vector>

// GUI windows removed

// --- Вспомогательные структуры ---

struct FieldOfVision {
  cv::Mat mask;
  std::vector<std::vector<cv::Point>> contours;
  int shape_h, shape_w;
  int lower_width, upper_width, height;
  float x0, y0;
  cv::Rect control_zone;

  FieldOfVision() : shape_h(0), shape_w(0) {}

  void init(cv::Size size, int l_w, int u_w, int h, float _x0 = 0.0,
            float _y0 = 0.0) {
    shape_h = size.height;
    shape_w = size.width;
    lower_width = l_w;
    upper_width = u_w;
    height = h;
    x0 = _x0;
    y0 = _y0;

    mask = create_field_of_vision_mask(size, lower_width, upper_width, height,
                                       x0, y0);

    cv::Mat thresh;
    cv::threshold(mask, thresh, 0, 255, cv::THRESH_BINARY);
    cv::findContours(thresh, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

    create_control_zone(size, upper_width, height);
  }

  cv::Mat create_field_of_vision_mask(cv::Size size, int lw, int uw, int h,
                                      float x_off, float y_off) {
    cv::Mat m = cv::Mat::zeros(size, CV_8UC1);
    int rows = size.height;
    int cols = size.width;

    float y_base = rows - 1 - y_off;
    float x_left_bot = (cols - lw) / 2.0 + x_off;
    float x_right_bot = x_left_bot + lw;
    float x_left_top = (cols - uw) / 2.0 + x_off;
    float x_right_top = x_left_top + uw;
    float y_top = y_base - h;

    std::vector<cv::Point> pts = {cv::Point((int)x_left_bot, (int)y_base),
                                  cv::Point((int)x_right_bot, (int)y_base),
                                  cv::Point((int)x_right_top, (int)y_top),
                                  cv::Point((int)x_left_top, (int)y_top)};

    std::vector<std::vector<cv::Point>> fillPts = {pts};
    cv::fillPoly(m, fillPts, cv::Scalar(255));
    return m;
  }

  void create_control_zone(cv::Size size, int uw, int h) {
    int r_start = size.height - h;
    int r_end = size.height - 0.7 * h;
    int c_start = size.width / 2.0 - 0.25 * uw;
    int c_end = size.width / 2.0 + 0.25 * uw;

    r_start = std::max(0, r_start);
    r_end = std::min(size.height, r_end);
    c_start = std::max(0, c_start);
    c_end = std::min(size.width, c_end);

    if (c_end > c_start && r_end > r_start)
      control_zone =
          cv::Rect(c_start, r_start, c_end - c_start, r_end - r_start);
    else
      control_zone = cv::Rect(0, 0, 1, 1);
  }
};

struct Controller {
  float v, w;
  float passability;
  float thresh_MODE0to1;
  float thresh_MODE1to0;
  int currMODE; // 0: Normal, 1: Rotating

  Controller()
      : v(0), w(0), passability(0), thresh_MODE0to1(0.2), thresh_MODE1to0(0.6),
        currMODE(0) {}

  float saturation(float u, float lower, float upper) {
    return std::max(lower, std::min(u, upper));
  }

  float rate_limiter(float y, float u, float falling, float rising) {
    float rate = u - y;
    if (rate < falling)
      return y + falling;
    if (rate > rising)
      return y + rising;
    return u;
  }

  void calcActuatingSig(const std::vector<int> &x, const std::vector<int> &y,
                        geometry_msgs::Twist &msg) {
    if (x.size() < 4)
      return;

    float dx = (float)x[x.size() - 3] - x[0];
    float dy = (float)-y[y.size() - 3] + y[0];
    float ang = std::atan2(dy, dx);

    // Switch mode logic
    if (passability <= thresh_MODE0to1)
      currMODE = 1;
    if (currMODE == 1 && passability >= thresh_MODE1to0)
      currMODE = 0;

    float target_v = 0, target_w = 0;

    if (currMODE == 0) {
      target_v = saturation(3.0 * passability, -1.0, 1.0);
      float target_w_raw = -M_PI / 2.0 + ang;
      target_w = saturation(5.0 * target_w_raw, -0.5, 0.5);
    } else {
      target_v = 0.0;
      target_w = -0.5;
    }

    v = rate_limiter(v, target_v, -0.1, 0.1);
    w = rate_limiter(w, target_w, -0.1, 0.1);

    msg.linear.x = v;
    msg.angular.z = w;
  }
};

class ImageConverter {
  ros::NodeHandle nh_;
  image_transport::ImageTransport it_;
  image_transport::Subscriber image_sub_;
  ros::Publisher cmd_pub_;
  ros::Publisher costmap_pub_;
  ros::Publisher scan_pub_;
  ros::Publisher passability_pub_;
  ros::Publisher road_grid_pub_;
  ros::Publisher grass_pc_pub_;
  image_transport::Publisher mask_image_pub_;

  FieldOfVision fov_, op_area_;
  Controller controller_;

  bool initialized_;
  std::vector<cv::Scalar> thresh_lower_hsv_, thresh_upper_hsv_;
  int K_COLORS;
  cv::Vec3i offset_lower_, offset_upper_;

  int mask_lb_x, mask_rb_x, mask_lt_x, mask_rt_x, mask_t_y, mask_b_y;
  bool interactive_mask_;
  bool standalone_mode_;


public:
  ros::Time last_image_time;
  std::string frame_id_param;

  ImageConverter() : it_(nh_), initialized_(false), K_COLORS(2) {
    last_image_time = ros::Time::now();
    cmd_pub_ = nh_.advertise<geometry_msgs::Twist>("/shoddy/cmd_vel", 1);
    costmap_pub_ = nh_.advertise<nav_msgs::OccupancyGrid>("/path_detector/costmap", 1);
    scan_pub_ = nh_.advertise<sensor_msgs::LaserScan>("/path_detector/scan", 1);
    passability_pub_ = nh_.advertise<std_msgs::Float32>("/path_detector/passability", 1);
    road_grid_pub_ = nh_.advertise<nav_msgs::OccupancyGrid>("/path_detector/road_grid", 1);
    grass_pc_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/path_detector/grass_cloud", 1);
    mask_image_pub_ = it_.advertise("/path_detector/mask", 1);

    offset_lower_ = cv::Vec3i(15, 25, 55);
    offset_upper_ = cv::Vec3i(15, 25, 55);

    ros::NodeHandle pnh("~");
    pnh.param("mask_lb_x", mask_lb_x, 0);
    pnh.param("mask_rb_x", mask_rb_x, 100);
    pnh.param("mask_lt_x", mask_lt_x, 0);
    pnh.param("mask_rt_x", mask_rt_x, 100);
    pnh.param("mask_t_y", mask_t_y, 36);
    pnh.param("mask_b_y", mask_b_y, 100);
    pnh.param("interactive_mask", interactive_mask_, true);
    pnh.param<std::string>("frame_id", frame_id_param, "odom");
    pnh.param("standalone_mode", standalone_mode_, false);


    image_sub_ = it_.subscribe("/camera/image_raw", 1, &ImageConverter::imageCb, this);
    ROS_INFO("[PathDetector] Unified node started. Frame ID: %s", frame_id_param.c_str());
  }

  void initialize(const cv::Mat &img, bool mark_initialized = true) {
    cv::Size size = img.size();
    float h_fov = size.height * (mask_b_y - mask_t_y) / 100.0;
    int lw = size.width * (mask_rb_x - mask_lb_x) / 100.0;
    int uw = size.width * (mask_rt_x - mask_lt_x) / 100.0;
    float x_off = size.width * (mask_lb_x + mask_rb_x) / 200.0 - size.width / 2.0;
    float y_off = size.height * (100 - mask_b_y) / 100.0;

    fov_.init(size, lw, uw, (int)h_fov, x_off, y_off);
    op_area_.init(size, size.width, size.width, 250, 0, 0);

    int rect_h = size.height / 9;
    int rect_w = size.width * 2 / 9;
    int roi_x = std::max(0, (int)(size.width / 2 - rect_w / 2));
    int roi_y = std::max(0, (int)(size.height - rect_h * 2));
    int roi_w = std::min((int)rect_w, size.width - roi_x);
    int roi_h = std::min((int)rect_h, size.height - roi_y);

    if (roi_w <= 0 || roi_h <= 0) {
      if (mark_initialized) initialized_ = true;
      return;
    }

    cv::Rect roi(roi_x, roi_y, roi_w, roi_h);
    cv::Mat centerImg = img(roi).clone();
    cv::Mat centerFloat;
    centerImg.reshape(1, centerImg.rows * centerImg.cols).convertTo(centerFloat, CV_32F);

    cv::Mat labels, centers;
    cv::kmeans(centerFloat, K_COLORS, labels, cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 10, 1.0), 3, cv::KMEANS_RANDOM_CENTERS, centers);

    thresh_lower_hsv_.clear();
    thresh_upper_hsv_.clear();
    for (int i = 0; i < centers.rows; ++i) {
      cv::Mat bgr(1, 1, CV_8UC3, cv::Scalar(centers.at<float>(i, 0), centers.at<float>(i, 1), centers.at<float>(i, 2)));
      cv::Mat hsv;
      cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);
      cv::Vec3b hsv_val = hsv.at<cv::Vec3b>(0, 0);
      thresh_lower_hsv_.push_back(cv::Scalar(std::max(0, hsv_val[0] - offset_lower_[0]), std::max(0, hsv_val[1] - offset_lower_[1]), std::max(0, hsv_val[2] - offset_lower_[2])));
      thresh_upper_hsv_.push_back(cv::Scalar(std::min(180, hsv_val[0] + offset_upper_[0]), std::min(255, hsv_val[1] + offset_upper_[1]), std::min(255, hsv_val[2] + offset_upper_[2])));
    }
    if (mark_initialized) initialized_ = true;
  }

  cv::Mat create_mask(const cv::Mat &img) {
    cv::Mat hsv;
    cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);
    cv::Mat combined = cv::Mat::zeros(img.size(), CV_8UC1);
    for (size_t i = 0; i < thresh_lower_hsv_.size(); ++i) {
      cv::Mat m;
      cv::inRange(hsv, thresh_lower_hsv_[i], thresh_upper_hsv_[i], m);
      combined |= m;
    }
    // Бинарное закрытие для объединения разрозненных фрагментов (более агрессивное)
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(11, 11));
    cv::morphologyEx(combined, combined, cv::MORPH_CLOSE, kernel);
    combined &= fov_.mask;

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(combined.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
    if (contours.empty()) return combined;

    // Заполняем ВСЕ контуры, которые имеют достаточную площадь (чтобы обработать разрывы в дороге)
    cv::Mat out = cv::Mat::zeros(combined.size(), CV_8UC1);
    for (size_t i = 0; i < contours.size(); i++) {
      if (cv::contourArea(contours[i]) > 500) {
        cv::drawContours(out, contours, (int)i, cv::Scalar(255), cv::FILLED);
      }
    }

    // Финальное сглаживание краев
    kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(out, out, cv::MORPH_CLOSE, kernel);

    return out;
  }

  void find_path_center(cv::Mat &img, const cv::Mat &mask_comb) {
    int ROWS = img.rows;
    int COLS = img.cols;
    int N = 5;
    float height = (float)fov_.height;
    int dN = (int)(height / N);
    std::vector<int> x(N + 1, COLS / 2);
    std::vector<int> y(N + 1, ROWS);
    float sum_inv = 0;
    for (int i = 1; i <= N; ++i) sum_inv += 1.0f / i;
    int yn = std::min(ROWS, (int)(ROWS - fov_.y0));
    float k_sum = sum_inv;
    int dyn = (int)(k_sum * dN);

    for (int n = 0; n < N; ++n) {
      int r_start = std::max(0, yn - dyn);
      int r_end = std::max(0, (n == 0) ? yn - (int)(dyn / 2.0) : yn);
      if (r_start < r_end && r_end <= ROWS) {
        cv::Mat strip = mask_comb(cv::Range(r_start, r_end), cv::Range::all());
        cv::Moments M = cv::moments(strip);
        if (M.m00 != 0) {
          x[n + 1] = (int)(M.m10 / M.m00);
          y[n + 1] = (int)(M.m01 / M.m00) + r_start;
        } else {
          x[n + 1] = x[n]; y[n + 1] = y[n];
        }
      } else {
        x[n + 1] = x[n]; y[n + 1] = y[n];
      }
      yn -= dyn; k_sum -= 1.0f / (n + 1.0f); dyn = (int)(k_sum * dN);
    }

    for (int n = 0; n < N; ++n) cv::line(img, cv::Point(x[n], y[n]), cv::Point(x[n+1], y[n+1]), cv::Scalar(0, 255, 0), 5);

    cv::Rect r = fov_.control_zone;
    if (r.width > 0 && r.height > 0) {
      cv::Mat cz = mask_comb(r);
      controller_.passability = (float)cv::countNonZero(cz) / (float)cz.total();
      cv::rectangle(img, r, cv::Scalar(255, 0, 0), 5);
    }

    std_msgs::Float32 pass_msg;
    pass_msg.data = controller_.passability;
    passability_pub_.publish(pass_msg);

    if (standalone_mode_) {
      geometry_msgs::Twist twist;
      controller_.calcActuatingSig(x, y, twist);
      cmd_pub_.publish(twist);
    } else {
      // DWA MODE: IPM Projection
      float y_base = ROWS - 1 - fov_.y0;
      float x_left_bot = (COLS - fov_.lower_width) / 2.0 + fov_.x0;
      float x_right_bot = x_left_bot + fov_.lower_width;
      float x_left_top = (COLS - fov_.upper_width) / 2.0 + fov_.x0;
      float x_right_top = x_left_top + fov_.upper_width;
      float y_top = y_base - fov_.height;

      std::vector<cv::Point2f> src_pts = {cv::Point2f(x_left_top, y_top), cv::Point2f(x_right_top, y_top), cv::Point2f(x_right_bot, y_base), cv::Point2f(x_left_bot, y_base)};
      float grid_res = 0.05;
      float world_h = 5.0;
      float world_w = 4.0;
      int bev_w = (int)(world_w / grid_res);
      int bev_h = (int)(world_h / grid_res);
      std::vector<cv::Point2f> dst_pts = {cv::Point2f(0, 0), cv::Point2f(bev_w, 0), cv::Point2f(bev_w, bev_h), cv::Point2f(0, bev_h)};

      cv::Mat M = cv::getPerspectiveTransform(src_pts, dst_pts);
      cv::Mat bev_mask;
      cv::warpPerspective(mask_comb, bev_mask, M, cv::Size(bev_w, bev_h), cv::INTER_NEAREST);
      
      // Дополнительная очистка шума прямо на BEV-маске
      cv::Mat k_bev = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
      cv::morphologyEx(bev_mask, bev_mask, cv::MORPH_CLOSE, k_bev);

      nav_msgs::OccupancyGrid grid;
      grid.header.stamp = ros::Time::now();
      grid.header.frame_id = "base_link";
      grid.info.resolution = grid_res;
      grid.info.width = bev_h;  // X-направление (вперед)
      grid.info.height = bev_w; // Y-направление (вбок)
      grid.info.origin.position.x = 0.0;
      grid.info.origin.position.y = -world_w / 2.0;
      grid.info.origin.orientation.w = 1.0;
      grid.data.resize(bev_w * bev_h);

      sensor_msgs::PointCloud2 cloud;
      cloud.header.stamp = grid.header.stamp;
      cloud.header.frame_id = grid.header.frame_id;
      sensor_msgs::PointCloud2Modifier modifier(cloud);
      modifier.setPointCloud2FieldsByString(1, "xyz");
      modifier.resize(bev_w * bev_h);
      sensor_msgs::PointCloud2Iterator<float> iter_x(cloud, "x");
      sensor_msgs::PointCloud2Iterator<float> iter_y(cloud, "y");
      sensor_msgs::PointCloud2Iterator<float> iter_z(cloud, "z");

      sensor_msgs::LaserScan scan;
      scan.header.stamp = grid.header.stamp;
      scan.header.frame_id = grid.header.frame_id;
      scan.angle_min = -1.57; scan.angle_max = 1.57; scan.angle_increment = 0.01;
      scan.range_min = 0.1; scan.range_max = 4.0;
      int scan_n = (int)((scan.angle_max - scan.angle_min) / scan.angle_increment);
      scan.ranges.resize(scan_n, scan.range_max);

      int p_count = 0;
      for (int i = 0; i < bev_h; ++i) {
        for (int j = 0; j < bev_w; ++j) {
          float wx = (bev_h - 1 - i) * grid_res;
          float wy = -world_w / 2.0 + (bev_w - 1 - j) * grid_res;

          // Индексы для OccupancyGrid [y * width + x]
          // где x - вперед (bev_h), y - вбок (bev_w)
          int grid_x = (bev_h - 1 - i); 
          int grid_y = (bev_w - 1 - j);
          int idx = grid_y * grid.info.width + grid_x;



          if (bev_mask.at<uchar>(i, j) > 127) {
            grid.data[idx] = 0;
          } else {
            grid.data[idx] = 100;
            if (i % 2 == 0 && j % 2 == 0) {
              *iter_x = wx; *iter_y = wy; *iter_z = 0.0;
              ++iter_x; ++iter_y; ++iter_z;
              p_count++;
            }
            float r = std::sqrt(wx * wx + wy * wy);
            float a = std::atan2(wy, wx);
            int s_idx = (int)((a - scan.angle_min) / scan.angle_increment);
            if (s_idx >= 0 && s_idx < scan_n && r < scan.ranges[s_idx]) scan.ranges[s_idx] = r;
          }
        }
      }
      modifier.resize(p_count);
      road_grid_pub_.publish(grid);
      costmap_pub_.publish(grid);
      grass_pc_pub_.publish(cloud);
      scan_pub_.publish(scan);
    }
  }

  void imageCb(const sensor_msgs::ImageConstPtr &msg) {
    last_image_time = ros::Time::now();
    cv_bridge::CvImagePtr cv_ptr;
    try { cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8); }
    catch (cv_bridge::Exception &e) { return; }

    cv::Mat img = cv_ptr->image.clone();
    if (img.empty()) return;
    cv::GaussianBlur(img, img, cv::Size(3, 3), 0);

    if (!initialized_) initialize(img, true);

    if (initialized_) {
      cv::Mat mask = create_mask(img);
      find_path_center(img, mask);
      sensor_msgs::ImagePtr mask_msg = cv_bridge::CvImage(std_msgs::Header(), "mono8", mask).toImageMsg();
      mask_image_pub_.publish(mask_msg);
    }
  }
};

int main(int argc, char **argv) {
  ros::init(argc, argv, "path_detection");
  ImageConverter ic;
  ros::Rate r(10);
  while (ros::ok()) {
    ros::spinOnce();
    if ((ros::Time::now() - ic.last_image_time).toSec() > 5.0) ROS_WARN_THROTTLE(5.0, "[PathDetector] No images received!");
    r.sleep();
  }
  return 0;
}
