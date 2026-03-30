#include <algorithm>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <nav_msgs/OccupancyGrid.h>
#include <opencv2/ml.hpp>
#include <opencv2/opencv.hpp>
#include <ros/ros.h>
#include <sensor_msgs/image_encodings.h>
#include <vector>

class ImageConverter {
  ros::NodeHandle nh_;
  image_transport::ImageTransport it_;
  image_transport::Subscriber image_sub_;
  image_transport::Publisher mask_image_pub_;
  ros::Publisher grid_pub_;

  int mask_t_y, mask_b_y;
  double color_tolerance_;
  int edge_thresh_;
  double texture_thresh_;

  double grid_res_;
  int grid_w_, grid_h_;
  cv::Mat M_ipm_;

  cv::Ptr<cv::ml::DTrees> dtree_;

private:
  cv::Mat prev_final_mask_;
  float temporal_alpha = 0.3f;

public:
  ImageConverter() : it_(nh_) {
    mask_image_pub_ = it_.advertise("/path_detector/mask", 1);
    grid_pub_ =
        nh_.advertise<nav_msgs::OccupancyGrid>("/path_detector/costmap", 1);

    ros::NodeHandle pnh("~");
    pnh.param("mask_t_y", mask_t_y, 40);
    pnh.param("mask_b_y", mask_b_y, 100);
    pnh.param("color_tolerance", color_tolerance_, 25.0);
    pnh.param("edge_thresh", edge_thresh_, 40);
    pnh.param("texture_thresh", texture_thresh_, 25.0);
    pnh.param("grid_resolution", grid_res_, 0.05);

    grid_w_ = 100;
    grid_h_ = 100;

    std::vector<cv::Point2f> src_pts = {
        cv::Point2f(230, 260), cv::Point2f(410, 260), cv::Point2f(640, 480),
        cv::Point2f(0, 480)};
    std::vector<cv::Point2f> dst_pts = {
        cv::Point2f(0, 0), cv::Point2f(grid_w_, 0),
        cv::Point2f(grid_w_, grid_h_), cv::Point2f(0, grid_h_)};
    M_ipm_ = cv::getPerspectiveTransform(src_pts, dst_pts);

    image_sub_ =
        it_.subscribe("/camera/image_raw", 1, &ImageConverter::imageCb, this);
    ROS_INFO("[PathDetector] Initialized Fast & Safe DTree ML ray-caster.");
  }

  // ОБНОВЛЕННЫЙ SKAN: Теперь он сам спрашивает ML модель на ходу!
  cv::Mat run_scan(const cv::Ptr<cv::ml::DTrees> &ml_model,
                   const cv::Mat &lab_img, const cv::Mat &edges,
                   const cv::Mat &texture, int horizon, int bottom,
                   int center_x, int w, int h) {

    cv::Mat mask = cv::Mat::zeros(h, w, CV_8UC1);
    std::vector<cv::Point> left_bounds;
    std::vector<cv::Point> right_bounds;
    int current_center_x = center_x;
    int narrow_count = 0;

    // Резервируем память один раз (спасает от сборщика мусора и ускоряет
    // работу)
    cv::Mat ml_sample(1, 4, CV_32F);
    float *s_ptr = ml_sample.ptr<float>(0);

    for (int y = bottom; y > horizon; y -= 4) {

      int max_half_width = (y - horizon) * (w / 1.0f) / (h - horizon) + 40;
      int left_x = current_center_x;
      int right_x = current_center_x;

      // --- Скан влево ---
      while (left_x > 2 && (current_center_x - left_x) < max_half_width) {
        if (edges.at<uchar>(y, left_x) > 0)
          break; // Жесткий край (бордюр)

        // ML Проверка конкретного пикселя (БЫСТРО и БЕЗ ОШИБОК ПАМЯТИ)
        if (ml_model->isTrained()) {
          cv::Vec3b p = lab_img.at<cv::Vec3b>(y, left_x);
          s_ptr[0] = p[0];
          s_ptr[1] = p[1];
          s_ptr[2] = p[2];
          s_ptr[3] = texture.at<float>(y, left_x);

          // Если ML предсказывает 0 (Не дорога), прекращаем скан
          if (ml_model->predict(ml_sample) < 0.5f)
            break;
        }

        left_x--;
      }

      // --- Скан вправо ---
      while (right_x < w - 3 && (right_x - current_center_x) < max_half_width) {
        if (edges.at<uchar>(y, right_x) > 0)
          break;

        if (ml_model->isTrained()) {
          cv::Vec3b p = lab_img.at<cv::Vec3b>(y, right_x);
          s_ptr[0] = p[0];
          s_ptr[1] = p[1];
          s_ptr[2] = p[2];
          s_ptr[3] = texture.at<float>(y, right_x);

          if (ml_model->predict(ml_sample) < 0.5f)
            break;
        }

        right_x++;
      }

      // Защита от тупиков: если проезд стал слишком узким
      if (right_x - left_x < w * 0.08) {
        narrow_count++;
        if (narrow_count > 10)
          break;
        if (!left_bounds.empty()) {
          left_x = left_bounds.back().x;
          right_x = right_bounds.back().x;
        }
      } else {
        narrow_count = 0;
      }

      // РАСШИРЕНИЕ МАСКИ: добавляем запас ширины
      int expand = (int)(max_half_width * 0.20f) + 20;
      left_bounds.push_back(cv::Point(std::max(0, left_x - expand), y));
      right_bounds.push_back(cv::Point(std::min(w - 1, right_x + expand), y));

      current_center_x =
          (int)(current_center_x * 0.7 + ((left_x + right_x) / 2) * 0.3);
    }

    if (left_bounds.size() > 5) {
      std::vector<cv::Point> poly;
      for (auto &p : left_bounds)
        poly.push_back(p);
      for (int i = right_bounds.size() - 1; i >= 0; i--)
        poly.push_back(right_bounds[i]);

      std::vector<cv::Point> straight_poly;
      cv::convexHull(poly, straight_poly);
      cv::approxPolyDP(straight_poly, straight_poly, 30.0, true);

      cv::fillPoly(mask, std::vector<std::vector<cv::Point>>{straight_poly},
                   cv::Scalar(255));
    }
    return mask;
  }

  cv::Mat create_adaptive_mask(cv::Mat &img) {
    int h = img.rows;
    int w = img.cols;

    cv::Mat clean_img;
    cv::medianBlur(img, clean_img, 7);

    cv::Mat gray, edges;
    cv::cvtColor(clean_img, gray, cv::COLOR_BGR2GRAY);

    // Карта текстур
    cv::Mat sobel_x, sobel_y, texture;
    cv::Sobel(gray, sobel_x, CV_32F, 1, 0, 3);
    cv::Sobel(gray, sobel_y, CV_32F, 0, 1, 3);
    cv::magnitude(sobel_x, sobel_y, texture);
    cv::GaussianBlur(texture, texture, cv::Size(21, 21), 0);

    // Контуры
    cv::Canny(gray, edges, edge_thresh_, edge_thresh_ * 2.5);
    cv::dilate(edges, edges,
               cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));

    cv::Mat lab_img;
    cv::cvtColor(clean_img, lab_img, cv::COLOR_BGR2Lab);

    int horizon = (int)(h * mask_t_y / 100.0f);
    int bottom = (int)(h * mask_b_y / 100.0f) - 10;
    int center_x = w / 2;

    // --- MACHINE LEARNING (ОБУЧЕНИЕ ТОЛЬКО) ---
    int r_w = 80, r_h = 40;
    int rx = std::max(0, center_x - r_w / 2);
    int ry = std::max(0, std::min(h - 1, bottom - r_h));

    int act_r_w = std::min(w - rx, r_w);
    int act_r_h = std::min(h - ry, r_h);

    int horizon_clmp = std::max(0, std::min(h - 1, horizon));
    int obs_y = std::max(0, horizon_clmp - 60);

    int road_count = 0, obs_count = 0;
    for (int y = ry; y < ry + act_r_h; y += 2) {
      for (int x = rx; x < rx + act_r_w; x += 2)
        road_count++;
    }
    for (int y = obs_y; y < horizon_clmp; y += 3) {
      for (int x = 0; x < w; x += 5)
        obs_count++;
    }

    // ВАЖНО: Создаем "свежее" дерево на каждом кадре для избежания утечек
    // памяти OpenCV!
    dtree_ = cv::ml::DTrees::create();
    dtree_->setMaxDepth(8);
    dtree_->setMinSampleCount(5);

    // Жесткая защита от сбоев (Обучаемся только если видно небо и дорогу)
    if (road_count > 10 && obs_count > 10) {
      int n_samples = road_count + obs_count;
      cv::Mat training_data(n_samples, 4, CV_32F);
      cv::Mat labels(n_samples, 1, CV_32S);

      int s_idx = 0;
      // Дорога (Метка 1)
      for (int y = ry; y < ry + act_r_h; y += 2) {
        for (int x = rx; x < rx + act_r_w; x += 2) {
          cv::Vec3b p = lab_img.at<cv::Vec3b>(y, x);
          training_data.at<float>(s_idx, 0) = p[0];
          training_data.at<float>(s_idx, 1) = p[1];
          training_data.at<float>(s_idx, 2) = p[2];
          training_data.at<float>(s_idx, 3) = texture.at<float>(y, x);
          labels.at<int>(s_idx, 0) = 1;
          s_idx++;
        }
      }

      // Препятствия/Небо (Метка 0)
      for (int y = obs_y; y < horizon_clmp; y += 3) {
        for (int x = 0; x < w; x += 5) {
          cv::Vec3b p = lab_img.at<cv::Vec3b>(y, x);
          training_data.at<float>(s_idx, 0) = p[0];
          training_data.at<float>(s_idx, 1) = p[1];
          training_data.at<float>(s_idx, 2) = p[2];
          training_data.at<float>(s_idx, 3) = texture.at<float>(y, x);
          labels.at<int>(s_idx, 0) = 0;
          s_idx++;
        }
      }

      cv::Ptr<cv::ml::TrainData> tData =
          cv::ml::TrainData::create(training_data, cv::ml::ROW_SAMPLE, labels);
      dtree_->train(tData);
    }

    // Запускаем ML скан!
    cv::Mat current_mask = run_scan(dtree_, lab_img, edges, texture, horizon,
                                    bottom, center_x, w, h);

    if (prev_final_mask_.empty())
      prev_final_mask_ = current_mask.clone();

    cv::Mat fused_mask;
    cv::addWeighted(current_mask, temporal_alpha, prev_final_mask_,
                    1.0 - temporal_alpha, 0, fused_mask);
    prev_final_mask_ = fused_mask.clone();

    cv::Mat bin_mask;
    cv::threshold(fused_mask, bin_mask, 127, 255, cv::THRESH_BINARY);
    return bin_mask;
  }

  void imageCb(const sensor_msgs::ImageConstPtr &msg) {
    cv_bridge::CvImagePtr cv_ptr;
    try {
      cv_ptr = cv_bridge::toCvCopy(msg, "bgr8");
    } catch (...) {
      return;
    }

    cv::Mat img = cv_ptr->image.clone();
    cv::Mat road_mask = create_adaptive_mask(img);

    cv::Mat bev_mask;
    cv::warpPerspective(road_mask, bev_mask, M_ipm_,
                        cv::Size(grid_w_, grid_h_));

    nav_msgs::OccupancyGrid grid;
    grid.header.stamp = msg->header.stamp;
    grid.header.frame_id = "base_link";
    grid.info.resolution = grid_res_;
    grid.info.width = grid_h_;
    grid.info.height = grid_w_;

    grid.info.origin.position.x = 0.5;
    grid.info.origin.position.y = -(grid_w_ * grid_res_) / 2.0;
    grid.info.origin.orientation.w = 1.0;
    grid.data.resize(grid_w_ * grid_h_);

    for (int gy = 0; gy < grid_w_; gy++) {
      for (int gx = 0; gx < grid_h_; gx++) {
        int img_y = grid_h_ - 1 - gx;
        int img_x = grid_w_ - 1 - gy;

        uint8_t val = bev_mask.at<uint8_t>(img_y, img_x);
        grid.data[gy * grid_h_ + gx] = (val > 127) ? 0 : 100;
      }
    }
    grid_pub_.publish(grid);

    // Визуализация
    cv::Mat green_overlay(img.size(), CV_8UC3, cv::Scalar(0, 255, 0));
    cv::Mat result = img.clone();
    cv::addWeighted(result, 1.0, green_overlay, 0.4, 0, result, -1);
    cv::Mat final_result;
    img.copyTo(final_result);
    result.copyTo(final_result, road_mask);

    sensor_msgs::ImagePtr mask_msg =
        cv_bridge::CvImage(msg->header, "mono8", road_mask).toImageMsg();
    mask_image_pub_.publish(mask_msg);

    cv::imshow("ML Dynamic Route Filter", final_result);
    cv::waitKey(1);
  }
};

int main(int argc, char **argv) {
  ros::init(argc, argv, "path_detection");
  ImageConverter ic;
  ros::spin();
  return 0;
}