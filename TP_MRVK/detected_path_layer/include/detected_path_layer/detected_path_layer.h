#ifndef DETECTED_PATH_LAYER_H_
#define DETECTED_PATH_LAYER_H_

#include <boost/thread/mutex.hpp>
#include <costmap_2d/GenericPluginConfig.h>
#include <costmap_2d/layer.h>
#include <costmap_2d/layered_costmap.h>
#include <cv_bridge/cv_bridge.h>
#include <dynamic_reconfigure/server.h>
#include <math.h>
#include <path_detector/DetectedPath.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud.h>
#include <vector>

#define PI 3.1415926536

namespace costmap_2d {
class DetectedPathLayer : public costmap_2d::Layer {
public:
  DetectedPathLayer();
  virtual void onInitialize();
  virtual void updateBounds(double origin_x, double origin_y, double origin_yaw,
                            double *min_x, double *min_y, double *max_x,
                            double *max_y);
  virtual void updateCosts(costmap_2d::Costmap2D &master_grid, int min_i,
                           int min_j, int max_i, int max_j);

private:
  void reconfigureCB(costmap_2d::GenericPluginConfig &config, uint32_t level);
  void PathDetectionCallback(const path_detector::DetectedPath::ConstPtr &msg);
  void frame_to_costmap(unsigned int height, unsigned int width,
                        const std::vector<unsigned char> &frame_data);

  dynamic_reconfigure::Server<costmap_2d::GenericPluginConfig> *dsrv_;
  ros::Subscriber sub;
  ros::Publisher point_pub_;
  ros::Publisher image_pub_;
  boost::mutex lock_;

  std::vector<double> relative_x;
  std::vector<double> relative_y;
  std::vector<double> global_mark_x;
  std::vector<double> global_mark_y;

  double min_x, min_y, max_x, max_y;

  const double CAMERA_FOCAL_LENGTH = 474.4268835;
  const double CAMERA_X = 0.1946454825;
  const double CAMERA_HEIGHT = 0.5706493713;
  const double CAMERA_PITCH = PI / 8;
  const double PHI = PI - CAMERA_PITCH;
  const double COS_CAMERA_PITCH = cos(CAMERA_PITCH);
  const double SIN_CAMERA_PITCH = sin(CAMERA_PITCH);
};
} // namespace costmap_2d
#endif