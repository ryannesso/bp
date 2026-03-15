#ifndef IMPROVED_DWA_LOCAL_PLANNER_H_
#define IMPROVED_DWA_LOCAL_PLANNER_H_

#include <angles/angles.h>
#include <base_local_planner/goal_functions.h>
#include <base_local_planner/odometry_helper_ros.h>
#include <costmap_2d/costmap_2d_ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Twist.h>
#include <mutex>
#include <nav_core/base_local_planner.h>
#include <nav_msgs/OccupancyGrid.h>
#include <nav_msgs/Path.h>
#include <ros/ros.h>
#include <tf2_ros/buffer.h>
#include <visualization_msgs/MarkerArray.h>

// Подключаем тип сообщения для получения данных о препятствиях
#include <obstacle_detector/Obstacles.h>

#include <string>
#include <vector>

namespace improved_dwa_local_planner {

class ImprovedDWALocalPlanner : public nav_core::BaseLocalPlanner {
public:
  ImprovedDWALocalPlanner();
  ~ImprovedDWALocalPlanner();

  void initialize(std::string name, tf2_ros::Buffer *tf,
                  costmap_2d::Costmap2DROS *costmap_ros) override;
  bool computeVelocityCommands(geometry_msgs::Twist &cmd_vel) override;
  bool isGoalReached() override;
  bool setPlan(const std::vector<geometry_msgs::PoseStamped> &plan) override;

private:
  // Структура для траектории
  struct Trajectory {
    double vx;
    double vth;
    double cost;
    std::vector<geometry_msgs::PoseStamped> poses;
    int collision_pose_idx = -1; // Индекс позы столкновения (-1 если нет)
    geometry_msgs::Point
        obstacle_pos; // Предсказанная позиция объекта при столкновении
    geometry_msgs::Point
        obstacle_start_pos; // Текущая позиция объекта в момент планирования
  };

  // --- ОСНОВНЫЕ МЕТОДЫ АЛГОРИТМА ---
  void obstaclesCallback(const obstacle_detector::Obstacles::ConstPtr &msg);
  Trajectory generateTrajectory(double vx, double vth);
  double
  scoreTrajectory(Trajectory &traj,
                  const std::vector<geometry_msgs::PoseStamped> &local_plan);

  // --- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ИЗ СТАТЬИ ---
  double calculate_dis_hv(double robot_vx,
                          const geometry_msgs::PoseStamped &robot_pose,
                          const obstacle_detector::CircleObstacle &obs);
  double calculate_dis_fp(Trajectory &traj,
                          const obstacle_detector::CircleObstacle &obs);
  void
  publishMarkers(const std::vector<Trajectory> &trajectories); // Для отладки
  void publishTrackedObstacles(); // Vizualizácia sledovaných objektov

  // --- ПЕРЕМЕННЫЕ-ЧЛЕНЫ КЛАССА ---
  bool initialized_;
  costmap_2d::Costmap2DROS *costmap_ros_;
  tf2_ros::Buffer *tf_;
  std::vector<geometry_msgs::PoseStamped> global_plan_;
  base_local_planner::OdometryHelperRos odom_helper_;

  // --- ПОДПИСЧИКИ И ПАБЛИШЕРЫ ---
  ros::Publisher local_plan_pub_;
  ros::Publisher candidate_trajs_pub_;
  ros::Publisher collision_markers_pub_; // Визуализация точек коллизий
  ros::Publisher tracked_objects_pub_;   // Vizualizácia dynamických objektov
  ros::Subscriber obstacles_sub_; // Подписчик на динамические препятствия
  ros::Subscriber road_grid_sub_; // Подписчик на карту не-дорожных областей
  obstacle_detector::Obstacles
      last_obstacles_; // Хранилище для последних увиденных препятствий

  // --- ДАННЫЕ О ДОРОГЕ ОТ КАМЕРЫ ---
  nav_msgs::OccupancyGrid::ConstPtr road_grid_; // последняя карта дороги
  std::mutex road_grid_mutex_;
  void roadGridCallback(const nav_msgs::OccupancyGrid::ConstPtr &msg);

  // --- ПАРАМЕТРЫ, ЗАГРУЖАЕМЫЕ ИЗ YAML ---
  // Веса для оценочной функции
  double alpha_;   // heading_cost (направление на локальную цель)
  double beta_;    // dist_cost (расстояние до статических препятствий)
  double gamma_;   // velocity_cost (предпочтение высокой скорости)
  double kappa_;   // Вес для dis_hv
  double epsilon_; // Вес для dis_fp
  double zeta_;    // Штраф за движение через не-дорожные зоны (камера)

  // Ограничения
  double max_vel_x_, min_vel_x_, max_vel_th_;
  double acc_lim_x_, acc_lim_th_;

  // Параметры симуляции
  double predict_time_, dt_;
  int vx_samples_, vth_samples_;

  // Параметры цели
  double xy_goal_tolerance_, yaw_goal_tolerance_;
  double rot_stopped_vel_, trans_stopped_vel_;

  double speed_threshold_;
};

} // namespace improved_dwa_local_planner

#endif // IMPROVED_DWA_LOCAL_PLANNER_H_