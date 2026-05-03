#pragma once

#include <angles/angles.h>
#include <base_local_planner/costmap_model.h>
#include <base_local_planner/goal_functions.h>
#include <base_local_planner/odometry_helper_ros.h>
#include <costmap_2d/costmap_2d_ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Twist.h>
#include <nav_core/base_local_planner.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <ros/ros.h>
#include <mutex>
#include <tf2_ros/buffer.h>
#include <visualization_msgs/MarkerArray.h>

// Обогащённые данные об объектах из ScanCleaner
#include <tracked_obstacle_msgs/TrackedCircle.h>
#include <tracked_obstacle_msgs/TrackedCircleArray.h>

namespace improved_dwa_local_planner {

class ImprovedDWALocalPlanner : public nav_core::BaseLocalPlanner {
public:
  ImprovedDWALocalPlanner();
  ~ImprovedDWALocalPlanner();

  // --- nav_core::BaseLocalPlanner interface ---
  void initialize(std::string name, tf2_ros::Buffer *tf,
                  costmap_2d::Costmap2DROS *costmap_ros) override;
  bool setPlan(const std::vector<geometry_msgs::PoseStamped> &plan) override;
  bool isGoalReached() override;
  bool computeVelocityCommands(geometry_msgs::Twist &cmd_vel) override;


  struct Gap {
    double gap_center_x;
    double gap_center_y;
    double gap_width;
    double time_until_closed;  // Время до смыкания (секунды)
    bool is_viable;
  };


private:


  Gap findGapBetweenObjects(
    const tracked_obstacle_msgs::TrackedCircle &obs1,
    const tracked_obstacle_msgs::TrackedCircle &obs2,
    const geometry_msgs::PoseStamped &robot_pose);
  // --- ПАМЯТЬ ОБЪЕКТОВ ---
  std::map<int, ros::Time> passed_obstacles_;

  struct TrackedObstacle {
    int id;
    tracked_obstacle_msgs::TrackedCircle circle;
    ros::Time last_seen;
  };
  std::map<int, TrackedObstacle> obstacle_memory_;
  int generateObstacleId();
  int next_obstacle_id_ = 10000;

  double obstacle_memory_duration_;
  double prediction_horizon_;

  enum RejectReason {
    REJECT_NONE = 0,
    REJECT_STATIC = 1,
    REJECT_DYNAMIC = 2
  };

  // ---------------------------------------------------------------------------
  // СТРУКТУРА ТРАЕКТОРИИ
  // ---------------------------------------------------------------------------
  struct Trajectory {
    double vx = 0.0;
    double vth = 0.0;
    double cost = -1.0;
    std::vector<geometry_msgs::PoseStamped> poses;
    int collision_pose_idx = -1;
    geometry_msgs::Point obstacle_pos; // Предсказанная позиция объекта в момент удара
    geometry_msgs::Point obstacle_start_pos; // Начальная позиция объекта (для стрелки)
    
    // Debug variables
    double debug_alpha = 0.0;
    double debug_path = 0.0;
    double debug_heading = 0.0;
    double debug_velocity = 0.0;
    double debug_min_safety = 0.0;
    double debug_turning_bonus = 0.0;
    double debug_speed_bonus = 0.0;
    std::string debug_state = "IDLE";
    RejectReason reject_reason = REJECT_NONE;
  };

  // ---------------------------------------------------------------------------
  // CALLBACKS
  // ---------------------------------------------------------------------------

  // Подписка на /obstacles_tracked — обогащённые данные из ScanCleaner
  void trackedObstaclesCallback(
      const tracked_obstacle_msgs::TrackedCircleArray::ConstPtr &msg);

  // ---------------------------------------------------------------------------
  // АЛГОРИТМ DWA
  // ---------------------------------------------------------------------------
  Trajectory generateTrajectory(double vx, double vth);

  // Оценка траектории — возвращает положительное число (лучше = больше) или -1
  // при опасности
  double
  scoreTrajectory(Trajectory &traj,
                  const std::vector<geometry_msgs::PoseStamped> &local_plan);

  // ---------------------------------------------------------------------------
  // ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ОЦЕНКИ
  // ---------------------------------------------------------------------------

  // Оценка направления движения относительно вектора скорости объекта
  // Аргументы: скорость робота, поза робота, данные об объекте
  double calculate_dis_hv(const Trajectory &traj,
                          const geometry_msgs::PoseStamped &robot_pose,
                          const tracked_obstacle_msgs::TrackedCircle &obs);

  // Предсказание столкновения во времени.
  // safety_multiplier: 1.0 — норма, >1.0 — расширенная зона (для
  // ghost/unstable)
  double calculate_dis_fp(Trajectory &traj,
                          const tracked_obstacle_msgs::TrackedCircle &obs,
                          double safety_multiplier = 1.0);

  // Расчёт коэффициента безопасности на основе состояния объекта
  double getSafetyMultiplier(const tracked_obstacle_msgs::TrackedCircle &obs);

  // ---------------------------------------------------------------------------
  // ВИЗУАЛИЗАЦИЯ
  // ---------------------------------------------------------------------------
  void publishMarkers(const std::vector<Trajectory> &trajectories);
  void publishTrackedObstaclesViz();

  // ---------------------------------------------------------------------------
  // ROS-ИНФРАСТРУКТУРА
  // ---------------------------------------------------------------------------
  bool initialized_;
  tf2_ros::Buffer *tf_;
  costmap_2d::Costmap2DROS *costmap_ros_;

  ros::Subscriber tracked_obstacles_sub_;
  ros::Publisher local_plan_pub_;
  ros::Publisher candidate_trajs_pub_;
  ros::Publisher tracked_objects_pub_;
  ros::Publisher collision_markers_pub_;

  base_local_planner::OdometryHelperRos odom_helper_;

  // ---------------------------------------------------------------------------
  // СОСТОЯНИЕ
  // ---------------------------------------------------------------------------
  std::vector<geometry_msgs::PoseStamped> global_plan_;
  tracked_obstacle_msgs::TrackedCircleArray last_tracked_obstacles_;

  // Синхронизация доступа к last_tracked_obstacles_ между callback и планировщиком
  std::mutex obstacles_mutex_;

  // ---------------------------------------------------------------------------
  // ПАРАМЕТРЫ (загружаются из ROS param server в initialize())
  // ---------------------------------------------------------------------------
  // Веса стоимости
  double alpha_;   // Следование глобальному пути
  double beta_;    // Удаление от статических препятствий
  double gamma_;   // Поощрение целевой скорости
  double kappa_;   // Направление относительно динамических объектов
  double epsilon_; // Прогнозируемая безопасность (время до столкновения)

  // Физика робота
  double max_vel_x_;
  double min_vel_x_;
  double max_vel_th_;
  double acc_lim_x_;
  double acc_lim_th_;

  // Параметры симуляции
  double predict_time_;
  double dt_;
  int vx_samples_;
  int vth_samples_;

  // Допуски достижения цели
  double xy_goal_tolerance_;
  double yaw_goal_tolerance_;

  // Порог скорости: выше — динамический объект
  double speed_threshold_;

  // Новые параметры дистанции и безопасности
  double robot_radius_;
  double safe_clearance_dist_;
  double hard_collision_buffer_;
  double radar_range_;
  double safety_mult_unstable_;
  double safety_mult_ghost_;
  double safety_mult_ghost_unstable_;

  // Штраф за угловую скорость (предотвращает повороты 360°).
  // score -= rotation_cost_ * vth²
  double rotation_cost_;

  // Штраф за скорость сближения препятствия (relative approach velocity)
  double approach_penalty_weight_;

  // Горизонт времени для обнаружения 'коридора' столкновения
  double corridor_predict_time_;

  // Минимальная "крейсерская" скорость для успешного пересечения коридора
  double corridor_cruising_speed_;

  // Запас времени (с) для успешного проезда перед динамическим препятствием
  double safe_crossing_margin_;

  // Запас (м), после которого коридор перестает штрафовать траекторию
  double corridor_clear_margin_;

  // Разрешить fallback на менее безопасную траекторию при отсутствии валидных
  bool allow_dynamic_fallback_;

  // Масштаб скорости fallback-траектории
  double fallback_speed_scale_;

  // Минимальная безопасность для выбора fallback-траектории
  double fallback_min_safety_;
};

} // namespace improved_dwa_local_planner