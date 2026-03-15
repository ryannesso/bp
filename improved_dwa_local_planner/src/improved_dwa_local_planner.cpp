#include "improved_dwa_local_planner/improved_dwa_local_planner.h"
#include <angles/angles.h>
#include <base_local_planner/costmap_model.h>
#include <pluginlib/class_list_macros.h>
#include <tf2/utils.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

PLUGINLIB_EXPORT_CLASS(improved_dwa_local_planner::ImprovedDWALocalPlanner,
                       nav_core::BaseLocalPlanner)

namespace improved_dwa_local_planner {

// --- Конструктор и деструктор ---
ImprovedDWALocalPlanner::ImprovedDWALocalPlanner() : initialized_(false) {}
ImprovedDWALocalPlanner::~ImprovedDWALocalPlanner() {}

// --- Callback для получения данных об объектах от obstacle_detector ---
void ImprovedDWALocalPlanner::obstaclesCallback(
    const obstacle_detector::Obstacles::ConstPtr &msg) {
  last_obstacles_ = *msg;

  // Режим "Радара": если нет цели, все равно считаем коллизии для визуалки
  if (global_plan_.empty() && initialized_) {
    std::vector<Trajectory> debug_trajs;

    // Генерируем тестовый веер
    double step_x = max_vel_x_ / 4.0;
    double step_th = (2.0 * max_vel_th_) / 10.0;

    for (double vx = 0.0; vx <= max_vel_x_ + 1e-3; vx += step_x) {
      for (double vth = -max_vel_th_; vth <= max_vel_th_ + 1e-3;
           vth += step_th) {
        Trajectory traj = generateTrajectory(vx, vth);
        traj.cost = 1.0;

        for (const auto &obs : last_obstacles_.circles) {
          if (std::hypot(obs.velocity.x, obs.velocity.y) > speed_threshold_) {
            if (calculate_dis_fp(traj, obs) < 0.15) {
              traj.cost = -1.0;
            }
          }
        }
        debug_trajs.push_back(traj);
      }
    }
    publishMarkers(debug_trajs);
    publishTrackedObstacles();
  }
}

// --- Callback для получения сетки дороги от камеры ---
void ImprovedDWALocalPlanner::roadGridCallback(
    const nav_msgs::OccupancyGrid::ConstPtr &msg) {
  std::lock_guard<std::mutex> lock(road_grid_mutex_);
  road_grid_ = msg;
}

// --- Инициализация плагина ---
void ImprovedDWALocalPlanner::initialize(
    std::string name, tf2_ros::Buffer *tf,
    costmap_2d::Costmap2DROS *costmap_ros) {
  if (!initialized_) {
    ros::NodeHandle private_nh("~/" + name);
    ros::NodeHandle nh;

    tf_ = tf;
    costmap_ros_ = costmap_ros;

    // Подписка на топик с препятствиями
    obstacles_sub_ = nh.subscribe(
        "/obstacles", 1, &ImprovedDWALocalPlanner::obstaclesCallback, this);
    
    // Подписка на топик с детекцией дороги (камера)
    road_grid_sub_ = nh.subscribe(
        "/path_detector/road_grid", 1, &ImprovedDWALocalPlanner::roadGridCallback, this);

    // Издатели для визуализации и локального плана
    local_plan_pub_ = private_nh.advertise<nav_msgs::Path>("local_plan", 1);
    candidate_trajs_pub_ =
        private_nh.advertise<visualization_msgs::MarkerArray>(
            "candidate_trajectories", 1);
    tracked_objects_pub_ =
        private_nh.advertise<visualization_msgs::MarkerArray>("tracked_objects",
                                                              1);
    collision_markers_pub_ =
        private_nh.advertise<visualization_msgs::MarkerArray>(
            "collision_markers", 1);

    // --- Загрузка весовых коэффициентов стоимости ---
    private_nh.param("alpha", alpha_, 0.1); // Вес следования глобальному пути
    private_nh.param("beta", beta_,
                     10.0); // Вес удаления от статических препятствий
    private_nh.param("gamma", gamma_, 0.14); // Вес достижения целевой скорости
    private_nh.param("kappa", kappa_,
                     2.0); // Вес направления относительно динамических объектов
    private_nh.param("epsilon", epsilon_,
                     5.0); // Вес прогнозируемого времени до столкновения
    private_nh.param("zeta", zeta_, 5.0); // Штраф за движение через не-дорогу

    // --- Физические параметры робота ---
    private_nh.param("max_vel_x", max_vel_x_, 0.8);
    private_nh.param("min_vel_x", min_vel_x_, 0.0);
    private_nh.param("max_vel_th", max_vel_th_, 1.5);
    private_nh.param("acc_lim_x", acc_lim_x_, 1.5);
    private_nh.param("acc_lim_th", acc_lim_th_, 2.5);

    // --- Параметры симуляции ---
    private_nh.param("predict_time", predict_time_,
                     3.0);            // Горизонт прогнозирования в секундах
    private_nh.param("dt", dt_, 0.1); // Шаг симуляции
    private_nh.param("vx_samples", vx_samples_,
                     20); // Количество пробных линейных скоростей
    private_nh.param("vth_samples", vth_samples_,
                     30); // Количество пробных угловых скоростей

    // --- Допуски цели ---
    private_nh.param("xy_goal_tolerance", xy_goal_tolerance_, 0.25);
    private_nh.param("yaw_goal_tolerance", yaw_goal_tolerance_, 0.3);

    // Порог скорости для отделения динамических объектов от статических
    private_nh.param("speed_threshold", speed_threshold_, 0.1);

    odom_helper_.setOdomTopic("odom");
    initialized_ = true;
    ROS_INFO("Improved DWA with Spatio-Temporal Prediction initialized.");
  }
}

// --- Установка глобального плана для локального планировщика ---
bool ImprovedDWALocalPlanner::setPlan(
    const std::vector<geometry_msgs::PoseStamped> &plan) {
  if (!initialized_)
    return false;
  global_plan_ = plan;
  return true;
}

// --- Проверка достижения цели ---
bool ImprovedDWALocalPlanner::isGoalReached() {
  if (!initialized_)
    return false;
  geometry_msgs::PoseStamped global_pose;
  if (!costmap_ros_->getRobotPose(global_pose))
    return false;
  nav_msgs::Odometry base_odom;
  odom_helper_.getOdom(base_odom);
  return base_local_planner::isGoalReached(
      *tf_, global_plan_, *(costmap_ros_->getCostmap()),
      costmap_ros_->getGlobalFrameID(), global_pose, base_odom, 0.1, 0.1,
      xy_goal_tolerance_, yaw_goal_tolerance_);
}

// --- ГЛАВНЫЙ ЦИКЛ: Вычисление скоростей управления ---
bool ImprovedDWALocalPlanner::computeVelocityCommands(
    geometry_msgs::Twist &cmd_vel) {
  if (!initialized_)
    return false;

  // 1. Получение текущей позиции робота
  geometry_msgs::PoseStamped robot_pose;
  if (!costmap_ros_->getRobotPose(robot_pose))
    return false;

  // 2. Преобразование глобального плана в локальную систему координат
  std::vector<geometry_msgs::PoseStamped> local_plan;
  if (!base_local_planner::transformGlobalPlan(
          *tf_, global_plan_, robot_pose, *costmap_ros_->getCostmap(),
          costmap_ros_->getGlobalFrameID(), local_plan))
    return false;

  // 3. Получение текущей одометрии (скорости)
  nav_msgs::Odometry base_odom;
  odom_helper_.getOdom(base_odom);
  geometry_msgs::Twist current_vel = base_odom.twist.twist;

  // 4. Расчет Динамического Окна (Dynamic Window)
  // Мы ищем доступные скорости в пределах возможностей ускорения робота
  double sim_accel_time = 0.5; // Время, за которое оценивается разгон

  double min_vx =
      std::max(min_vel_x_, current_vel.linear.x - acc_lim_x_ * sim_accel_time);
  double max_vx =
      std::min(max_vel_x_, current_vel.linear.x + acc_lim_x_ * sim_accel_time);
  double min_vth = std::max(-max_vel_th_, current_vel.angular.z -
                                              acc_lim_th_ * sim_accel_time);
  double max_vth = std::min(max_vel_th_, current_vel.angular.z +
                                             acc_lim_th_ * sim_accel_time);

  Trajectory best_traj;
  best_traj.cost = -1.0;
  std::vector<Trajectory> all_trajectories;

  // Шаг перебора скоростей
  double dvx = (vx_samples_ > 1) ? (max_vx - min_vx) / (vx_samples_ - 1) : 0.01;
  double dvth =
      (vth_samples_ > 1) ? (max_vth - min_vth) / (vth_samples_ - 1) : 0.01;

  // 5. ПЕРЕБОР ТРАЕКТОРИЙ: Симуляция каждого варианта (v, w)
  for (double vx = min_vx; vx <= max_vx + 1e-6; vx += dvx) {
    for (double vth = min_vth; vth <= max_vth + 1e-6; vth += dvth) {
      Trajectory traj = generateTrajectory(vx, vth);
      if (traj.poses.empty())
        continue;

      // Оценка стоимости траектории (чем больше, тем лучше)
      traj.cost = scoreTrajectory(traj, local_plan);
      all_trajectories.push_back(traj);

      // Сохранение лучшего варианта
      if (traj.cost >= 0.0 && traj.cost > best_traj.cost) {
        best_traj = traj;
      }
    }
  }

  // Публикация всех вееров траекторий в RViz для отладки
  publishMarkers(all_trajectories);

  // Если безопасная траектория не найдена — экстренная остановка
  if (best_traj.cost < 0) {
    cmd_vel.linear.x = 0;
    cmd_vel.angular.z = 0;
    ROS_INFO("STOP");
    return false;
  }

  // 6. Установка вычисленных скоростей в команду
  cmd_vel.linear.x = best_traj.vx;
  cmd_vel.angular.z = best_traj.vth;

  // Визуализация выбранного пути
  nav_msgs::Path local_path;
  local_path.header.stamp = ros::Time::now();
  local_path.header.frame_id = costmap_ros_->getGlobalFrameID();
  local_path.poses = best_traj.poses;
  local_plan_pub_.publish(local_path);

  // Периодический вывод отладочной информации
  static ros::Time last_log = ros::Time(0);
  if ((ros::Time::now() - last_log).toSec() > 1.0) {
    last_log = ros::Time::now();
    int dyn_count = 0;
    for (const auto &o : last_obstacles_.circles)
      if (std::hypot(o.velocity.x, o.velocity.y) > speed_threshold_)
        dyn_count++;
    // ROS_INFO("[DWA] V=%.2f | W=%.2f | DynObj=%d", best_traj.vx,
    // best_traj.vth,
    //          dyn_count);
  }

  publishTrackedObstacles();

  return true;
}

// --- [SCORING] Оценка стоимости траектории ---
// Чем выше результат, тем предпочтительнее траектория.
double ImprovedDWALocalPlanner::scoreTrajectory(
    Trajectory &traj,
    const std::vector<geometry_msgs::PoseStamped> &local_plan) {

  // 1. Проверка на столкновение со статической картой (Costmap)
  base_local_planner::CostmapModel world_model(*(costmap_ros_->getCostmap()));
  double max_footprint_cost = 0.0;

  for (size_t i = 0; i < traj.poses.size(); ++i) {
    const auto &pose = traj.poses[i];
    double pt_cost = world_model.footprintCost(
        pose.pose.position.x, pose.pose.position.y,
        tf2::getYaw(pose.pose.orientation), costmap_ros_->getRobotFootprint());

    // ЖЕСТКИЙ ЗАПРЕТ: Если стоимость точки >= 253 (INSCRIBED_INFLATED_OBSTACLE),
    // мы категорически отбраковываем эту траекторию!
    if (pt_cost < 0 || pt_cost >= 253) {
      traj.collision_pose_idx = (int)i;
      return -1.0; // УБИЙСТВЕННАЯ ОТБРАКОВКА ТРАЕКТОРИИ
    }
    if (pt_cost > max_footprint_cost)
      max_footprint_cost = pt_cost;
  }

  // 1b. Проверка на движение по не-дорожным зонам (трава/камера)
  double road_cost = 0.0;
  {
    std::lock_guard<std::mutex> lock(road_grid_mutex_);
    if (road_grid_ && !traj.poses.empty()) {
      const auto &info = road_grid_->info;
      int hit_count = 0;

      // Получаем трансформ из кадровой системы траектории (global) в систему сетки (base_link)
      geometry_msgs::TransformStamped transform;
      bool has_transform = false;
      std::string global_frame = costmap_ros_->getGlobalFrameID();
      
      try {
        transform = tf_->lookupTransform(road_grid_->header.frame_id, 
                                        global_frame, 
                                        ros::Time(0), // Берем последний доступный трансформ
                                        ros::Duration(0.01));
        has_transform = true;
      } catch (tf2::TransformException &ex) {
        ROS_WARN_THROTTLE(5.0, "Road grid transform failed: %s", ex.what());
      }

      if (has_transform) {
        for (const auto &pose : traj.poses) {
          // Трансформируем точку траектории в систему координат сетки (base_link)
          geometry_msgs::PoseStamped pose_in;
          geometry_msgs::PoseStamped pose_local;
          pose_in.header.frame_id = global_frame;
          pose_in.header.stamp = ros::Time(0);
          // Проверяем не только центр, но и 4 угла робота (footprint)
          // Примерные размеры: +/- 0.3м по X, +/- 0.4м по Y
          std::vector<std::pair<double, double>> footprint_offsets = {
            {0.0, 0.0},   // Центр
            {0.3, 0.4},   // Передний левый
            {0.3, -0.4},  // Передний правый
            {-0.3, 0.4},  // Задний левый
            {-0.3, -0.4}  // Задний правый
          };

          for (const auto& offset : footprint_offsets) {
            geometry_msgs::PoseStamped pose_in;
            geometry_msgs::PoseStamped pose_local;
            pose_in.header.frame_id = global_frame;
            pose_in.header.stamp = ros::Time(0);
            
            // Смещение в системе координат траектории (global)
            double yaw = tf2::getYaw(pose.pose.orientation);
            pose_in.pose.position.x = pose.pose.position.x + offset.first * cos(yaw) - offset.second * sin(yaw);
            pose_in.pose.position.y = pose.pose.position.y + offset.first * sin(yaw) + offset.second * cos(yaw);
            pose_in.pose.orientation = pose.pose.orientation;

            tf2::doTransform(pose_in, pose_local, transform);

            int my = (int)((pose_local.pose.position.x - info.origin.position.x) / info.resolution);
            int mx = (int)((pose_local.pose.position.y - info.origin.position.y) / info.resolution);
            
            if (mx >= 0 && mx < (int)info.width && my >= 0 && my < (int)info.height) {
              int idx = my * info.width + mx;
              if (road_grid_->data[idx] > 50) { 
                return -1.0; // Любая часть корпуса на траве = отказ
              }
            }
          }
        }
      }
    }
  }

  geometry_msgs::PoseStamped robot_pose;
  costmap_ros_->getRobotPose(robot_pose);

  // 2. Оценка близости к глобальному пути (Alpha)
  int closest_idx = 0;
  double min_dist_sq = -1.0;
  for (int i = 0; i < local_plan.size(); ++i) {
    double d2 = std::pow(traj.poses.back().pose.position.x -
                             local_plan[i].pose.position.x,
                         2) +
                std::pow(traj.poses.back().pose.position.y -
                             local_plan[i].pose.position.y,
                         2);
    if (min_dist_sq < 0 || d2 < min_dist_sq) {
      min_dist_sq = d2;
      closest_idx = i;
    }
  }
  double path_dist_score = 1.0 / (1.0 + std::sqrt(min_dist_sq));
  double angle_diff = std::abs(angles::shortest_angular_distance(
      tf2::getYaw(traj.poses.back().pose.orientation),
      tf2::getYaw(local_plan[closest_idx].pose.orientation)));
  double heading_score = (M_PI - angle_diff) / M_PI;

  // 3. Штраф за близость к статическим препятствиям (Beta)
  double dist_score =
      1.0 - (max_footprint_cost / costmap_2d::INSCRIBED_INFLATED_OBSTACLE);

  // 4. Оценка скорости (Gamma) — поощряем движение к цели
  double dist_to_goal = std::hypot(
      robot_pose.pose.position.x - global_plan_.back().pose.position.x,
      robot_pose.pose.position.y - global_plan_.back().pose.position.y);

  // Вычисление угла к цели (разница между текущим направлением и направлением
  // на цель)
  double robot_yaw = tf2::getYaw(robot_pose.pose.orientation);
  double angle_to_goal =
      atan2(global_plan_.back().pose.position.y - robot_pose.pose.position.y,
            global_plan_.back().pose.position.x - robot_pose.pose.position.x);
  double angle_diff_to_goal =
      std::abs(angles::shortest_angular_distance(robot_yaw, angle_to_goal));

  double velocity_score = 0.0;
  if (dist_to_goal > 1.0) {
    velocity_score = traj.vx / max_vel_x_;
  } else {
    // Плавное замедление при подходе к цели
    double desired_vel = max_vel_x_ * (dist_to_goal / 1.0);
    velocity_score = 1.0 - (std::abs(traj.vx - desired_vel) / max_vel_x_);
  }

  // --- НОВОЕ: Оценка отклонения траектории от цели ---
  // Вычисляем куда ПРИВЕДЁТ траектория (конечная точка)
  double traj_end_x = traj.poses.back().pose.position.x;
  double traj_end_y = traj.poses.back().pose.position.y;

  // Угол от текущей позиции к конечной точке траектории
  double angle_of_trajectory = atan2(traj_end_y - robot_pose.pose.position.y,
                                     traj_end_x - robot_pose.pose.position.x);

  // Насколько траектория отклоняется от направления на цель
  double trajectory_deviation_from_goal = std::abs(
      angles::shortest_angular_distance(angle_of_trajectory, angle_to_goal));

  // Оценка: чем меньше отклонение, тем лучше (0°=1.0, 90°=0.0, 180°=-1.0)
  double goal_alignment_score = 1.0 - (trajectory_deviation_from_goal / M_PI);

  // --- 5. ДИНАМИЧЕСКИЙ АНАЛИЗ (Оценка угроз) ---
  double min_safety_score = 1.0;
  double critical_heading_score = 0.0;
  const obstacle_detector::CircleObstacle *critical_obs = nullptr;

  // ОТЛАДКА: Считаем динамические объекты
  int total_obs = last_obstacles_.circles.size();
  int dynamic_obs_count = 0;
  for (const auto &obs : last_obstacles_.circles) {
    double v_obs = std::hypot(obs.velocity.x, obs.velocity.y);
    if (v_obs > speed_threshold_) {
      dynamic_obs_count++;
    }
  }

  ROS_INFO_THROTTLE(0.2, "[DEBUG] Total obstacles: %d | Dynamic (v>%.2f): %d",
                    total_obs, speed_threshold_, dynamic_obs_count);

  // Поиск самого опасного динамического объекта для этой траектории
  for (const auto &obs : last_obstacles_.circles) {
    double v_obs = std::hypot(obs.velocity.x, obs.velocity.y);

    if (v_obs > speed_threshold_) {
      // Предсказание пересечения траекторий во времени
      double safety = calculate_dis_fp(traj, obs);

      if (safety < min_safety_score) {
        min_safety_score = safety;
        critical_obs = &obs;
      }
    }
  }

  // Оценка направления движения относительно самой опасной угрозы
  if (critical_obs != nullptr) {
    critical_heading_score = calculate_dis_hv(
        traj.vx, robot_pose,
        *critical_obs); // оценка траектории у которой параметр меньше чем у
                        // других. параметр отвечает за дистанцию объекта к
                        // роботу относительную скорость робота/объекта и угол
                        // встречи
  }

  // --- 6. АДАПТИВНАЯ ЛОГИКА ПОВЕДЕНИЯ (Трёхфазная система) ---
  double current_alpha = alpha_;
  double turning_bonus = 0.0;
  double speed_bonus = 0.0;

  if (critical_obs != nullptr) {
    // Расчет углов и векторов относительно препятствия
    double robot_yaw = tf2::getYaw(robot_pose.pose.orientation);
    double angle_to_obs =
        atan2(critical_obs->center.y - robot_pose.pose.position.y,
              critical_obs->center.x - robot_pose.pose.position.x);
    double angle_diff =
        std::abs(angles::shortest_angular_distance(robot_yaw, angle_to_obs));

    double in_front_factor = std::max(0.0, 1.0 - (angle_diff / 1.0));
    double angle_to_robot =
        atan2(robot_pose.pose.position.y - critical_obs->center.y,
              robot_pose.pose.position.x - critical_obs->center.x);
    double obs_vel_angle =
        atan2(critical_obs->velocity.y,
              critical_obs->velocity.x); // направление движения объекта

    // Фактор того, насколько объект движется точно в сторону робота
    double toward_factor = std::max(0.0, cos(angles::shortest_angular_distance(
                                             obs_vel_angle, angle_to_robot)));

    // Коэффициент агрессивности угрозы (Лобовое столкновение)
    double aggressive_danger = in_front_factor * toward_factor;
    double evade_side = 0.0;
    if (angles::shortest_angular_distance(robot_yaw, angle_to_obs) > 0) {
      evade_side = -1.0;
    } else {
      evade_side = 1.0;
    }

    if (aggressive_danger > 0.5) {
      current_alpha *= 0.3;
      if (traj.vth * evade_side > 0) {
        turning_bonus += 8.0 * std::abs(traj.vth);
      }
      speed_bonus += 5.0;
    } else if (aggressive_danger < 0.5) {
      if ((traj.vth *
           angles::shortest_angular_distance(robot_yaw, angle_to_goal)) > 0) {
        turning_bonus += 5.0;
        speed_bonus += 5.0;
      }
      turning_bonus += 2.0;
      speed_bonus += 2.0;
      if (current_alpha < alpha_) {
        current_alpha += 2.0 * std::abs(angles::shortest_angular_distance(
                                   robot_yaw, angle_to_goal));
      }
    }

    // ПРИНУДИТЕЛЬНЫЙ S-МАНЕВР: Если робот уже отвернулся от препятствия
    // достаточно (угол > 45°), СРАЗУ возвращаемся к цели, не дожидаясь снижения
    // aggressive_danger
    // bool already_evaded = (angle_diff > 0.78); // ~45 градусов (СНИЖЕНО с
    // 70°)

    // === ФАЗА 1: ТАКТИЧЕСКОЕ СМЕЩЕНИЕ (Обход по касательной) ===
    // if (aggressive_danger > 0.8 && !already_evaded) {
    //   current_alpha = alpha_ * 0.4; // Держим связь с планом чуть сильнее

    //   // 1. Выбираем сторону (где больше места до цели)
    //   double evade_side =
    //       (angles::shortest_angular_distance(robot_yaw, angle_to_goal) > 0)
    //           ? 1.0
    //           : -1.0;

    //   // 2. Вместо одной точки 45°, создаем "зону комфорта" ухода
    //   double target_evade_heading =
    //       angles::normalize_angle(angle_to_obs + evade_side * 0.78);
    //   double traj_final_yaw =
    //   tf2::getYaw(traj.poses.back().pose.orientation); double angle_err =
    //   std::abs(angles::shortest_angular_distance(
    //       traj_final_yaw, target_evade_heading));

    //   // Бонус за направление: широкий диапазон (плавнее поворот)
    //   double alignment_score =
    //       std::max(0.0, 1.0 - (angle_err / 0.6)); // Коридор ~35 градусов

    //   // 3. СКОРОСТЬ — ЭТО ЖИЗНЬ: жестко наказываем за попытку остановиться
    //   if (traj.vx < 0.4) {
    //     speed_bonus = -10.0; // Штраф за замирание
    //   } else {
    //     speed_bonus = 30.0 * (traj.vx / max_vel_x_); // Огромный стимул ехать
    //   }

    //   turning_bonus = 15.0 * alignment_score * (traj.vx / max_vel_x_);

    //   if ((traj.vth *
    //        angles::shortest_angular_distance(robot_yaw, angle_to_goal)) > 0)
    //        {
    //     turning_bonus += 5.0 * std::abs(traj.vth) * (traj.vx / max_vel_x_);
    //   }

    //   // Дополнительный бонус за безопасность траектории
    //   if (min_safety_score > 0.6) {
    //     turning_bonus += 10.0;
    //   }
    //   ROS_INFO_THROTTLE(
    //       0.1, "S1 | DWA: Alpha=%.2f | SpeedBonus=%.2f | TurnBonus=%.2f",
    //       current_alpha, speed_bonus, turning_bonus);

    // }
    // // === ФАЗА 2: ДИНАМИЧЕСКИЙ ВОЗВРАТ ===
    // else if (already_evaded || aggressive_danger < 0.65) {
    //   // Плавный коэффициент возврата
    //   double recovery_mult = already_evaded ? 10.0 : 8.0;
    //   current_alpha = alpha_ * recovery_mult;

    //   // Приоритет скорости сохраняем, чтобы не тупил при возврате
    //   if (traj.vx < 0.1) {
    //     speed_bonus -= 10.0 * (1.0 - (traj.vx / max_vel_x_));
    //   } else {
    //     speed_bonus += 10.0 * (traj.vx / max_vel_x_);
    //   }
    //   // Поворот к цели
    //   double drd = angles::shortest_angular_distance(robot_yaw,
    //   angle_to_goal); if ((traj.vth * drd) > 0) {
    //     turning_bonus += 22.0 * std::abs(traj.vth) * (traj.vx / max_vel_x_);
    //   }

    //   // Бонус за выравнивание на цель
    //   if (goal_alignment_score > 0.7) {
    //     turning_bonus += 20.0 * goal_alignment_score;
    //   }

    //   turning_bonus += 2.0 * std::abs(traj.vth);
    //   ROS_INFO_THROTTLE(
    //       0.1, "S2 | DWA: Alpha=%.2f | SpeedBonus=%.2f | TurnBonus=%.2f",
    //       current_alpha, speed_bonus, turning_bonus);

    // }
    // // === ФАЗА 3: ОБЫЧНОЕ СБЛИЖЕНИЕ ===
    // else {
    //   bool is_unsafe_proximity = (min_safety_score < 0.9);

    //   if ((traj.vth *
    //        angles::shortest_angular_distance(robot_yaw, angle_to_goal)) > 0)
    //        {
    //     turning_bonus += 15.0 * std::abs(traj.vth) * (traj.vx / max_vel_x_);
    //   } else {
    //     turning_bonus -= 10.0 * (1.0 - (traj.vx / max_vel_x_));
    //   }
    //   if (is_unsafe_proximity) {
    //     current_alpha = alpha_ * 0.3;
    //   } else {
    //     current_alpha = alpha_ * (1.0 - in_front_factor);
    //   }

    //   if (traj.vx < 0.1) {
    //     speed_bonus -= 10.0;
    //   } else {
    //     speed_bonus += 10.0 * (traj.vx / max_vel_x_);
    //   }

    //   if (goal_alignment_score > 0.7) {
    //     turning_bonus += 20.0 * goal_alignment_score;
    //   }

    // Увеличенные бонусы для плавности
    //   turning_bonus += 4.0 * std::abs(traj.vth) * (traj.vx / max_vel_x_);
    //   speed_bonus += 5.0 * (traj.vx / max_vel_x_);
    //   ROS_INFO_THROTTLE(0.1,
    //                     "S3 | DWA: Alpha=%.2f | SpeedBonus=%.2f |
    //                     TurnBonus=%.2f", current_alpha, speed_bonus,
    //                     turning_bonus);
    // }

    // // ЛОГИКА: "Проезд сзади" (Cut Behind)
    // double cutting_behind_bonus = 0.0;
    // double rel_obs_move_angle =
    //     angles::shortest_angular_distance(robot_yaw, obs_vel_angle);

    // if (std::abs(rel_obs_move_angle) > 0.5 &&
    //     std::abs(rel_obs_move_angle) < 2.6) {
    //   if (rel_obs_move_angle * traj.vth < 0) {
    //     // Поощряем только если едет (не стоит на месте)
    //     cutting_behind_bonus = 12.0 * std::abs(traj.vth) * (traj.vx /
    //     max_vel_x_);
    //   }
    // }

    // turning_bonus += cutting_behind_bonus;

    // // Отладочный вывод текущей фазы
    // const char *current_phase = "NONE";
    // if (aggressive_danger > 0.8 && !already_evaded) {
    //   current_phase = "S1";
    // } else if (already_evaded || aggressive_danger < 0.65) {
    //   if (already_evaded) {
    //     current_phase = "S2";
    //   } else {
    //     current_phase = "S2_2";
    //   }
    // } else {
    //   current_phase = "S3";
    // }

    // сделать проезд сзади!!!!!!!!!!!!

    ROS_INFO_THROTTLE(0.1, "aggresive: %.2f", aggressive_danger);
  }

  // Если столкновение неизбежно — красим в красный
  if (min_safety_score < 0.05) {
    return -100.0 + min_safety_score;
  }

  // Финальное суммирование всех весов
  return (current_alpha * (path_dist_score + heading_score)) +
         (beta_ * dist_score) + (gamma_ * velocity_score) +
         (kappa_ * critical_heading_score) + (epsilon_ * min_safety_score) +
         turning_bonus + speed_bonus;
}

// --- [HELPER] Расчет фактора направления движения ---
// Оценивает, насколько "безопасно" направление движения относительно вектора
// скорости объекта.
double ImprovedDWALocalPlanner::calculate_dis_hv(
    double robot_vx, const geometry_msgs::PoseStamped &robot_pose,
    const obstacle_detector::CircleObstacle &obs) {
  double obs_speed = std::hypot(obs.velocity.x, obs.velocity.y);
  double obs_heading = atan2(obs.velocity.y, obs.velocity.x);
  double d = std::hypot(obs.center.x - robot_pose.pose.position.x,
                        obs.center.y - robot_pose.pose.position.y);
  double eta = std::max(
      0.0, 4.0 - d); // Бонус за расстояние (чем ближе, тем важнее множитель)
  double theta = angles::shortest_angular_distance(
      tf2::getYaw(robot_pose.pose.orientation), obs_heading);

  // Коэффициент сонаправленности векторов скоростей
  double heading_factor = 0.5 * (1.0 + cos(theta));

  double obs_speed_mag = std::hypot(obs.velocity.x, obs.velocity.y);
  double rel_speed_proxy = std::abs(robot_vx) + obs_speed_mag;

  return eta * rel_speed_proxy * heading_factor;
}

// --- [HELPER] Прогноз будущего положения (Collision Checking) ---
// Симулирует взаимное движение робота и объекта во времени.
double ImprovedDWALocalPlanner::calculate_dis_fp(
    Trajectory &traj, const obstacle_detector::CircleObstacle &obs) {
  double min_dist_sq = 1e6;
  double collision_time = -1.0;

  for (size_t i = 0; i < traj.poses.size(); ++i) {
    double t = i * dt_;

    // Позиция объекта в момент времени t (линейное предсказание)
    double pred_obs_x = obs.center.x + obs.velocity.x * t;
    double pred_obs_y = obs.center.y + obs.velocity.y * t;

    double dx = traj.poses[i].pose.position.x - pred_obs_x;
    double dy = traj.poses[i].pose.position.y - pred_obs_y;
    double d2 = dx * dx + dy * dy;

    if (d2 < min_dist_sq)
      min_dist_sq = d2;

    // Проверка на "жесткое" столкновение с учетом радиусов и отступа 0.55м
    double hard_limit = obs.radius + 0.2;
    if (d2 <= std::pow(hard_limit, 2)) {
      if (collision_time < 0) {
        collision_time = t; // Первый момент контакта
        traj.collision_pose_idx = (int)i;
        traj.obstacle_pos.x = pred_obs_x;
        traj.obstacle_pos.y = pred_obs_y;
        traj.obstacle_pos.z = 0;
        traj.obstacle_start_pos = obs.center;
      }
    }
  }

  // Если столкновение неизбежно, штрафуем (чем раньше удар, тем хуже)
  if (collision_time >= 0) {
    return (collision_time / predict_time_) * 0.1;
  }

  // Если столкновения нет, применяем мягкий штраф за дистанцию (Gradient)
  double min_dist = std::sqrt(min_dist_sq);

  double safe_clearance_dist = 0.3; // Желаемый отступ
  double robot_radius_approx = 0.3; // Примерный радиус робота

  double soft_limit = obs.radius + robot_radius_approx +
                      safe_clearance_dist; // ~0.8м от центра до центра
  double hard_limit = obs.radius + robot_radius_approx + 0.05; // Порог касания

  if (min_dist < soft_limit) {
    // Нелинейный штраф при вхождении в "зону безопасности"
    double score = (min_dist - hard_limit) / (soft_limit - hard_limit);
    if (score < 0)
      score = 0;
    return 0.1 + (0.9 * std::pow(score, 2));
  }

  return 1.0; // Безопасно
}

// --- [GENERATION] Генерация траектории робота ---
// Используется для предсказания пути при выбранных (v, w)
ImprovedDWALocalPlanner::Trajectory
ImprovedDWALocalPlanner::generateTrajectory(double vx, double vth) {
  Trajectory traj;
  traj.vx = vx;
  traj.vth = vth;
  geometry_msgs::PoseStamped cp;
  costmap_ros_->getRobotPose(cp);
  double x = cp.pose.position.x, y = cp.pose.position.y,
         th = tf2::getYaw(cp.pose.orientation);

  for (double t = 0; t <= predict_time_; t += dt_) {
    x += vx * cos(th) * dt_;
    y += vx * sin(th) * dt_;
    th += vth * dt_;
    geometry_msgs::PoseStamped p;
    p.header.frame_id = costmap_ros_->getGlobalFrameID();
    p.pose.position.x = x;
    p.pose.position.y = y;
    tf2::Quaternion q;
    q.setRPY(0, 0, th);
    p.pose.orientation = tf2::toMsg(q);
    traj.poses.push_back(p);
  }
  return traj;
}

// --- [VIS] Публикация веера траекторий в RViz ---
void ImprovedDWALocalPlanner::publishMarkers(
    const std::vector<Trajectory> &trajectories) {
  visualization_msgs::MarkerArray markers;
  visualization_msgs::Marker clear;
  clear.ns = "dwa";
  clear.action = 3; // DELETEALL
  markers.markers.push_back(clear);
  for (size_t i = 0; i < trajectories.size(); ++i) {
    visualization_msgs::Marker m;
    m.header.frame_id = costmap_ros_->getGlobalFrameID();
    m.header.stamp = ros::Time::now();
    m.ns = "dwa";
    m.id = i;
    m.type = 4; // LINE_STRIP
    m.scale.x = 0.01;
    m.color.a = 0.2;
    if (trajectories[i].cost < 0) {
      m.color.r = 1.0; // Красный — коллизия
    } else {
      m.color.g = 1.0; // Зеленый — доступный маршрут
    }
    for (const auto &p : trajectories[i].poses)
      m.points.push_back(p.pose.position);
    markers.markers.push_back(m);
  }
  candidate_trajs_pub_.publish(markers);

  // --- Визуализация точек коллизий ---
  visualization_msgs::MarkerArray collision_markers;

  visualization_msgs::Marker clear_collision;
  clear_collision.ns = "collisions";
  clear_collision.action = 3; // DELETEALL
  collision_markers.markers.push_back(clear_collision);

  int marker_id = 0;
  for (const auto &traj : trajectories) {
    if (traj.cost < 0 && traj.collision_pose_idx >= 0 &&
        traj.collision_pose_idx < (int)traj.poses.size()) {
      // 1. Позиция робота при столкновении (Красная сфера)
      visualization_msgs::Marker m_robot;
      m_robot.header.frame_id = costmap_ros_->getGlobalFrameID();
      m_robot.header.stamp = ros::Time::now();
      m_robot.ns = "collisions";
      m_robot.id = marker_id++;
      m_robot.type = visualization_msgs::Marker::SPHERE;
      m_robot.pose = traj.poses[traj.collision_pose_idx].pose;
      m_robot.scale.x = 0.3;
      m_robot.scale.y = 0.3;
      m_robot.scale.z = 0.3;
      m_robot.color.r = 1.0;
      m_robot.color.g = 0.0;
      m_robot.color.b = 0.0;
      m_robot.color.a = 0.8;
      collision_markers.markers.push_back(m_robot);

      // 2. Позиция препятствия при столкновении (Оранжевая сфера)
      visualization_msgs::Marker m_obs;
      m_obs.header = m_robot.header;
      m_obs.ns = "collisions";
      m_obs.id = marker_id++;
      m_obs.type = visualization_msgs::Marker::SPHERE;
      m_obs.pose.position = traj.obstacle_pos;
      m_obs.pose.orientation.w = 1.0;
      m_obs.scale.x = 0.4;
      m_obs.scale.y = 0.4;
      m_obs.scale.z = 0.4;
      m_obs.color.r = 1.0;
      m_obs.color.g = 0.5;
      m_obs.color.b = 0.0;
      m_obs.color.a = 0.6;
      collision_markers.markers.push_back(m_obs);

      // 3. Линия от текущей позиции объекта до точки коллизии (LINE_LIST)
      visualization_msgs::Marker m_line;
      m_line.header = m_robot.header;
      m_line.ns = "collision_vectors";
      m_line.id = marker_id++;
      m_line.type = visualization_msgs::Marker::LINE_LIST;
      m_line.action = 0;
      m_line.scale.x = 0.02; // Толщина линии
      m_line.color.r = 1.0;
      m_line.color.g = 1.0;
      m_line.color.b = 0.0;
      m_line.color.a = 0.5;

      m_line.points.push_back(traj.obstacle_start_pos);
      m_line.points.push_back(traj.obstacle_pos);
      collision_markers.markers.push_back(m_line);
    }
  }
  collision_markers_pub_.publish(collision_markers);
}

// --- [VIS] Публикация данных об отслеживаемых препятствиях ---
void ImprovedDWALocalPlanner::publishTrackedObstacles() {
  visualization_msgs::MarkerArray markers;

  visualization_msgs::Marker clear_msg;
  clear_msg.action = 3;
  clear_msg.ns = "tracked_obstacles";
  markers.markers.push_back(clear_msg);

  int marker_id = 1000; // Начинаем с высокого ID, чтобы не пересекаться

  for (const auto &obs : last_obstacles_.circles) {
    double v = std::hypot(obs.velocity.x, obs.velocity.y);
    if (v < speed_threshold_)
      continue;

    // 1. Маркер текста: ID объекта
    visualization_msgs::Marker text;
    text.header.frame_id = costmap_ros_->getGlobalFrameID();
    text.header.stamp = ros::Time::now();
    text.ns = "tracked_obstacles";
    text.id = obs.id;
    text.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
    text.action = 0;
    text.pose.position = obs.center;
    text.pose.position.z += 1.0;
    text.scale.z = 0.4;
    text.color.r = 1.0;
    text.color.g = 1.0;
    text.color.b = 1.0;
    text.color.a = 1.0;
    text.text = "ID: " + std::to_string(obs.id);
    markers.markers.push_back(text);

    // 2. Маркер цилиндра: Само тело препятствия
    visualization_msgs::Marker body;
    body.header = text.header;
    body.ns = "tracked_obstacles_body";
    body.id = obs.id;
    body.type = visualization_msgs::Marker::CYLINDER;
    body.action = 0;
    body.pose.position = obs.center;
    body.pose.position.z = 0.1;
    body.scale.x = obs.radius * 2.0;
    body.scale.y = obs.radius * 2.0;
    body.scale.z = 0.2;
    body.color.r = 1.0;
    body.color.g = 0.0;
    body.color.b = 0.0;
    body.color.a = 0.8;
    markers.markers.push_back(body);

    // 3. Маркер цилиндра: Зона "Hard Limit" (Буфер коллизии)
    // В коде это: obs.radius + 0.2 (согласно строке 628)
    double hard_limit = obs.radius + 0.2;
    visualization_msgs::Marker buffer_hard;
    buffer_hard.header = text.header;
    buffer_hard.ns = "tracked_obstacles_hard_limit";
    buffer_hard.id = obs.id;
    buffer_hard.type = visualization_msgs::Marker::CYLINDER;
    buffer_hard.action = 0;
    buffer_hard.pose.position = obs.center;
    buffer_hard.pose.position.z = 0.05;
    buffer_hard.scale.x = hard_limit * 2.0;
    buffer_hard.scale.y = hard_limit * 2.0;
    buffer_hard.scale.z = 0.1;
    buffer_hard.color.r = 1.0;
    buffer_hard.color.g = 0.0;
    buffer_hard.color.b = 0.0;
    buffer_hard.color.a = 0.2; // Бледно-красный
    markers.markers.push_back(buffer_hard);

    // 4. Маркер цилиндра: Зона "Soft Limit" (Зона осторожности)
    // В коде это: obs.radius + 0.3 + 0.3 (согласно строке 651)
    double soft_limit = obs.radius + 0.3 + 0.3;
    visualization_msgs::Marker buffer_soft;
    buffer_soft.header = text.header;
    buffer_soft.ns = "tracked_obstacles_soft_limit";
    buffer_soft.id = obs.id;
    buffer_soft.type = visualization_msgs::Marker::CYLINDER;
    buffer_soft.action = 0;
    buffer_soft.pose.position = obs.center;
    buffer_soft.pose.position.z = 0.02;
    buffer_soft.scale.x = soft_limit * 2.0;
    buffer_soft.scale.y = soft_limit * 2.0;
    buffer_soft.scale.z = 0.05;
    buffer_soft.color.r = 1.0;
    buffer_soft.color.g = 1.0;
    buffer_soft.color.b = 0.0;
    buffer_soft.color.a = 0.1; // Бледно-желтый
    markers.markers.push_back(buffer_soft);

    // 5. Маркер стрелки: Вектор скорости объекта
    visualization_msgs::Marker arrow;
    arrow.header = text.header;
    arrow.ns = "tracked_obstacles_vel";
    arrow.id = obs.id;
    arrow.type = visualization_msgs::Marker::ARROW;
    arrow.action = 0;
    arrow.scale.x = 0.05;
    arrow.scale.y = 0.1;
    arrow.scale.z = 0.1;
    arrow.color.r = 1.0;
    arrow.color.g = 0.0;
    arrow.color.b = 0.0;
    arrow.color.a = 1.0;

    geometry_msgs::Point p1 = obs.center;
    geometry_msgs::Point p2;
    p2.x = p1.x + obs.velocity.x;
    p2.y = p1.y + obs.velocity.y;
    p2.z = p1.z;

    arrow.points.push_back(p1);
    arrow.points.push_back(p2);
    markers.markers.push_back(arrow);
  }
  tracked_objects_pub_.publish(markers);
}

} // namespace improved_dwa_local_planner