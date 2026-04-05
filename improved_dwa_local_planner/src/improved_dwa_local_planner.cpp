#include "improved_dwa_local_planner/improved_dwa_local_planner.h"
#include <angles/angles.h>
#include <base_local_planner/costmap_model.h>
#include <pluginlib/class_list_macros.h>
#include <tf2/utils.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

PLUGINLIB_EXPORT_CLASS(improved_dwa_local_planner::ImprovedDWALocalPlanner,
                       nav_core::BaseLocalPlanner)

namespace improved_dwa_local_planner {

// =============================================================================
// МЕТОД: Расчёт коэффициента безопасности
// =============================================================================
double ImprovedDWALocalPlanner::getSafetyMultiplier(
    const tracked_obstacle_msgs::TrackedCircle &obs) {
  if (obs.is_ghost && obs.is_unstable)
    return safety_mult_ghost_unstable_;
  if (obs.is_ghost)
    return safety_mult_ghost_;
  if (obs.is_unstable)
    return safety_mult_unstable_;
  return 1.0;
}

// =============================================================================
// Конструктор / Деструктор
// =============================================================================
ImprovedDWALocalPlanner::ImprovedDWALocalPlanner() : initialized_(false) {}
ImprovedDWALocalPlanner::~ImprovedDWALocalPlanner() {}

// =============================================================================
// CALLBACK: Получение обогащённых данных об объектах от ScanCleaner
// =============================================================================
void ImprovedDWALocalPlanner::trackedObstaclesCallback(
    const tracked_obstacle_msgs::TrackedCircleArray::ConstPtr &msg) {
  last_tracked_obstacles_ = *msg;

  // Режим "Радара": если нет активного плана — всё равно считаем коллизии для
  // отладки
  if (global_plan_.empty() && initialized_) {
    std::vector<Trajectory> debug_trajs;
    double step_x = max_vel_x_ / 4.0;
    double step_th = (2.0 * max_vel_th_) / 10.0;

    for (double vx = 0.0; vx <= max_vel_x_ + 1e-3; vx += step_x) {
      for (double vth = -max_vel_th_; vth <= max_vel_th_ + 1e-3;
           vth += step_th) {
        Trajectory traj = generateTrajectory(vx, vth);
        traj.cost = 1.0;

        for (const auto &tracked_obs : last_tracked_obstacles_.circles) {
          double v_obs =
              std::hypot(tracked_obs.velocity.x, tracked_obs.velocity.y);
          if (v_obs > speed_threshold_ || tracked_obs.is_ghost) {
            double multiplier = getSafetyMultiplier(tracked_obs);
            if (calculate_dis_fp(traj, tracked_obs, multiplier) < (hard_collision_buffer_ * 0.75)) {
              traj.cost = -1.0;
            }
          }
        }
        debug_trajs.push_back(traj);
      }
    }
    publishMarkers(debug_trajs);
    publishTrackedObstaclesViz();
  }
}

// =============================================================================
// Инициализация плагина
// =============================================================================
void ImprovedDWALocalPlanner::initialize(
    std::string name, tf2_ros::Buffer *tf,
    costmap_2d::Costmap2DROS *costmap_ros) {
  if (!initialized_) {
    ros::NodeHandle private_nh("~/" + name);
    ros::NodeHandle nh;

    tf_ = tf;
    costmap_ros_ = costmap_ros;

    tracked_obstacles_sub_ =
        nh.subscribe("/obstacles_tracked", 1,
                     &ImprovedDWALocalPlanner::trackedObstaclesCallback, this);

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

    private_nh.param("alpha", alpha_, 0.1);
    private_nh.param("beta", beta_, 10.0);
    private_nh.param("gamma", gamma_, 0.14);
    private_nh.param("kappa", kappa_, 2.0);
    private_nh.param("epsilon", epsilon_, 5.0);

    private_nh.param("max_vel_x", max_vel_x_, 0.8);
    private_nh.param("min_vel_x", min_vel_x_, 0.0);
    private_nh.param("max_vel_th", max_vel_th_, 1.5);
    private_nh.param("acc_lim_x", acc_lim_x_, 1.5);
    private_nh.param("acc_lim_th", acc_lim_th_, 2.5);

    private_nh.param("predict_time", predict_time_, 3.0);
    private_nh.param("dt", dt_, 0.1);
    private_nh.param("vx_samples", vx_samples_, 20);
    private_nh.param("vth_samples", vth_samples_, 30);

    private_nh.param("xy_goal_tolerance", xy_goal_tolerance_, 0.25);
    private_nh.param("yaw_goal_tolerance", yaw_goal_tolerance_, 0.3);
    private_nh.param("speed_threshold", speed_threshold_, 0.1);

    // Новые параметры дистанции и безопасности
    private_nh.param("robot_radius", robot_radius_, 0.3);
    private_nh.param("safe_clearance_dist", safe_clearance_dist_, 0.3);
    private_nh.param("hard_collision_buffer", hard_collision_buffer_, 0.2);
    private_nh.param("radar_range", radar_range_, 4.0);
    private_nh.param("safety_multiplier_unstable", safety_mult_unstable_, 1.3);
    private_nh.param("safety_multiplier_ghost", safety_mult_ghost_, 1.5);
    private_nh.param("safety_multiplier_ghost_unstable",
                     safety_mult_ghost_unstable_, 1.8);

    odom_helper_.setOdomTopic("odom");
    initialized_ = true;
    // ROS_INFO("[ImprovedDWA] Initialized. Subscribed to /obstacles_tracked "
    //          "(ghost/unstable-aware).");
  }
}

// =============================================================================
// Установка глобального плана
// =============================================================================
bool ImprovedDWALocalPlanner::setPlan(
    const std::vector<geometry_msgs::PoseStamped> &plan) {
  if (!initialized_)
    return false;
  global_plan_ = plan;
  return true;
}

// =============================================================================
// Проверка достижения цели
// =============================================================================
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

// =============================================================================
// ГЛАВНЫЙ ЦИКЛ: Вычисление команды скоростей
// =============================================================================
bool ImprovedDWALocalPlanner::computeVelocityCommands(
    geometry_msgs::Twist &cmd_vel) {
  if (!initialized_)
    return false;

  // Чистка старой памяти (удаляем объекты, которые мы "объехали" более 3-4 секунд назад)
  for (auto it = passed_obstacles_.begin(); it != passed_obstacles_.end();) {
    if ((ros::Time::now() - it->second).toSec() > 4.0) {
      it = passed_obstacles_.erase(it);
    } else {
      ++it;
    }
  }

  geometry_msgs::PoseStamped robot_pose;
  if (!costmap_ros_->getRobotPose(robot_pose))
    return false;

  std::vector<geometry_msgs::PoseStamped> local_plan;
  if (!base_local_planner::transformGlobalPlan(
          *tf_, global_plan_, robot_pose, *costmap_ros_->getCostmap(),
          costmap_ros_->getGlobalFrameID(), local_plan))
    return false;

  nav_msgs::Odometry base_odom;
  odom_helper_.getOdom(base_odom);
  geometry_msgs::Twist current_vel = base_odom.twist.twist;
  double robot_yaw = tf2::getYaw(robot_pose.pose.orientation);

  // Логика обнаружения "обойденных" объектов (state tracking)
  for (const auto &obs : last_tracked_obstacles_.circles) {
      double dx = obs.center.x - robot_pose.pose.position.x;
      double dy = obs.center.y - robot_pose.pose.position.y;
      double dist = std::hypot(dx, dy);

      // Скалярное произведение вектора на объект с вектором взгляда робота
      double look_dot = dx * std::cos(robot_yaw) + dy * std::sin(robot_yaw);

      // Если объект ближе 1.5 метров, и он физически оказался СЗАДИ робота (look_dot < 0)
      if (dist < 1.5 && look_dot < 0.0) {
          // Записываем его в память, чтобы отключить предиктивные коридоры на 4 секунды
          passed_obstacles_[obs.memory_id] = ros::Time::now();
      }
  }

  const double sim_accel_time = 0.5;
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

  double dvx = (vx_samples_ > 1) ? (max_vx - min_vx) / (vx_samples_ - 1) : 0.01;
  double dvth =
      (vth_samples_ > 1) ? (max_vth - min_vth) / (vth_samples_ - 1) : 0.01;

  for (double vx = min_vx; vx <= max_vx + 1e-6; vx += dvx) {
    for (double vth = min_vth; vth <= max_vth + 1e-6; vth += dvth) {
      Trajectory traj = generateTrajectory(vx, vth);
      if (traj.poses.empty())
        continue;

      traj.cost = scoreTrajectory(traj, local_plan);
      all_trajectories.push_back(traj);

      if (traj.cost >= 0.0 && traj.cost > best_traj.cost)
        best_traj = traj;
    }
  }

  publishMarkers(all_trajectories);

  if (best_traj.cost < 0) {
    cmd_vel.linear.x = 0;
    cmd_vel.angular.z = 0;
    // ROS_WARN_THROTTLE(1.0, "[ImprovedDWA] No safe trajectory found. STOP.");
    return false;
  }

  cmd_vel.linear.x = best_traj.vx;
  cmd_vel.angular.z = best_traj.vth;

  nav_msgs::Path local_path;
  local_path.header.stamp = ros::Time::now();
  local_path.header.frame_id = costmap_ros_->getGlobalFrameID();
  local_path.poses = best_traj.poses;
  local_plan_pub_.publish(local_path);

  static ros::Time last_log = ros::Time(0);
  if ((ros::Time::now() - last_log).toSec() > 1.0) {
    last_log = ros::Time::now();
    int dyn = 0, ghost = 0, unstable = 0;
    for (const auto &o : last_tracked_obstacles_.circles) {
      if (std::hypot(o.velocity.x, o.velocity.y) > speed_threshold_)
        ++dyn;
      if (o.is_ghost)
        ++ghost;
      if (o.is_unstable)
        ++unstable;
    }
    // ROS_INFO("[DWA] V=%.2f W=%.2f | Dyn=%d Ghost=%d Unstable=%d", best_traj.vx,
    //          best_traj.vth, dyn, ghost, unstable);
    ROS_INFO("[DWA] Alpha: %.2f | Speed Bonus: %.2f | Turning Bonus: %.2f",
             best_traj.debug_alpha, best_traj.debug_speed_bonus, best_traj.debug_turning_bonus);
  }

  publishTrackedObstaclesViz();
  return true;
}

// =============================================================================
// SCORING: Оценка траектории
// =============================================================================
double ImprovedDWALocalPlanner::scoreTrajectory(
    Trajectory &traj,
    const std::vector<geometry_msgs::PoseStamped> &local_plan) {

  // 1. Статические препятствия (Costmap)
  base_local_planner::CostmapModel world_model(*(costmap_ros_->getCostmap()));
  double max_footprint_cost = 0.0;
  for (size_t i = 0; i < traj.poses.size(); ++i) {
    const auto &pose = traj.poses[i];
    double pt_cost = world_model.footprintCost(
        pose.pose.position.x, pose.pose.position.y,
        tf2::getYaw(pose.pose.orientation), costmap_ros_->getRobotFootprint());

    if (pt_cost < 0 || pt_cost >= 253) {
      traj.collision_pose_idx = (int)i;
      return -1.0;
    }
    if (pt_cost > max_footprint_cost)
      max_footprint_cost = pt_cost;
  }

  geometry_msgs::PoseStamped robot_pose;
  costmap_ros_->getRobotPose(robot_pose);

  // 2. Близость к глобальному плану (Alpha)
  int closest_idx = 0;
  double min_dist_sq = -1.0;
  for (int i = 0; i < (int)local_plan.size(); ++i) {
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

  // 4. Скорость (Gamma)
  double robot_yaw = tf2::getYaw(robot_pose.pose.orientation);
  double dist_to_goal = std::hypot(
      robot_pose.pose.position.x - global_plan_.back().pose.position.x,
      robot_pose.pose.position.y - global_plan_.back().pose.position.y);
  double angle_to_goal =
      atan2(global_plan_.back().pose.position.y - robot_pose.pose.position.y,
            global_plan_.back().pose.position.x - robot_pose.pose.position.x);

  double velocity_score = 0.0;
  if (dist_to_goal > 1.0) {
    // В обычном режиме крейсерская скорость 0.5 м/с
    double cruise_vel = 0.5;
    // Идеальная оценка (1.0) дается, если скорость точно равна cruise_vel. 
    // Разгон свыше 0.5 штрафуется базовой метрикой (чтобы робот не спешил), 
    // НО этот штраф легко перебивается бонусом за скорость (speed_bonus) при уклонении.
    velocity_score = 1.0 - (std::abs(traj.vx - cruise_vel) / max_vel_x_);
  } else {
    double desired_vel = max_vel_x_ * (dist_to_goal / 1.0);
    velocity_score = 1.0 - (std::abs(traj.vx - desired_vel) / max_vel_x_);
  }

  // 5. ДИНАМИЧЕСКИЙ АНАЛИЗ
  double min_safety_score = 1.0;
  double critical_heading_score = 0.0;
  const tracked_obstacle_msgs::TrackedCircle *critical_obs = nullptr;

  int total_obs = (int)last_tracked_obstacles_.circles.size();
  int dynamic_cnt = 0, ghost_cnt = 0, unstable_cnt = 0;

  for (const auto &tracked_obs : last_tracked_obstacles_.circles) {
    double v_obs = std::hypot(tracked_obs.velocity.x, tracked_obs.velocity.y);
    bool is_passed = (passed_obstacles_.find(tracked_obs.memory_id) != passed_obstacles_.end());

    if (v_obs > speed_threshold_)
      ++dynamic_cnt;
    if (tracked_obs.is_ghost)
      ++ghost_cnt;
    if (tracked_obs.is_unstable)
      ++unstable_cnt;

    // Считаем активной динамикой тех, кто быстрее 0.2, не обойден
    bool is_active_dynamic = (v_obs > 0.2) && !is_passed;
    bool is_relevant = is_active_dynamic || tracked_obs.is_ghost;

    if (!is_relevant) {
        // Мы все еще должны не врезаться в "обойденные" объекты, но проверим их только как статику.
        // calculate_dis_fp внутри отключит коридоры для is_passed.
        if (is_passed || v_obs <= 0.2) {
             double multiplier = getSafetyMultiplier(tracked_obs);
             double safety = calculate_dis_fp(traj, tracked_obs, multiplier);
             if (safety < min_safety_score) min_safety_score = safety;
        }
        continue;
    }

    double multiplier = getSafetyMultiplier(tracked_obs);
    double safety = calculate_dis_fp(traj, tracked_obs, multiplier);

    if (safety < min_safety_score) {
      min_safety_score = safety;
      critical_obs = &tracked_obs;
    }
  }

  // ROS_INFO_THROTTLE(
  //     0.5, "[DWA] Obstacles: total=%d | dyn=%d | ghost=%d | unstable=%d",
  //     total_obs, dynamic_cnt, ghost_cnt, unstable_cnt);

  // 6. Адаптивная логика поведения
  double current_alpha = alpha_;
  double turning_bonus = 0.0;
  double speed_bonus = 0.0;

  if (critical_obs != nullptr) {
    double angle_to_obs =
        atan2(critical_obs->center.y - robot_pose.pose.position.y,
              critical_obs->center.x - robot_pose.pose.position.x);
    double diff_to_obs =
        std::abs(angles::shortest_angular_distance(robot_yaw, angle_to_obs));
    double in_front_factor = std::max(0.0, 1.0 - (diff_to_obs / 1.0));

    double angle_to_robot =
        atan2(robot_pose.pose.position.y - critical_obs->center.y,
              robot_pose.pose.position.x - critical_obs->center.x);
    double obs_vel_angle =
        atan2(critical_obs->velocity.y, critical_obs->velocity.x);

    double danger_scale = getSafetyMultiplier(*critical_obs);
    double aggressive_danger =
        in_front_factor * std::min(danger_scale, 1.0);

    // ПРИЧИНА "СЛЕДОВАНИЯ ЗА ОБЪЕКТОМ" ИСПРАВЛЕНА:
    // Функция calculate_dis_hv заставляет робота отворачиваться (смотреть в другую сторону) от объекта. 
    // Если объект пересекает наш путь и уходит направо, робот подсознательно пытался отвернуться налево.
    // Из-за этого он буквально заруливал прямо в пустой след движущегося человека!
    // Теперь мы умножаем это "отторжение" на aggressive_danger:
    // как только объект прошел нас (или не опасен), магнит отключается, и робот не крутит за ним рулем!
    critical_heading_score = calculate_dis_hv(traj, robot_pose, *critical_obs) * aggressive_danger;

    double evade_side = 1.0;
    if (aggressive_danger > 0.5) {
      // ПРЕДСКАЗУЕМОЕ УКЛОНЕНИЕ:
      // Вместо текущего положения объекта, смотрим куда он приедет через ~1 сек
      // Это позволяет уклоняться от вектора скорости, а не только от центра тела
      double t_pred = 1.0;
      double future_obs_x = critical_obs->center.x + critical_obs->velocity.x * t_pred;
      double future_obs_y = critical_obs->center.y + critical_obs->velocity.y * t_pred;
      double angle_to_future_obs = atan2(future_obs_y - robot_pose.pose.position.y,
                                         future_obs_x - robot_pose.pose.position.x);

      double diff_future = angles::shortest_angular_distance(angle_to_obs, angle_to_future_obs);
      double head_on_diff = std::abs(angles::shortest_angular_distance(obs_vel_angle, angle_to_robot));

      // Явное правило при лобовом столкновении (в пределах ~10 градусов)
      if (head_on_diff < 0.17) {
        evade_side = -1.0; 
      } else {
        evade_side = (diff_future > 0) ? -1.0 : 1.0;
      }

      current_alpha = alpha_ * 0.8; 

      // СЕКРЕТ S-МАНЕВРА №2: Умное ожидание! 
      // Вместо жесткого угла в 0.7, используем более точную модель для Head-on (около 135-180 градусов, т.е. > 2.3 радиан ~ 130+ град)
      // head_on_diff меряет угол между вектором скорости объекта и вектором на робота.
      // Если объект летит НА робота, head_on_diff близок к 0. Встречное движение.
      bool is_head_on = (head_on_diff < 0.8); // ~45 градусов отклонения от идеального лобового

      if (is_head_on) {
          if (traj.vth * evade_side > 0 && traj.vx > 0.1) {
              turning_bonus += 0.5; 
              speed_bonus += 1.5 * traj.vx;              
          } else if (std::abs(traj.vx) < 0.1) {
              if (min_safety_score < 0.5) {
                  speed_bonus -= 10.0;                       
              }
          } else {
              turning_bonus -= 2.0;                      
          }
      } else {
          // ПОПЕРЕЧНОЕ ДВИЖЕНИЕ ИЛИ РАЗЪЕЗД СБОКУ ПОСЛЕ УВОРОТА
          // Важное разделение: 
          // 1. Опасность < 0.35 (это 0.0 - тело, 0.1 - красный коридор, 0.2 - hard limit). Это реальная угроза перед бампером!
          // 2. Опасность 0.4-1.0 (это soft limit). Мы просто проезжаем близко сбоку. Тормозить не нужно!
          if (min_safety_score < 0.35) {
              double dist_to_obs = std::hypot(critical_obs->center.x - robot_pose.pose.position.x, 
                                              critical_obs->center.y - robot_pose.pose.position.y);
              
              nav_msgs::Odometry base_odom;
              odom_helper_.getOdom(base_odom);
              double current_vx = base_odom.twist.twist.linear.x;

              // ТОЧКА НЕВОЗВРАТА: экстренный побег с линии атаки
              if (current_vx > 0.15 && dist_to_obs < 1.5) {
                  speed_bonus += 2.0 * traj.vx;  // Педаль в пол!
                  turning_bonus -= 1.0 * std::abs(traj.vth);
              } else {
                  speed_bonus -= 2.0 * traj.vx; // ТОРМОЗИМ!
                  // ЖЕСТКИЙ ЗАПРЕТ на виляние рулем в хвост объекта ради скорости!
                  // Штраф в 10 раз мощнее, чтобы он не искал математические лазейки для объезда.
                  turning_bonus -= 10.0 * std::abs(traj.vth);
              }
          } else {
              // МЫ ВНЕ КРАСНОГО КОРИДОРА (Объект просто где-то сбоку или сзади)
              // Робот сейчас в безопасности. Нужно немедленно вернуться на маршрут.
              // Раньше тут стоял жесткий штраф на поворот руля (turning_bonus -= 3.0),
              // из-за чего робот не мог повернуть обратно на маршрут и ехал прямо,
              // "следуя" за объектом. Теперь мы позволяем ему поворачивать.
              
              current_alpha = alpha_ * 5.0; // В 5 раз сильнее тянем к пути!
              turning_bonus = 0.0;
          }
      }
    } else {
      // КРАСНОГО КОРИДОРА НЕТ И ОБЪЕКТ УЖЕ НЕ ПЕРЕД НАМИ (aggressive_danger <= 0.5)
      // В этом случае единственный приоритет - вернуть нос на цель!
      double angle_diff = angles::shortest_angular_distance(robot_yaw, angle_to_goal);
      
      if ((traj.vth * angle_diff) > 0) {
        turning_bonus += 1.0 * std::abs(traj.vth); // Награждаем поворот к цели
      } else {
        turning_bonus -= 2.0 * std::abs(traj.vth); // ЖЕСТКО штрафуем поворот от цели
      }
      
      speed_bonus += 0.5;
      current_alpha = alpha_ * 5.0; // Жестко привязываем к зеленой линии
    }

    // ROS_INFO_THROTTLE(
    //     0.2,
    //     "[DWA] CritObs: ghost=%d unstable=%d | aggr=%.2f | multiplier=%.1f",
    //     (int)critical_obs->is_ghost, (int)critical_obs->is_unstable,
    //     aggressive_danger, getSafetyMultiplier(*critical_obs));
  }

  if (min_safety_score < 0.05) {
    return -100.0 + min_safety_score;
  }

  traj.debug_alpha = current_alpha;
  traj.debug_speed_bonus = speed_bonus;
  traj.debug_turning_bonus = turning_bonus;

  // 7. Финальное суммирование
  return (current_alpha * (path_dist_score + heading_score)) +
         (beta_ * dist_score) + (gamma_ * velocity_score) +
         (kappa_ * critical_heading_score) + (epsilon_ * min_safety_score) +
         turning_bonus + speed_bonus;
}

// =============================================================================
// HELPER: Оценка направления движения
// =============================================================================
double ImprovedDWALocalPlanner::calculate_dis_hv(
    const Trajectory &traj, const geometry_msgs::PoseStamped &robot_pose,
    const tracked_obstacle_msgs::TrackedCircle &obs) {

  double obs_speed = std::hypot(obs.velocity.x, obs.velocity.y);
  // Определяем истинный угол НА объект от робота, а не вектор скорости объекта
  double angle_to_obs = atan2(obs.center.y - robot_pose.pose.position.y,
                              obs.center.x - robot_pose.pose.position.x);

  double d = std::hypot(obs.center.x - robot_pose.pose.position.x,
                        obs.center.y - robot_pose.pose.position.y);

  double eta = std::max(0.0, radar_range_ - d);
  
  // Оцениваем финальный разворот робота (на конце маневра)
  double final_yaw = tf2::getYaw(robot_pose.pose.orientation);
  if (!traj.poses.empty()) {
    final_yaw = tf2::getYaw(traj.poses.back().pose.orientation);
  }
  
  // Штрафуем траектории, которые "смотрят" на объект 
  // (cos = 1 -> factor = 0, cos = -1 -> factor = 1)
  // Теперь робот будет награжден баллами за то, что отворачивается от объекта
  double theta = angles::shortest_angular_distance(final_yaw, angle_to_obs);
  double heading_factor = 0.5 * (1.0 - std::cos(theta)); 
  
  double rel_speed_proxy = std::abs(traj.vx) + obs_speed;
  double ghost_discount = obs.is_ghost ? 0.6 : 1.0;

  return eta * rel_speed_proxy * heading_factor * ghost_discount;
}

// =============================================================================
// HELPER: Предсказание столкновения
// =============================================================================
double ImprovedDWALocalPlanner::calculate_dis_fp(
    Trajectory &traj, const tracked_obstacle_msgs::TrackedCircle &obs,
    double safety_multiplier) {

  double min_dist_sq = 1e6;
  int collision_idx = -1;
  double collision_time = -1.0;

  double core_limit = obs.radius + robot_radius_; // Только физические габариты (базовый радиус)
  double obs_v_sq = obs.velocity.x * obs.velocity.x + obs.velocity.y * obs.velocity.y;
  double v_obs = std::sqrt(obs_v_sq);

  bool core_hit = false;
  bool corridor_hit = false;

  // Игнорируем векторы статики / ложной динамики (медленнее 0.2 м/с)
  bool is_dynamic = (v_obs > 0.1);

  // Игнорируем объекты, которые мы уже благополучно обошли
  bool is_passed = (passed_obstacles_.find(obs.memory_id) != passed_obstacles_.end());

  for (size_t i = 0; i < traj.poses.size(); ++i) {
    double t = i * dt_;

    // Если объект "обойден" или статический, он для планировщика никуда не летит
    double pred_obs_x = obs.center.x + (is_dynamic && !is_passed ? obs.velocity.x * t : 0.0);
    double pred_obs_y = obs.center.y + (is_dynamic && !is_passed ? obs.velocity.y * t : 0.0);
    double dx = traj.poses[i].pose.position.x - pred_obs_x;
    double dy = traj.poses[i].pose.position.y - pred_obs_y;
    double d2 = dx * dx + dy * dy;

    if (d2 < min_dist_sq) {
      min_dist_sq = d2;
      collision_idx = i;
      collision_time = t;
    }

    if (d2 <= core_limit * core_limit) {
        core_hit = true;
    }

    if (is_dynamic && !is_passed) {
        double rx = traj.poses[i].pose.position.x - obs.center.x;
        double ry = traj.poses[i].pose.position.y - obs.center.y;
        double dot = rx * obs.velocity.x + ry * obs.velocity.y;
        
        // Коридор существует ТОЛЬКО спереди объекта (вектор скорости).
        // Если точка траектории находится сзади объекта (dot < 0), мы свободно проезжаем.
        if (dot > 0) {
            double proj_len = dot / v_obs;
            // Проецируем коридор на 5.5 секунд вперед 
            if (proj_len < v_obs * 5.5) {
                // "Конус неопределенности": чем дальше впереди объекта, тем шире математический коридор.
                // Это защищает от внезапных изменений вектора. На каждый метр длины коридор расширяется на 25 см.
                double dynamic_core_limit = core_limit + (proj_len * 0.01);
                
                double dist_to_line_sq = (rx * rx + ry * ry) - (dot * dot / obs_v_sq);
                if (dist_to_line_sq < dynamic_core_limit * dynamic_core_limit) {
                    corridor_hit = true;
                }
            }
        }
    }
  }

  double start_d2 = 1e6;
  if (!traj.poses.empty()) {
      double start_dx = traj.poses[0].pose.position.x - obs.center.x;
      double start_dy = traj.poses[0].pose.position.y - obs.center.y;
      start_d2 = start_dx * start_dx + start_dy * start_dy;
  }

  if (core_hit) {
      if (start_d2 <= core_limit * core_limit) {
          // Если уже в объекте - выбираем траекторию, которая увеличит расстояние
          if (collision_idx == 0 || min_dist_sq > start_d2 - 1e-4) {
              return 0.1; // Выживаем
          }
      }
      traj.collision_pose_idx = collision_idx;
      traj.obstacle_pos.x = obs.center.x + obs.velocity.x * collision_time;
      traj.obstacle_pos.y = obs.center.y + obs.velocity.y * collision_time;
      traj.obstacle_start_pos = obs.center;
      return 0.0; // Жесткий сброс = Остановка
  }

  if (corridor_hit) {
      // ИСКЛЮЧЕНИЕ ДЛЯ ОЖИДАНИЯ: Если мы стоим (или почти стоим), 
      // и объект физически нас не заденет (он проедет МИМО впереди),
      // то игнорируем математический коридор! Это позволяет пропускать горизонтальный трафик.
      double hard_limit = (obs.radius + robot_radius_ + hard_collision_buffer_) * safety_multiplier;
      if (std::abs(traj.vx) < 0.05 && min_dist_sq > hard_limit * hard_limit) {
          // Стоять безопасно, коридор не учитываем.
      } else {
          // Оставляем 0.3 чтобы она выжила (прошла проверку < 0.05),
          // но это хуже, чем остановиться (1.0).
          return 0.3;
      }
  }

  // ВАЖНО: hard_limit обязательно должен учитывать габариты самого робота (robot_radius_)! 
  // Иначе робот будет генерировать траектории сквозь край объекта.
  double hard_limit = (obs.radius + robot_radius_ + hard_collision_buffer_) * safety_multiplier;
  
  if (min_dist_sq <= hard_limit * hard_limit) {
    // RESCUE MODE: Проверяем, не были ли мы уже изначально внутри hard_limit при t=0
    double start_d2 = 1e6;
    if (!traj.poses.empty()) {
        double start_dx = traj.poses[0].pose.position.x - obs.center.x;
        double start_dy = traj.poses[0].pose.position.y - obs.center.y;
        start_d2 = start_dx * start_dx + start_dy * start_dy;
    }

    if (start_d2 <= hard_limit * hard_limit) {
        // Мы УЖЕ заблокированы. Даем шанс траектории, если она явно увеличивает дистанцию.
        if (collision_idx == 0 || min_dist_sq > start_d2 - 1e-4) {
            return 0.2; // Траектория спасения
        }
    }

    // Если это неизбежное столкновение внутри предикшена (даже через 1.5-2.0 сек),
    // жестко отбраковываем траекторию (возвращаем 0.0), чтобы робот не рисковал.
    traj.collision_pose_idx = collision_idx;
    traj.obstacle_pos.x = obs.center.x + obs.velocity.x * collision_time;
    traj.obstacle_pos.y = obs.center.y + obs.velocity.y * collision_time;
    traj.obstacle_pos.z = 0;
    traj.obstacle_start_pos = obs.center;
    
    return 0.2;
  }

  // ГРАДИЕНТ ОТТАЛКИВАНИЯ (Soft Limit)
  // Заставляет робота делать S-маневр, обходя объект заранее ради лучшего балла.
  double min_dist = std::sqrt(min_dist_sq);
  double soft_limit = (obs.radius + robot_radius_ + safe_clearance_dist_) * safety_multiplier;
  
  if (min_dist < soft_limit) {
    // Плавно занижаем оценку при приближении от soft_limit к hard_limit
    double score = (min_dist - hard_limit) / (soft_limit - hard_limit);
    if (score < 0) score = 0;
    return 0.1 + 0.9 * std::pow(score, 2);
  }

  return 1.0;
}

// =============================================================================
// GENERATION: Генерация траектории
// =============================================================================
ImprovedDWALocalPlanner::Trajectory
ImprovedDWALocalPlanner::generateTrajectory(double vx, double vth) {
  Trajectory traj;
  traj.vx = vx;
  traj.vth = vth;

  geometry_msgs::PoseStamped cp;
  costmap_ros_->getRobotPose(cp);

  double x = cp.pose.position.x;
  double y = cp.pose.position.y;
  double th = tf2::getYaw(cp.pose.orientation);

  for (double t = 0; t <= predict_time_; t += dt_) {
    x += vx * std::cos(th) * dt_;
    y += vx * std::sin(th) * dt_;
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

// =============================================================================
// VIZ: Веер траекторий и маркеры коллизий
// =============================================================================
void ImprovedDWALocalPlanner::publishMarkers(
    const std::vector<Trajectory> &trajectories) {

  visualization_msgs::MarkerArray markers;
  visualization_msgs::Marker clear;
  clear.ns = "dwa";
  clear.action = 3;
  markers.markers.push_back(clear);

  for (size_t i = 0; i < trajectories.size(); ++i) {
    visualization_msgs::Marker m;
    m.header.frame_id = costmap_ros_->getGlobalFrameID();
    m.header.stamp = ros::Time::now();
    m.ns = "dwa";
    m.id = (int)i;
    m.type = visualization_msgs::Marker::LINE_STRIP;
    m.scale.x = 0.01;
    m.color.a = 0.2;
    if (trajectories[i].cost < 0)
      m.color.r = 1.0;
    else
      m.color.g = 1.0;
    for (const auto &p : trajectories[i].poses)
      m.points.push_back(p.pose.position);
    markers.markers.push_back(m);
  }
  candidate_trajs_pub_.publish(markers);

  visualization_msgs::MarkerArray collision_markers;
  visualization_msgs::Marker clear_col;
  clear_col.ns = "collisions";
  clear_col.action = 3;
  collision_markers.markers.push_back(clear_col);

  int marker_id = 0;
  for (const auto &traj : trajectories) {
    if (traj.cost >= 0 || traj.collision_pose_idx < 0 ||
        traj.collision_pose_idx >= (int)traj.poses.size())
      continue;

    visualization_msgs::Marker m_robot;
    m_robot.header.frame_id = costmap_ros_->getGlobalFrameID();
    m_robot.header.stamp = ros::Time::now();
    m_robot.ns = "collisions";
    m_robot.id = marker_id++;
    m_robot.type = visualization_msgs::Marker::SPHERE;
    m_robot.pose = traj.poses[traj.collision_pose_idx].pose;
    m_robot.scale.x = m_robot.scale.y = m_robot.scale.z = 0.3;
    m_robot.color.r = 1.0;
    m_robot.color.a = 0.8;
    collision_markers.markers.push_back(m_robot);

    visualization_msgs::Marker m_obs;
    m_obs.header = m_robot.header;
    m_obs.ns = "collisions";
    m_obs.id = marker_id++;
    m_obs.type = visualization_msgs::Marker::SPHERE;
    m_obs.pose.position = traj.obstacle_pos;
    m_obs.pose.orientation.w = 1.0;
    m_obs.scale.x = m_obs.scale.y = m_obs.scale.z = 0.4;
    m_obs.color.r = 1.0;
    m_obs.color.g = 0.5;
    m_obs.color.a = 0.6;
    collision_markers.markers.push_back(m_obs);

    visualization_msgs::Marker m_line;
    m_line.header = m_robot.header;
    m_line.ns = "collision_vectors";
    m_line.id = marker_id++;
    m_line.type = visualization_msgs::Marker::LINE_LIST;
    m_line.scale.x = 0.02;
    m_line.color.r = 1.0;
    m_line.color.g = 1.0;
    m_line.color.a = 0.5;
    m_line.points.push_back(traj.obstacle_start_pos);
    m_line.points.push_back(traj.obstacle_pos);
    collision_markers.markers.push_back(m_line);
  }
  collision_markers_pub_.publish(collision_markers);
}

// =============================================================================
// VIZ: Отображение отслеживаемых объектов
// =============================================================================
void ImprovedDWALocalPlanner::publishTrackedObstaclesViz() {
  visualization_msgs::MarkerArray markers;
  visualization_msgs::Marker clear;
  clear.action = 3;
  clear.ns = "tracked_obstacles";
  markers.markers.push_back(clear);

  for (const auto &obs : last_tracked_obstacles_.circles) {
    double v = std::hypot(obs.velocity.x, obs.velocity.y);
    bool is_relevant =
        (v > 0.2) || obs.is_ghost || obs.is_unstable; // Увеличили порог до 0.2 м/с, чтобы отсечь углы стен

    bool is_passed = (passed_obstacles_.find(obs.memory_id) != passed_obstacles_.end());

    if (!is_relevant)
      continue;

    float r_col = 1.0f, g_col = 0.0f, b_col = 0.0f, a_col = 0.8f;
    std::string state_label;
    if (is_passed) {
      r_col = 0.2f;
      g_col = 0.8f;
      b_col = 0.2f;
      a_col = 0.4f;
      state_label = "[PASSED]";
    } else if (obs.is_ghost && obs.is_unstable) {
      r_col = 0.8f;
      g_col = 0.2f;
      b_col = 0.8f;
      a_col = 0.5f;
      state_label = "[GHOST+UNSTABLE]";
    } else if (obs.is_ghost) {
      r_col = 0.5f;
      g_col = 0.5f;
      b_col = 0.5f;
      a_col = 0.4f;
      state_label = "[GHOST " +
                    std::to_string((int)(obs.time_since_seen * 10) / 10.0) +
                    "s]";
    } else if (obs.is_unstable) {
      r_col = 1.0f;
      g_col = 0.4f;
      b_col = 0.0f;
      a_col = 0.9f;
      state_label = "[UNSTABLE]";
    } else {
      state_label = "[ACTIVE]";
    }

    visualization_msgs::Marker text;
    text.header.frame_id = last_tracked_obstacles_.header.frame_id;
    text.header.stamp = ros::Time::now();
    text.ns = "tracked_obstacles";
    text.id = obs.memory_id;
    text.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
    text.pose.position = obs.center;
    text.pose.position.z += 1.0;
    text.scale.z = 0.35;
    text.color.r = r_col;
    text.color.g = g_col;
    text.color.b = b_col;
    text.color.a = 1.0f;
    text.text = state_label + " MemID:" + std::to_string(obs.memory_id);
    markers.markers.push_back(text);

    visualization_msgs::Marker body;
    body.header = text.header;
    body.ns = "tracked_obstacles_body";
    body.id = obs.memory_id;
    body.type = visualization_msgs::Marker::CYLINDER;
    body.pose.position = obs.center;
    body.pose.position.z = 0.1;
    body.scale.x = obs.radius * 2.0;
    body.scale.y = obs.radius * 2.0;
    body.scale.z = obs.is_ghost ? 0.1 : 0.3;
    body.color.r = r_col;
    body.color.g = g_col;
    body.color.b = b_col;
    body.color.a = a_col;
    markers.markers.push_back(body);

    double multiplier = getSafetyMultiplier(obs);
    double hard_limit = (obs.radius + hard_collision_buffer_) * multiplier;
    visualization_msgs::Marker hard_zone;
    hard_zone.header = text.header;
    hard_zone.ns = "tracked_obstacles_hard_limit";
    hard_zone.id = obs.memory_id;
    hard_zone.type = visualization_msgs::Marker::CYLINDER;
    hard_zone.pose.position = obs.center;
    hard_zone.pose.position.z = 0.05;
    hard_zone.scale.x = hard_limit * 2.0;
    hard_zone.scale.y = hard_limit * 2.0;
    hard_zone.scale.z = 0.05;
    hard_zone.color.r = r_col;
    hard_zone.color.g = g_col;
    hard_zone.color.b = b_col;
    hard_zone.color.a = 0.2f;
    markers.markers.push_back(hard_zone);

    double soft_limit = (obs.radius + robot_radius_ + safe_clearance_dist_) * multiplier;
    visualization_msgs::Marker soft_zone;
    soft_zone.header = text.header;
    soft_zone.ns = "tracked_obstacles_soft_limit";
    soft_zone.id = obs.memory_id;
    soft_zone.type = visualization_msgs::Marker::CYLINDER;
    soft_zone.pose.position = obs.center;
    soft_zone.pose.position.z = 0.02;
    soft_zone.scale.x = soft_limit * 2.0;
    soft_zone.scale.y = soft_limit * 2.0;
    soft_zone.scale.z = 0.03;
    soft_zone.color.r = 1.0f;
    soft_zone.color.g = 1.0f;
    soft_zone.color.b = 0.0f;
    soft_zone.color.a = 0.1f;
    markers.markers.push_back(soft_zone);

    if (v > 0.05 && !is_passed) {
      // ВИЗУАЛИЗАЦИЯ КОРИДОРА (зоны жесткой отбраковки вектора)
      double core_limit = obs.radius + robot_radius_;
      double yaw = atan2(obs.velocity.y, obs.velocity.x);

      visualization_msgs::Marker corridor;
      corridor.header = text.header;
      corridor.ns = "tracked_obstacles_corridor";
      corridor.id = obs.memory_id;
      corridor.type = visualization_msgs::Marker::CUBE;
      // Центр куба - на середине 3-секундного отрезка (1.5 секунды)
      corridor.pose.position.x = obs.center.x + obs.velocity.x * 1.5;
      corridor.pose.position.y = obs.center.y + obs.velocity.y * 1.5;
      corridor.pose.position.z = 0.01;
      
      tf2::Quaternion q;
      q.setRPY(0, 0, yaw);
      corridor.pose.orientation = tf2::toMsg(q);
      
      corridor.scale.x = v * 3.0; // длина (3 секунды * скорость)
      corridor.scale.y = core_limit * 2.0; // ширина (радиус объекта + радиус робота * 2)
      corridor.scale.z = 0.01; // плоский
      
      corridor.color.r = 1.0f;
      corridor.color.g = 0.0f;
      corridor.color.b = 0.0f;
      corridor.color.a = 0.15f; // Полупрозрачный красный
      markers.markers.push_back(corridor);

      visualization_msgs::Marker arrow;
      arrow.header = text.header;
      arrow.ns = "tracked_obstacles_vel";
      arrow.id = obs.memory_id;
      arrow.type = visualization_msgs::Marker::ARROW;
      arrow.scale.x = 0.05;
      arrow.scale.y = 0.1;
      arrow.scale.z = 0.1;
      arrow.color.r = r_col;
      arrow.color.g = g_col;
      arrow.color.b = b_col;
      arrow.color.a = 1.0f;
      geometry_msgs::Point p1 = obs.center;
      geometry_msgs::Point p2;
      p2.x = p1.x + obs.velocity.x;
      p2.y = p1.y + obs.velocity.y;
      p2.z = p1.z;
      arrow.points.push_back(p1);
      arrow.points.push_back(p2);
      markers.markers.push_back(arrow);
    }
  }
  tracked_objects_pub_.publish(markers);
}

} // namespace improved_dwa_local_planner