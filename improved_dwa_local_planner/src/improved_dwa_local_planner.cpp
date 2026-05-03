#include "improved_dwa_local_planner/improved_dwa_local_planner.h"
#include <angles/angles.h>
#include <base_local_planner/costmap_model.h>
#include <pluginlib/class_list_macros.h>
#include <tf2/utils.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

using Gap = improved_dwa_local_planner::ImprovedDWALocalPlanner::Gap;

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
// НОВЫЙ МЕТОД: Поиск промежутков между движущимися объектами
// =============================================================================

Gap ImprovedDWALocalPlanner::findGapBetweenObjects(
    const tracked_obstacle_msgs::TrackedCircle &obs1,
    const tracked_obstacle_msgs::TrackedCircle &obs2,
    const geometry_msgs::PoseStamped &robot_pose) {
  
  Gap gap;
  gap.is_viable = false;
  
  // Вычисляем текущее расстояние между центрами
  double dx = obs2.center.x - obs1.center.x;
  double dy = obs2.center.y - obs1.center.y;
  double dist_between = std::hypot(dx, dy);
  
  // Ширина промежутка (за вычетом радиусов)
  gap.gap_width = dist_between - obs1.radius - obs2.radius;
  
  // Центр промежутка
  gap.gap_center_x = (obs1.center.x + obs2.center.x) / 2.0;
  gap.gap_center_y = (obs1.center.y + obs2.center.y) / 2.0;
  
  // Проверяем: движутся ли объекты навстречу друг другу?
  double v1_mag = std::hypot(obs1.velocity.x, obs1.velocity.y);
  double v2_mag = std::hypot(obs2.velocity.x, obs2.velocity.y);
  
  if (v1_mag < 0.1 && v2_mag < 0.1) {
    // Оба статичны - промежуток постоянен
    gap.time_until_closed = 999.0;
    gap.is_viable = (gap.gap_width > robot_radius_ * 2.0 + 0.15);  // Запас 15см
    return gap;
  }
  
  // Относительная скорость сближения (проекция на линию между центрами)
  double rel_vx = obs2.velocity.x - obs1.velocity.x;
  double rel_vy = obs2.velocity.y - obs1.velocity.y;
  
  // Нормализованный вектор между объектами
  double nx = dx / dist_between;
  double ny = dy / dist_between;
  
  // Скорость сближения (скалярное произведение)
  double closing_speed = -(rel_vx * nx + rel_vy * ny);
  
  if (closing_speed > 0.05) {
    // Объекты сближаются
    gap.time_until_closed = gap.gap_width / closing_speed;
  } else if (closing_speed < -0.05) {
    // Объекты расходятся
    gap.time_until_closed = 999.0;
  } else {
    // Движутся параллельно
    gap.time_until_closed = 999.0;
  }
  
  // Промежуток пригоден, если:
  // 1. Ширина > диаметра робота + запас
  // 2. Времени до смыкания достаточно (или расходятся)
  gap.is_viable = (gap.gap_width > robot_radius_ * 2.0 + 0.15);
  
  return gap;
}

// =============================================================================
// Конструктор / Деструктор
// =============================================================================
ImprovedDWALocalPlanner::ImprovedDWALocalPlanner() : initialized_(false) {}
ImprovedDWALocalPlanner::~ImprovedDWALocalPlanner() {}

int ImprovedDWALocalPlanner::generateObstacleId() {
    return next_obstacle_id_++;
}

// =============================================================================
// CALLBACK: Получение обогащённых данных об объектах от ScanCleaner
// =============================================================================
void ImprovedDWALocalPlanner::trackedObstaclesCallback(
    const tracked_obstacle_msgs::TrackedCircleArray::ConstPtr &msg) {
  
  static std::map<int, ros::Time> consumed_ghosts;
  ros::Time now = ros::Time::now();
  for (auto it = consumed_ghosts.begin(); it != consumed_ghosts.end();) {
    if ((now - it->second).toSec() > 2.0) {
      it = consumed_ghosts.erase(it);
    } else {
      ++it;
    }
  }

  tracked_obstacle_msgs::TrackedCircleArray processed_obstacles;
  processed_obstacles.header = msg->header;
  
  std::vector<tracked_obstacle_msgs::TrackedCircle> modified_circles = msg->circles;
  std::vector<bool> kill_inactive(modified_circles.size(), false);

  for (size_t i = 0; i < modified_circles.size(); ++i) {
    if (modified_circles[i].is_ghost || modified_circles[i].is_unstable) {
      if (consumed_ghosts.find(modified_circles[i].memory_id) != consumed_ghosts.end()) {
        kill_inactive[i] = true;
      }
    }
  }

  for (size_t i = 0; i < modified_circles.size(); ++i) {
    if (!modified_circles[i].is_ghost && !modified_circles[i].is_unstable) {
      for (size_t j = 0; j < modified_circles.size(); ++j) {
        bool j_is_inactive = modified_circles[j].is_ghost || modified_circles[j].is_unstable;
        if (i == j || kill_inactive[j] || !j_is_inactive) continue;
        
        double dist = std::hypot(modified_circles[i].center.x - modified_circles[j].center.x, 
                                 modified_circles[i].center.y - modified_circles[j].center.y);
        
        if (dist <= 1.45) {
          double v_active = std::hypot(modified_circles[i].velocity.x, modified_circles[i].velocity.y);
          double v_inactive = std::hypot(modified_circles[j].velocity.x, modified_circles[j].velocity.y);
          
          bool similar_vector = false;
          if (v_active > 0.1 && v_inactive > 0.1) {
            double dot = modified_circles[i].velocity.x * modified_circles[j].velocity.x + 
                         modified_circles[i].velocity.y * modified_circles[j].velocity.y;
            double cos_angle = dot / (v_active * v_inactive);
            if (cos_angle > 0.3) { 
              similar_vector = true;
            }
          } else {
            similar_vector = true; 
          }

          if (similar_vector) {
            modified_circles[i].memory_id = modified_circles[j].memory_id;
            kill_inactive[j] = true;
            consumed_ghosts[modified_circles[j].memory_id] = now;
            break;
          }
        }
      }
    }
  }

  for (size_t i = 0; i < modified_circles.size(); ++i) {
    if (!kill_inactive[i]) {
      processed_obstacles.circles.push_back(modified_circles[i]);
    }
  }

  {
    std::lock_guard<std::mutex> lock(obstacles_mutex_);
    ros::Time now = ros::Time::now();
    for (const auto& c : processed_obstacles.circles) {
        bool found = false;
        if (c.memory_id != 0 && obstacle_memory_.find(c.memory_id) != obstacle_memory_.end()) {
            obstacle_memory_[c.memory_id].circle = c;
            obstacle_memory_[c.memory_id].last_seen = now;
            found = true;
        } else {
            for (auto& pair : obstacle_memory_) {
                double dx = pair.second.circle.center.x - c.center.x;
                double dy = pair.second.circle.center.y - c.center.y;
                if (std::hypot(dx, dy) < 0.4) {
                    pair.second.circle = c;
                    pair.second.circle.memory_id = pair.first;
                    pair.second.last_seen = now;
                    found = true;
                    break;
                }
            }
        }
        if (!found) {
            int new_id = (c.memory_id != 0) ? c.memory_id : generateObstacleId();
            TrackedObstacle tobs;
            tobs.id = new_id;
            tobs.circle = c;
            tobs.circle.memory_id = new_id;
            tobs.last_seen = now;
            obstacle_memory_[new_id] = tobs;
        }
    }
    
    last_tracked_obstacles_.circles.clear();
    last_tracked_obstacles_.header = msg->header;
    for (auto it = obstacle_memory_.begin(); it != obstacle_memory_.end(); ) {
        double age = (now - it->second.last_seen).toSec();
        if (age > obstacle_memory_duration_) {
            it = obstacle_memory_.erase(it);
        } else {
            auto circle_copy = it->second.circle;
            // Extrapolate position slightly if not seen recently
            if (age > 0.05) {
                circle_copy.center.x += circle_copy.velocity.x * age;
                circle_copy.center.y += circle_copy.velocity.y * age;
            }
            last_tracked_obstacles_.circles.push_back(circle_copy);
            ++it;
        }
    }
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

    private_nh.param("alpha", alpha_, 0.35);
    private_nh.param("beta", beta_, 10.0);
    private_nh.param("gamma", gamma_, 0.14);
    private_nh.param("kappa", kappa_, 1.0);
    private_nh.param("epsilon", epsilon_, 2.0);

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

    private_nh.param("robot_radius", robot_radius_, 0.3);
    private_nh.param("safe_clearance_dist", safe_clearance_dist_, 0.3);
    private_nh.param("hard_collision_buffer", hard_collision_buffer_, 0.2);
    private_nh.param("radar_range", radar_range_, 4.0);
    private_nh.param("safety_multiplier_unstable", safety_mult_unstable_, 1.3);
    private_nh.param("safety_multiplier_ghost", safety_mult_ghost_, 1.5);
    private_nh.param("safety_multiplier_ghost_unstable",
                     safety_mult_ghost_unstable_, 1.8);

    private_nh.param("obstacle_memory_duration", obstacle_memory_duration_, 1.5);
    private_nh.param("prediction_horizon", prediction_horizon_, 0.3);
    private_nh.param("rotation_cost", rotation_cost_, 0.4);
    private_nh.param("approach_penalty_weight", approach_penalty_weight_, 2.0);
    private_nh.param("corridor_predict_time", corridor_predict_time_, 8.0);
    private_nh.param("corridor_cruising_speed", corridor_cruising_speed_, 0.15);
    private_nh.param("safe_crossing_margin", safe_crossing_margin_, 0.8);
    private_nh.param("corridor_clear_margin", corridor_clear_margin_, 0.15);
    private_nh.param("allow_dynamic_fallback", allow_dynamic_fallback_, true);
    private_nh.param("fallback_speed_scale", fallback_speed_scale_, 0.35);
    private_nh.param("fallback_min_safety", fallback_min_safety_, 0.05);

    odom_helper_.setOdomTopic("odom");
    initialized_ = true;
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

  std::lock_guard<std::mutex> lock(obstacles_mutex_);

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

  for (const auto &obs : last_tracked_obstacles_.circles) {
      double dx = obs.center.x - robot_pose.pose.position.x;
      double dy = obs.center.y - robot_pose.pose.position.y;
      double dist = std::hypot(dx, dy);
      double look_dot = dx * std::cos(robot_yaw) + dy * std::sin(robot_yaw);

      // Relative velocity projection to determine if object is approaching
      double rx = -dx;
      double ry = -dy;
      double approach_vel = 0.0;
      if (dist > 0.01) {
          approach_vel = (obs.velocity.x * rx + obs.velocity.y * ry) / dist;
      }

      if (dist < 1.5 && look_dot < -0.3 * dist) {
          if (approach_vel < 0.1) {
              passed_obstacles_[obs.memory_id] = ros::Time::now();
          } else {
              passed_obstacles_.erase(obs.memory_id);
          }
      } else {
          if (approach_vel > 0.1 && look_dot > -0.1 * dist) {
              passed_obstacles_.erase(obs.memory_id);
          }
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
  Trajectory best_fallback_traj;
  best_fallback_traj.cost = -1.0;
  bool has_fallback = false;
  double best_fallback_score = -1e9;
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
      if (traj.cost < 0.0 && traj.reject_reason == REJECT_DYNAMIC) {
        if (traj.debug_min_safety >= fallback_min_safety_) {
          double fallback_score =
              (alpha_ * (traj.debug_path + traj.debug_heading)) +
              (gamma_ * traj.debug_velocity) -
              (rotation_cost_ * traj.vth * traj.vth);
          double safety_term = (traj.debug_min_safety < 0.3)
                                   ? traj.debug_min_safety
                                   : 0.3;
          fallback_score += epsilon_ * safety_term;

          if (!has_fallback || fallback_score > best_fallback_score) {
            best_fallback_score = fallback_score;
            best_fallback_traj = traj;
            has_fallback = true;
          }
        }
      }
      all_trajectories.push_back(traj);

      if (traj.cost >= 0.0 && traj.cost > best_traj.cost)
        best_traj = traj;
    }
  }

  publishMarkers(all_trajectories);

  if (best_traj.cost < 0) {
    if (allow_dynamic_fallback_ && has_fallback) {
      cmd_vel.linear.x = best_fallback_traj.vx * fallback_speed_scale_;
      cmd_vel.angular.z = best_fallback_traj.vth * fallback_speed_scale_;

      nav_msgs::Path local_path;
      local_path.header.stamp = ros::Time::now();
      local_path.header.frame_id = costmap_ros_->getGlobalFrameID();
      local_path.poses = best_fallback_traj.poses;
      local_plan_pub_.publish(local_path);

      ROS_WARN_THROTTLE(1.0,
                        "[DWA] Fallback traj used | V=%.2f W=%.2f | Score=%.3f",
                        cmd_vel.linear.x, cmd_vel.angular.z,
                        best_fallback_score);
      publishTrackedObstaclesViz();
      return true;
    }
    cmd_vel.linear.x = 0;
    cmd_vel.angular.z = 0;
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
    ROS_INFO("[DWA] State: %s | V=%.2f W=%.2f | Safety: %.3f | ApproachPen: %.3f | RotPen: %.3f",
             best_traj.debug_state.c_str(), best_traj.vx, best_traj.vth,
             best_traj.cost, best_traj.debug_speed_bonus, best_traj.debug_turning_bonus);
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
      traj.reject_reason = REJECT_STATIC;
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
  traj.debug_path = path_dist_score;
  traj.debug_heading = heading_score;

  // 3. Штраф за близость к статическим препятствиям (Beta)
  double dist_score =
      1.0 - (max_footprint_cost / costmap_2d::INSCRIBED_INFLATED_OBSTACLE);

  // 4. Скорость (Gamma)
  double robot_yaw = tf2::getYaw(robot_pose.pose.orientation);
  double dist_to_goal = std::hypot(
      robot_pose.pose.position.x - global_plan_.back().pose.position.x,
      robot_pose.pose.position.y - global_plan_.back().pose.position.y);

  double velocity_score = 0.0;
  if (dist_to_goal > 1.0) {
    double cruise_vel = 0.6;
    velocity_score = 1.0 - (std::abs(traj.vx - cruise_vel) / max_vel_x_);
  } else {
    double desired_vel = max_vel_x_ * (dist_to_goal / 1.0);
    velocity_score = 1.0 - (std::abs(traj.vx - desired_vel) / max_vel_x_);
  }
  traj.debug_velocity = velocity_score;

  // 5. ДИНАМИЧЕСКИЙ АНАЛИЗ С УЛУЧШЕНИЯМИ
  double min_safety_score = 1.0;
  double critical_heading_score = 0.0;
  const tracked_obstacle_msgs::TrackedCircle *critical_obs = nullptr;

  // =========================================================================
  // 6. DYNAMICНЫЕ ПРЕПЯТСТВИЯ — ЕДИНЫЙ МАТЕМАТИЧЕСКИЙ ПОДХОД
  //
  // Вместо хардкода (S_MANEUVER, PASS_BEHIND, и т.д.) используем:
  //   a) calculate_dis_fp — уже даёт safety∈[0,1] для ЛЮБОГО угла атаки
  //   b) approach_penalty — штраф ∝ скорости сближения с препятствием
  //   c) rotation_cost   — квадратичный штраф за ω, исключает 360°
  // =========================================================================

  double approach_penalty = 0.0;

  for (const auto &tracked_obs : last_tracked_obstacles_.circles) {
    bool is_passed = (passed_obstacles_.find(tracked_obs.memory_id) != passed_obstacles_.end());

    double multiplier = getSafetyMultiplier(tracked_obs);
    double safety = calculate_dis_fp(traj, tracked_obs, multiplier);

    // Пропускаем уже обогнанные препятствия, которые далеко позади
    if (is_passed) {
      double dx = tracked_obs.center.x - robot_pose.pose.position.x;
      double dy = tracked_obs.center.y - robot_pose.pose.position.y;
      double dist = std::hypot(dx, dy);
      double look_dot = dx * std::cos(robot_yaw) + dy * std::sin(robot_yaw);
      if (dist < 3.0 && look_dot < -0.2 * dist) {
        continue;
      }
    }

    // Накапливаем минимальную safety (худшее препятствие из всех)
    if (safety < min_safety_score) {
      min_safety_score = safety;
      critical_obs = &tracked_obs;
    }

    // ПОДХОД-ШТРАФ: штраф ∝ компонент скорости объекта НАВСТРЕЧУ роботу.
    // Это и есть "предиктивное" уклонение — геометрически нейтральное к углу.
    // Работает одинаково при лобовом столкновении, косом, боковом, и т.д.
    double dx = tracked_obs.center.x - robot_pose.pose.position.x;
    double dy = tracked_obs.center.y - robot_pose.pose.position.y;
    double obs_dist = std::hypot(dx, dy);

    if (obs_dist > 0.01 && obs_dist < radar_range_) {
      // Вектор от робота к объекту (нормализованный)
      double to_obs_x = dx / obs_dist;
      double to_obs_y = dy / obs_dist;
      
      // Глобальный вектор скорости симулируемой траектории
      double robot_vx_global = traj.vx * std::cos(robot_yaw);
      double robot_vy_global = traj.vx * std::sin(robot_yaw);
      
      // Скорость робота НАПРАВЛЕННАЯ к объекту
      double robot_v_toward = robot_vx_global * to_obs_x + robot_vy_global * to_obs_y;
      
      // Скорость объекта, направленная к роботу
      // (минус, потому что to_obs смотрит от робота к объекту)
      double approach_vel = -(tracked_obs.velocity.x * to_obs_x + tracked_obs.velocity.y * to_obs_y);
      
      // Нас интересует только сближающийся объект
      if (approach_vel > 0.1) {
          // relative_approach - это общая скорость сближения
          double relative_approach = robot_v_toward + approach_vel;
          
          if (relative_approach > 0) {
              double proximity_factor = std::max(0.0, 1.0 - obs_dist / radar_range_);
              // Штрафуем траекторию за скорость сближения
              approach_penalty += relative_approach * proximity_factor * approach_penalty_weight_;
          }
      }
    }
  }

  // Жёсткая блокировка при критической близости
  const double hard_safety_reject = 0.003;
  if (min_safety_score < hard_safety_reject) {
    traj.debug_min_safety = min_safety_score;
    traj.debug_alpha = min_safety_score;
    traj.reject_reason = REJECT_DYNAMIC;
    return -100.0 + min_safety_score;
  }

  // ШТРАФ ЗА УГЛОВУЮ СКОРОСТЬ (квадратичный, нет хардкода)
  // Позволяет умеренно повернуть, но жёстко штрафует широкие дуги/360°
  double rotation_penalty = rotation_cost_ * traj.vth * traj.vth;

  // Итоговая формула — полностью математическая, без состояний:
  //   path_following + static_safety + velocity + dynamic_safety - approach - rotation
  traj.debug_turning_bonus = -rotation_penalty;
  traj.debug_speed_bonus = -approach_penalty;
  traj.debug_alpha = min_safety_score;
  traj.debug_min_safety = min_safety_score;
  traj.debug_state = (critical_obs != nullptr) ? "DYN_AVOID" : "IDLE";

  return (alpha_ * (path_dist_score + heading_score)) +
         (beta_ * dist_score) +
         (gamma_ * velocity_score) +
         (epsilon_ * min_safety_score) -
         approach_penalty -
         rotation_penalty;
}

// =============================================================================
// HELPER: Оценка направления движения
// =============================================================================
double ImprovedDWALocalPlanner::calculate_dis_hv(
    const Trajectory &traj, const geometry_msgs::PoseStamped &robot_pose,
    const tracked_obstacle_msgs::TrackedCircle &obs) {

  double obs_speed = std::hypot(obs.velocity.x, obs.velocity.y);
  double angle_to_obs = atan2(obs.center.y - robot_pose.pose.position.y,
                              obs.center.x - robot_pose.pose.position.x);

  double d = std::hypot(obs.center.x - robot_pose.pose.position.x,
                        obs.center.y - robot_pose.pose.position.y);

  double eta = std::max(0.0, radar_range_ - d);
  
  double final_yaw = tf2::getYaw(robot_pose.pose.orientation);
  if (!traj.poses.empty()) {
    final_yaw = tf2::getYaw(traj.poses.back().pose.orientation);
  }
  
  double theta = angles::shortest_angular_distance(final_yaw, angle_to_obs);
  double heading_factor = 0.5 * (1.0 - std::cos(theta)); 
  
  double rel_speed_proxy = std::abs(traj.vx) + obs_speed;
  double ghost_discount = obs.is_ghost ? 0.6 : 1.0;

  return eta * rel_speed_proxy * heading_factor * ghost_discount;
}

// =============================================================================
// HELPER: Предсказание столкновения (ИСПРАВЛЕНА ЛОГИКА КОРИДОРА)
// =============================================================================
double ImprovedDWALocalPlanner::calculate_dis_fp(
    Trajectory &traj, const tracked_obstacle_msgs::TrackedCircle &obs,
    double safety_multiplier) {

  double min_dist_sq = 1e6;
  int collision_idx = -1;
  double collision_time = -1.0;

  double core_limit = obs.radius + robot_radius_;
  double obs_v_sq = obs.velocity.x * obs.velocity.x + obs.velocity.y * obs.velocity.y;
  double v_obs = std::sqrt(obs_v_sq);

  bool core_hit = false;
  bool corridor_hit = false;

  bool is_dynamic = (v_obs > 0.1);
  bool is_passed = (passed_obstacles_.find(obs.memory_id) != passed_obstacles_.end());

  for (size_t i = 0; i < traj.poses.size(); ++i) {
    double t = i * dt_;

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

    // ИСПРАВЛЕНА ЛОГИКА КОРИДОРА: Временная проверка вместо пространственной
    if (is_dynamic && !is_passed) {
        double rx = traj.poses[i].pose.position.x - obs.center.x;
        double ry = traj.poses[i].pose.position.y - obs.center.y;
        double dot = rx * obs.velocity.x + ry * obs.velocity.y;
        
        if (dot > 0) {
            double proj_len = dot / v_obs;
            
            // КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ: Горизонт планирования коридора (corridor_predict_time_)
            // Робот должен знать о надвигающейся угрозе заранее, даже если она приедет после 2.5с
            if (proj_len < v_obs * corridor_predict_time_) {
                double dynamic_core_limit = core_limit + proj_len * 0.05; // Легкое расширение коридора для дальних

                
                double dist_to_line_sq = (rx * rx + ry * ry) - (dot * dot / obs_v_sq);
                
                // ВРЕМЕННАЯ ПРОВЕРКА: Будет ли объект в этой точке ОДНОВРЕМЕННО или робот задержится там?
                double time_for_obj_to_reach = proj_len / v_obs;
                
                // Если робот находится в коридоре в момент t, нужно понять, успеет ли он уехать.
                // Если робот движется быстро и это не конец траектории — он проедет.
                bool robot_will_clear = false;
                if (std::abs(traj.vx) > corridor_cruising_speed_) {
                    // Если он проезжает эту точку гораздо раньше, чем приедет объект (запас > safe_crossing_margin_)
                    if (t < time_for_obj_to_reach - safe_crossing_margin_) {
                        robot_will_clear = true; 
                    }
                }

                // ВАЖНО: Если скорость робота близка к нулю (он остановился или стоит), 
                // он НЕ покинет эту точку. Значит, robot_will_clear = false, и если 
                // объект едет в эту точку, это НЕИЗБЕЖНЫЙ corridor hit.
                
                if (dist_to_line_sq < dynamic_core_limit * dynamic_core_limit) {
                    if (!robot_will_clear) {
                        corridor_hit = true;
                    }
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
          if (collision_idx == 0) {
              return 0.1;
          }
      }
      traj.collision_pose_idx = collision_idx;
      traj.obstacle_pos.x = obs.center.x + obs.velocity.x * collision_time;
      traj.obstacle_pos.y = obs.center.y + obs.velocity.y * collision_time;
      traj.obstacle_start_pos = obs.center;
      return 0.0;
  }

  double hard_limit = (obs.radius + robot_radius_ + hard_collision_buffer_) * safety_multiplier;
  
  if (min_dist_sq <= hard_limit * hard_limit) {
    if (start_d2 <= hard_limit * hard_limit) {
        if (collision_idx == 0 || min_dist_sq > start_d2 - 1e-4) {
            return 0.2;
        }
    }

    traj.collision_pose_idx = collision_idx;
    traj.obstacle_pos.x = obs.center.x + obs.velocity.x * collision_time;
    traj.obstacle_pos.y = obs.center.y + obs.velocity.y * collision_time;
    traj.obstacle_pos.z = 0;
    traj.obstacle_start_pos = obs.center;
    
    return 0.0;
  }

  double min_dist = std::sqrt(min_dist_sq);
  double soft_limit = (obs.radius + robot_radius_ + safe_clearance_dist_) * safety_multiplier;
  
  if (min_dist < soft_limit) {
    double score = (min_dist - hard_limit) / (soft_limit - hard_limit);
    if (score < 0) score = 0;

    // Штраф за коридор применяется всегда, даже если робот стоит (иначе он просто ждёт столкновения)
    if (corridor_hit) {
      return (0.35 + 0.65 * std::pow(score, 2)) * 0.75;
    }
    return 0.35 + 0.65 * std::pow(score, 2);
  }

  if (corridor_hit && min_dist > (soft_limit + corridor_clear_margin_)) {
    return 1.0;
  }

  // Штраф за дальний коридор применяется всегда
  if (corridor_hit) {
    return 0.7;
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
    bool is_relevant = (v > 0.2) || obs.is_ghost || obs.is_unstable;
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
      // УКОРОЧЕНА ВИЗУАЛИЗАЦИЯ КОРИДОРА: 3.5 секунды вместо 8.5
      double core_limit = obs.radius + robot_radius_;
      double yaw = atan2(obs.velocity.y, obs.velocity.x);

      visualization_msgs::Marker corridor;
      corridor.header = text.header;
      corridor.ns = "tracked_obstacles_corridor";
      corridor.id = obs.memory_id;
      corridor.type = visualization_msgs::Marker::CUBE;
      // Центр на середине 3.5-секундного отрезка
      corridor.pose.position.x = obs.center.x + obs.velocity.x * 1.75;
      corridor.pose.position.y = obs.center.y + obs.velocity.y * 1.75;
      corridor.pose.position.z = 0.01;
      
      tf2::Quaternion q;
      q.setRPY(0, 0, yaw);
      corridor.pose.orientation = tf2::toMsg(q);
      
      corridor.scale.x = v * 3.5; // Длина (3.5 секунды * скорость)
      corridor.scale.y = core_limit * 2.0;
      corridor.scale.z = 0.01;
      
      corridor.color.r = 1.0f;
      corridor.color.g = 0.0f;
      corridor.color.b = 0.0f;
      corridor.color.a = 0.15f;
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