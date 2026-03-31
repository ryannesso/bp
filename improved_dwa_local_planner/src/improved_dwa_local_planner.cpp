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
// ВСПОМОГАТЕЛЬНАЯ ФУНКЦИЯ: Расчёт коэффициента безопасности
// Должна быть определена до первого вызова.
//
//  ACTIVE          : multiplier = 1.0  (стандартные зоны)
//  UNSTABLE        : multiplier = 1.3  (объект непредсказуем — расширяем зону
//  на 30%) GHOST           : multiplier = 1.5  (позиция неточна — расширяем
//  зону на 50%) GHOST + UNSTABLE: multiplier = 1.8
// =============================================================================
static double
getSafetyMultiplier(const tracked_obstacle_msgs::TrackedCircle &obs) {
  if (obs.is_ghost && obs.is_unstable)
    return 1.8;
  if (obs.is_ghost)
    return 1.5;
  if (obs.is_unstable)
    return 1.3;
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
            if (calculate_dis_fp(traj, tracked_obs, multiplier) < 0.15) {
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

    odom_helper_.setOdomTopic("odom");
    initialized_ = true;
    ROS_INFO("[ImprovedDWA] Initialized. Subscribed to /obstacles_tracked "
             "(ghost/unstable-aware).");
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
    ROS_WARN_THROTTLE(1.0, "[ImprovedDWA] No safe trajectory found. STOP.");
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
    ROS_INFO("[DWA] V=%.2f W=%.2f | Dyn=%d Ghost=%d Unstable=%d", best_traj.vx,
             best_traj.vth, dyn, ghost, unstable);
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
    velocity_score = traj.vx / max_vel_x_;
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
    if (v_obs > speed_threshold_)
      ++dynamic_cnt;
    if (tracked_obs.is_ghost)
      ++ghost_cnt;
    if (tracked_obs.is_unstable)
      ++unstable_cnt;

    bool is_relevant = (v_obs > speed_threshold_) || tracked_obs.is_ghost;
    if (!is_relevant)
      continue;

    double multiplier = getSafetyMultiplier(tracked_obs);
    double safety = calculate_dis_fp(traj, tracked_obs, multiplier);

    if (safety < min_safety_score) {
      min_safety_score = safety;
      critical_obs = &tracked_obs;
    }
  }

  ROS_INFO_THROTTLE(
      0.5, "[DWA] Obstacles: total=%d | dyn=%d | ghost=%d | unstable=%d",
      total_obs, dynamic_cnt, ghost_cnt, unstable_cnt);

  if (critical_obs != nullptr) {
    critical_heading_score =
        calculate_dis_hv(traj.vx, robot_pose, *critical_obs);
  }

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
    double toward_factor = std::max(0.0, cos(angles::shortest_angular_distance(
                                             obs_vel_angle, angle_to_robot)));

    double danger_scale = getSafetyMultiplier(*critical_obs);
    double aggressive_danger =
        in_front_factor * toward_factor * std::min(danger_scale, 1.0);

    double evade_side =
        (angles::shortest_angular_distance(robot_yaw, angle_to_obs) > 0) ? -1.0
                                                                         : 1.0;

    if (aggressive_danger > 0.5) {
      current_alpha *= 0.3;
      if (traj.vth * evade_side > 0)
        turning_bonus += 8.0 * std::abs(traj.vth);
      speed_bonus += 5.0;
    } else {
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

    ROS_INFO_THROTTLE(
        0.2,
        "[DWA] CritObs: ghost=%d unstable=%d | aggr=%.2f | multiplier=%.1f",
        (int)critical_obs->is_ghost, (int)critical_obs->is_unstable,
        aggressive_danger, getSafetyMultiplier(*critical_obs));
  }

  if (min_safety_score < 0.05) {
    return -100.0 + min_safety_score;
  }

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
    double robot_vx, const geometry_msgs::PoseStamped &robot_pose,
    const tracked_obstacle_msgs::TrackedCircle &obs) {

  double obs_speed = std::hypot(obs.velocity.x, obs.velocity.y);
  double obs_heading = atan2(obs.velocity.y, obs.velocity.x);
  double d = std::hypot(obs.center.x - robot_pose.pose.position.x,
                        obs.center.y - robot_pose.pose.position.y);

  double eta = std::max(0.0, 4.0 - d);
  double theta = angles::shortest_angular_distance(
      tf2::getYaw(robot_pose.pose.orientation), obs_heading);
  double heading_factor = 0.5 * (1.0 + std::cos(theta));
  double rel_speed_proxy = std::abs(robot_vx) + obs_speed;
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
  double collision_time = -1.0;

  for (size_t i = 0; i < traj.poses.size(); ++i) {
    double t = i * dt_;

    double pred_obs_x = obs.center.x + obs.velocity.x * t;
    double pred_obs_y = obs.center.y + obs.velocity.y * t;
    double dx = traj.poses[i].pose.position.x - pred_obs_x;
    double dy = traj.poses[i].pose.position.y - pred_obs_y;
    double d2 = dx * dx + dy * dy;

    if (d2 < min_dist_sq)
      min_dist_sq = d2;

    double hard_limit = (obs.radius + 0.2) * safety_multiplier;
    if (d2 <= hard_limit * hard_limit) {
      if (collision_time < 0) {
        collision_time = t;
        traj.collision_pose_idx = (int)i;
        traj.obstacle_pos.x = pred_obs_x;
        traj.obstacle_pos.y = pred_obs_y;
        traj.obstacle_pos.z = 0;
        traj.obstacle_start_pos = obs.center;
      }
    }
  }

  if (collision_time >= 0) {
    return (collision_time / predict_time_) * 0.1;
  }

  double min_dist = std::sqrt(min_dist_sq);
  const double robot_radius_approx = 0.3;
  const double safe_clearance_dist = 0.3;
  double soft_limit = (obs.radius + robot_radius_approx + safe_clearance_dist) *
                      safety_multiplier;
  double hard_limit2 = obs.radius + robot_radius_approx + 0.05;

  if (min_dist < soft_limit) {
    double score = (min_dist - hard_limit2) / (soft_limit - hard_limit2);
    if (score < 0)
      score = 0;
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
        (v > speed_threshold_) || obs.is_ghost || obs.is_unstable;
    if (!is_relevant)
      continue;

    float r_col = 1.0f, g_col = 0.0f, b_col = 0.0f, a_col = 0.8f;
    std::string state_label;
    if (obs.is_ghost && obs.is_unstable) {
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
    double hard_limit = (obs.radius + 0.2) * multiplier;
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

    double soft_limit = (obs.radius + 0.6) * multiplier;
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

    if (v > 0.05) {
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