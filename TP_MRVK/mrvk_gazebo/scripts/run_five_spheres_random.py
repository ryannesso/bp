#!/usr/bin/env python3
import math
import random

import actionlib
import rospy
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal


class FiveSpheresRandomController:
    def __init__(self):
        rospy.init_node("five_spheres_random_controller")

        self.area_x_min = rospy.get_param("~area_x_min", -2.2)
        self.area_x_max = rospy.get_param("~area_x_max", 2.2)
        self.area_y_min = rospy.get_param("~area_y_min", -1.0)
        self.area_y_max = rospy.get_param("~area_y_max", 3.0)
        self.sphere_z = rospy.get_param("~sphere_z", 0.3)

        self.goal_x = rospy.get_param("~goal_x", 0.0)
        self.goal_y = rospy.get_param("~goal_y", 7.0)

        self.update_rate_hz = rospy.get_param("~update_rate", 10.0)
        self.speed_min = rospy.get_param("~speed_min", 0.55)
        self.speed_max = rospy.get_param("~speed_max", 1.05)
        self.waypoint_tolerance = rospy.get_param("~waypoint_tolerance", 0.15)

        # Realism parameters (social-force style)
        self.relaxation_time = rospy.get_param("~relaxation_time", 0.8)
        self.max_accel = rospy.get_param("~max_accel", 1.2)
        self.speed_hard_cap = rospy.get_param("~speed_hard_cap", 1.25)
        self.noise_accel = rospy.get_param("~noise_accel", 0.25)

        self.neighbor_radius = rospy.get_param("~neighbor_radius", 1.4)
        self.separation_distance = rospy.get_param("~separation_distance", 0.75)
        self.separation_gain = rospy.get_param("~separation_gain", 1.8)

        self.wall_buffer = rospy.get_param("~wall_buffer", 0.8)
        self.wall_gain = rospy.get_param("~wall_gain", 1.2)

        self.repath_min = rospy.get_param("~repath_min", 2.5)
        self.repath_max = rospy.get_param("~repath_max", 5.5)
        self.crossing_bias = rospy.get_param("~crossing_bias", 0.7)

        self.pause_probability = rospy.get_param("~pause_probability", 0.18)
        self.pause_min = rospy.get_param("~pause_min", 0.3)
        self.pause_max = rospy.get_param("~pause_max", 1.2)

        self.spawn_clearance = rospy.get_param("~spawn_clearance", 0.9)

        self.sphere_names = rospy.get_param(
            "~sphere_names",
            [
                "moving_sphere_1",
                "moving_sphere_2",
                "moving_sphere_3",
                "moving_sphere_4",
                "moving_sphere_5",
            ],
        )

        self.spheres = {}
        random.seed()

        rospy.wait_for_service("/gazebo/set_model_state")
        self.set_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)

        self.move_base_client = actionlib.SimpleActionClient("move_base", MoveBaseAction)
        rospy.loginfo("[five_spheres] Waiting for move_base action server...")
        self.move_base_client.wait_for_server()

        self._init_spheres()
        self._send_goal()

    def _rand_point(self, margin=0.2):
        x = random.uniform(self.area_x_min + margin, self.area_x_max - margin)
        y = random.uniform(self.area_y_min + margin, self.area_y_max - margin)
        return x, y

    def _new_speed(self):
        return random.uniform(self.speed_min, self.speed_max)

    def _new_repath_time(self, now_t):
        return now_t + random.uniform(self.repath_min, self.repath_max)

    def _new_target(self, current_x, current_y):
        # Bias toward crossing trajectories so agents often traverse the robot corridor.
        if random.random() < self.crossing_bias:
            if current_x < 0.5 * (self.area_x_min + self.area_x_max):
                tx = random.uniform(self.area_x_max - 1.2, self.area_x_max - 0.2)
            else:
                tx = random.uniform(self.area_x_min + 0.2, self.area_x_min + 1.2)
            ty = random.uniform(self.area_y_min + 0.2, self.area_y_max - 0.2)
            return tx, ty
        return self._rand_point()

    def _sample_non_overlapping_start(self):
        for _ in range(200):
            x, y = self._rand_point()
            ok = True
            for s in self.spheres.values():
                if math.hypot(x - s["x"], y - s["y"]) < self.spawn_clearance:
                    ok = False
                    break
            if ok:
                return x, y
        return self._rand_point()

    def _init_spheres(self):
        now_t = rospy.Time.now().to_sec()
        for name in self.sphere_names:
            x, y = self._sample_non_overlapping_start()
            tx, ty = self._new_target(x, y)
            self.spheres[name] = {
                "x": x,
                "y": y,
                "vx": 0.0,
                "vy": 0.0,
                "yaw": random.uniform(-math.pi, math.pi),
                "tx": tx,
                "ty": ty,
                "pref_speed": self._new_speed(),
                "pause_until": 0.0,
                "repath_at": self._new_repath_time(now_t),
            }
            self._apply_state(name, x, y, 0.0, 0.0, self.spheres[name]["yaw"])

        rospy.loginfo(
            "[five_spheres] Initialized %d spheres in area x:[%.2f, %.2f], y:[%.2f, %.2f]",
            len(self.sphere_names),
            self.area_x_min,
            self.area_x_max,
            self.area_y_min,
            self.area_y_max,
        )

    def _send_goal(self):
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = self.goal_x
        goal.target_pose.pose.position.y = self.goal_y
        goal.target_pose.pose.orientation.w = 1.0

        self.move_base_client.send_goal(goal)
        rospy.loginfo(
            "[five_spheres] Goal sent: x=%.2f y=%.2f (straight above dynamic area)",
            self.goal_x,
            self.goal_y,
        )

    def _apply_state(self, model_name, x, y, vx, vy, yaw):
        state = ModelState()
        state.model_name = model_name
        state.pose.position.x = x
        state.pose.position.y = y
        state.pose.position.z = self.sphere_z
        state.pose.orientation.z = math.sin(0.5 * yaw)
        state.pose.orientation.w = math.cos(0.5 * yaw)
        state.twist.linear.x = vx
        state.twist.linear.y = vy
        state.reference_frame = "world"

        try:
            self.set_state(state)
        except rospy.ServiceException:
            pass

    def _social_forces(self, name, s):
        fx = 0.0
        fy = 0.0

        # Inter-agent repulsion.
        for other_name, o in self.spheres.items():
            if other_name == name:
                continue
            dx = s["x"] - o["x"]
            dy = s["y"] - o["y"]
            d = math.hypot(dx, dy)
            if d < 1e-4 or d > self.neighbor_radius:
                continue

            # Stronger push when too close.
            if d < self.separation_distance:
                k = self.separation_gain * (self.separation_distance - d) / self.separation_distance
                fx += k * (dx / d)
                fy += k * (dy / d)

        # Boundary soft forces.
        dl = s["x"] - self.area_x_min
        dr = self.area_x_max - s["x"]
        db = s["y"] - self.area_y_min
        dt = self.area_y_max - s["y"]

        if dl < self.wall_buffer:
            fx += self.wall_gain * (self.wall_buffer - dl) / self.wall_buffer
        if dr < self.wall_buffer:
            fx -= self.wall_gain * (self.wall_buffer - dr) / self.wall_buffer
        if db < self.wall_buffer:
            fy += self.wall_gain * (self.wall_buffer - db) / self.wall_buffer
        if dt < self.wall_buffer:
            fy -= self.wall_gain * (self.wall_buffer - dt) / self.wall_buffer

        # Small random perturbation for less robotic trajectories.
        fx += random.uniform(-self.noise_accel, self.noise_accel)
        fy += random.uniform(-self.noise_accel, self.noise_accel)
        return fx, fy

    def _desired_velocity(self, s, now_t):
        if now_t < s["pause_until"]:
            return 0.0, 0.0

        dx = s["tx"] - s["x"]
        dy = s["ty"] - s["y"]
        dist = math.hypot(dx, dy)

        if dist < self.waypoint_tolerance or now_t >= s["repath_at"]:
            s["tx"], s["ty"] = self._new_target(s["x"], s["y"])
            s["pref_speed"] = self._new_speed()
            s["repath_at"] = self._new_repath_time(now_t)
            if random.random() < self.pause_probability:
                s["pause_until"] = now_t + random.uniform(self.pause_min, self.pause_max)
                return 0.0, 0.0
            dx = s["tx"] - s["x"]
            dy = s["ty"] - s["y"]
            dist = max(1e-6, math.hypot(dx, dy))

        ux = dx / max(1e-6, dist)
        uy = dy / max(1e-6, dist)
        return ux * s["pref_speed"], uy * s["pref_speed"]

    def _enforce_bounds(self, s):
        # Reflect gently from hard borders.
        if s["x"] < self.area_x_min:
            s["x"] = self.area_x_min
            s["vx"] = abs(s["vx"]) * 0.25
        elif s["x"] > self.area_x_max:
            s["x"] = self.area_x_max
            s["vx"] = -abs(s["vx"]) * 0.25

        if s["y"] < self.area_y_min:
            s["y"] = self.area_y_min
            s["vy"] = abs(s["vy"]) * 0.25
        elif s["y"] > self.area_y_max:
            s["y"] = self.area_y_max
            s["vy"] = -abs(s["vy"]) * 0.25

    def spin(self):
        rate = rospy.Rate(self.update_rate_hz)
        last_t = rospy.Time.now().to_sec()

        while not rospy.is_shutdown():
            now_t = rospy.Time.now().to_sec()
            dt = max(0.001, min(0.2, now_t - last_t))
            last_t = now_t

            for name, s in self.spheres.items():
                desired_vx, desired_vy = self._desired_velocity(s, now_t)
                social_fx, social_fy = self._social_forces(name, s)

                ax = (desired_vx - s["vx"]) / max(0.05, self.relaxation_time) + social_fx
                ay = (desired_vy - s["vy"]) / max(0.05, self.relaxation_time) + social_fy

                a_norm = math.hypot(ax, ay)
                if a_norm > self.max_accel:
                    scale = self.max_accel / a_norm
                    ax *= scale
                    ay *= scale

                s["vx"] += ax * dt
                s["vy"] += ay * dt

                v_norm = math.hypot(s["vx"], s["vy"])
                if v_norm > self.speed_hard_cap:
                    scale = self.speed_hard_cap / v_norm
                    s["vx"] *= scale
                    s["vy"] *= scale

                s["x"] += s["vx"] * dt
                s["y"] += s["vy"] * dt
                self._enforce_bounds(s)

                if math.hypot(s["vx"], s["vy"]) > 0.03:
                    s["yaw"] = math.atan2(s["vy"], s["vx"])

                self._apply_state(name, s["x"], s["y"], s["vx"], s["vy"], s["yaw"])

            rate.sleep()


if __name__ == "__main__":
    try:
        node = FiveSpheresRandomController()
        node.spin()
    except rospy.ROSInterruptException:
        pass
