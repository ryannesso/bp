#!/usr/bin/env python3
import rospy
import math
import actionlib
import subprocess
import os
import signal
from datetime import datetime
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState, ModelStates
from geometry_msgs.msg import PoseStamped
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from nav_msgs.msg import Odometry, Path
from obstacle_detector.msg import Obstacles

class ExperimentOrchestrator:
    def __init__(self):
        rospy.init_node('experiment_orchestrator')

        # --- KONFIGURÁCIA SCÉNY ---
        self.robot_name = "robot"         
        self.sphere_name = "moving_sphere"
        self.init_x = 0.0
        self.init_y = -5.978              # Старт робота в Gazebo
        
        # Настройки для движения "в лоб"
        self.amplitude = 4.0               # Сфера будет ходить от -4.0 до +4.0 по Y
        self.frequency = 0.05        # Частота (скорость движения)
        self.fixed_x = -0.205792               # Робот едет по X=0, значит и сфера на X=0
        
        # Цель робота (едет прямо по коридору)
        self.goal_x = rospy.get_param('~goal_x', 0.0) 
        self.goal_y = rospy.get_param('~goal_y', 6.0)

        # --- STAV EXPERIMENTU ---
        self.active = False
        self.start_time = 0
        self.obs_detected = False
        self.last_recorded_time = rospy.Time(0)
        self.record_interval = 0.1  
        self.bag_process = None     

        # --- DÁTA PRE MATLAB ---
        self.path_robot = Path()
        self.path_robot.header.frame_id = "map"
        self.path_sphere = Path()
        self.path_sphere.header.frame_id = "map"

        # --- KOMUNIKÁCIA ---
        rospy.wait_for_service('/gazebo/set_model_state')
        self.set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        
        self.pub_odom = rospy.Publisher('/moving_sphere/odom', Odometry, queue_size=10)
        self.pub_path_robot = rospy.Publisher('/path_robot_fixed', Path, queue_size=1)
        self.pub_path_sphere = rospy.Publisher('/path_sphere_fixed', Path, queue_size=1)

        self.move_base_client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        
        rospy.Subscriber('/obstacles', Obstacles, self.obs_check_callback)
        rospy.Subscriber('/gazebo/model_states', ModelStates, self.model_states_callback)

        rospy.loginfo("Orchestrátor: Čakám na move_base...")
        self.move_base_client.wait_for_server()
        
        self.start_experiment()

    def obs_check_callback(self, msg):
        if len(msg.circles) > 0:
            self.obs_detected = True

    def stop_recording(self):
        if self.bag_process:
            rospy.loginfo("Ukončujem nahrávanie rosbagu...")
            self.bag_process.send_signal(signal.SIGINT)
            self.bag_process.wait()
            self.bag_process = None

    def model_states_callback(self, msg):
        if not self.active:
            return

        state = self.move_base_client.get_state()
        if state in [actionlib.GoalStatus.SUCCEEDED, actionlib.GoalStatus.ABORTED]:
            self.active = False
            self.stop_recording()
            return

        now = rospy.Time.now()
        t = (now - self.start_time).to_sec()

        # --- 1. ДВИЖЕНИЕ СФЕРЫ "ЛОБ В ЛОБ" (по оси Y) ---
        # Сфера качается вперед-назад по той же линии, где едет робот
        y_sph = self.amplitude * math.sin(2 * math.pi * self.frequency * t)
        vy_sph = self.amplitude * 2 * math.pi * self.frequency * math.cos(2 * math.pi * self.frequency * t)

        state_msg = ModelState()
        state_msg.model_name = self.sphere_name
        state_msg.pose.position.x = self.fixed_x
        state_msg.pose.position.y = y_sph
        state_msg.pose.position.z = 0.3
        state_msg.pose.orientation.w = 1.0
        state_msg.twist.linear.y = vy_sph # Скорость теперь по Y!
        state_frame = 'world'
        
        try:
            self.set_state(state_msg)
        except: pass

        # Odom для трекера
        odom = Odometry()
        odom.header.stamp = now
        odom.header.frame_id = 'odom'
        odom.pose.pose.position.x = self.fixed_x
        odom.pose.pose.position.y = y_sph
        odom.pose.pose.position.z = 0.3
        odom.twist.twist.linear.y = vy_sph
        self.pub_odom.publish(odom)

        # --- 2. ЗАПИСЬ ДЛЯ MATLAB ---
        if (now - self.last_recorded_time).to_sec() >= self.record_interval:
            if self.robot_name in msg.name:
                r_idx = msg.name.index(self.robot_name)
                rp = msg.pose[r_idx].position
                
                # Координаты робота в "карте" (с учетом его старта в -5.9)
                ps_r = PoseStamped()
                ps_r.header.stamp = now
                ps_r.pose.position.x = rp.x - self.init_x
                ps_r.pose.position.y = rp.y - self.init_y
                self.path_robot.poses.append(ps_r)

                # Координаты сферы в той же системе "карты"
                ps_s = PoseStamped()
                ps_s.header.stamp = now
                ps_s.pose.position.x = self.fixed_x - self.init_x
                ps_s.pose.position.y = y_sph - self.init_y
                self.path_sphere.poses.append(ps_s)

                self.pub_path_robot.publish(self.path_robot)
                self.pub_path_sphere.publish(self.path_sphere)
                self.last_recorded_time = now

    def start_experiment(self):
        warmup = 10
        rospy.loginfo(f"Stabilizácia systému... Čakám {warmup}s.")
        
        start_wait = rospy.get_time()
        while not self.obs_detected and (rospy.get_time() - start_wait) < warmup:
            rospy.loginfo_throttle(2, "Čakám na detekciu prekážky...")
            rospy.sleep(0.5)

        # --- АВТО-ИМЯ ФАЙЛА С ДАТОЙ ---
        bag_folder = os.path.expanduser("~/catkin_ws/src/TP_MRVK/mrvk_gazebo/bags")
        if not os.path.exists(bag_folder): os.makedirs(bag_folder)
        
        timestamp = datetime.now().strftime("%H%M%S")
        bag_path = os.path.join(bag_folder, "imr_1_head_2.bag")
        
        rospy.loginfo(f"Запись в файл: {bag_path}")
        self.bag_process = subprocess.Popen(["rosbag", "record", "-O", bag_path, 
                                            "/path_robot_fixed", "/path_sphere_fixed", "/obstacles"])

        # Отправка цели (прямо вперед по коридору)
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = self.goal_x
        goal.target_pose.pose.position.y = self.goal_y
        goal.target_pose.pose.orientation.w = 1.0

        rospy.loginfo(f"ШТАРТ: Робот едет в ({self.goal_x}, {self.goal_y})")
        self.move_base_client.send_goal(goal)

        self.start_time = rospy.Time.now()
        self.active = True

if __name__ == '__main__':
    try:
        orchestrator = ExperimentOrchestrator()
        rospy.on_shutdown(orchestrator.stop_recording)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass