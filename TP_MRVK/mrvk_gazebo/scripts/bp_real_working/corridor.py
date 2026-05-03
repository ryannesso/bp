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

        # --- 1. KONFIGURÁCIA ---
        self.robot_name = "robot"         
        self.s1_name = "moving_sphere_1"    
        self.s2_name = "moving_sphere_2"  
        
        self.init_x = 0.0
        self.init_y = -5.978              # Offset pre MATLAB (štart robota)
        
        # Nastavenia pohybu sfér (kmitanie zľava-doprava)
        self.amplitude = 3.0              
        self.frequency = 0.08             
        self.s1_y_pos = 1.5               # Fixná Y súradnica prvej sféry
        self.s2_y_pos = 0.0               # Fixná Y súradnica druhej sféry
        
        # Cieľ robota (vysoko v mape)
        self.goal_x = rospy.get_param('~goal_x', 11.421)
        self.goal_y = rospy.get_param('~goal_y', -0.054)

        # --- 2. STAV ---
        self.active = False
        self.start_time = 0
        self.obs_count = 0
        self.last_recorded_time = rospy.Time(0)
        self.record_interval = 0.1        
        self.bag_process = None

        # --- 3. DÁTA PRE MATLAB ---
        self.path_robot = Path()
        self.path_robot.header.frame_id = "map"
        self.path_s1 = Path()
        self.path_s1.header.frame_id = "map"
        self.path_s2 = Path()
        self.path_s2.header.frame_id = "map"

        # --- 4. KOMUNIKÁCIA ---
        rospy.wait_for_service('/gazebo/set_model_state')
        self.set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        
        self.pub_odom1 = rospy.Publisher('/moving_sphere_1/odom', Odometry, queue_size=10)
        self.pub_odom2 = rospy.Publisher('/moving_sphere_2/odom', Odometry, queue_size=10)
        
        self.pub_path_robot = rospy.Publisher('/path_robot_fixed', Path, queue_size=1)
        self.pub_path_s1 = rospy.Publisher('/path_sphere1_fixed', Path, queue_size=1)
        self.pub_path_s2 = rospy.Publisher('/path_sphere2_fixed', Path, queue_size=1)

        self.move_base_client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        
        rospy.Subscriber('/obstacles', Obstacles, self.obs_callback)
        rospy.Subscriber('/gazebo/model_states', ModelStates, self.model_states_callback)

        rospy.loginfo("Orchestrátor: Čakám na move_base server...")
        self.move_base_client.wait_for_server()
        
        self.start_experiment()

    def obs_callback(self, msg):
        self.obs_count = len(msg.circles)

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

        # 1. VÝPOČET POHYBU (Sféry idú proti sebe)
        x1 = self.amplitude * math.sin(2 * math.pi * self.frequency * t)
        vx1 = self.amplitude * 2 * math.pi * self.frequency * math.cos(2 * math.pi * self.frequency * t)
        
        x2 = -self.amplitude * math.sin(2 * math.pi * self.frequency * t)
        vx2 = -vx1

        # 2. AKTUALIZÁCIA GAZEBO
        def update_gz(name, x, y, vx):
            st = ModelState()
            st.model_name = name
            st.pose.position.x = x
            st.pose.position.y = y
            st.pose.position.z = 0.3
            st.twist.linear.x = vx
            st.reference_frame = 'world'
            try: self.set_state(st)
            except: pass

        update_gz(self.s1_name, x1, self.s1_y_pos, vx1)
        update_gz(self.s2_name, x2, self.s2_y_pos, vx2)

        # 3. ODOM PRE DETEKTOR
        def pub_o(p, x, y, vx):
            o = Odometry(); o.header.stamp = now; o.header.frame_id = 'odom'
            o.pose.pose.position.x = x; o.pose.pose.position.y = y; o.pose.pose.position.z = 0.3
            o.twist.twist.linear.x = vx
            p.publish(o)

        pub_o(self.pub_odom1, x1, self.s1_y_pos, vx1)
        pub_o(self.pub_odom2, x2, self.s2_y_pos, vx2)

        # 4. ZÁPIS CIEST PRE MATLAB
        if (now - self.last_recorded_time).to_sec() >= self.record_interval:
            if self.robot_name in msg.name:
                r_idx = msg.name.index(self.robot_name)
                rp = msg.pose[r_idx].position
                
                # Robot
                ps_r = PoseStamped(); ps_r.header.stamp = now
                ps_r.pose.position.x = rp.x - self.init_x
                ps_r.pose.position.y = rp.y - self.init_y
                self.path_robot.poses.append(ps_r)

                # Sféra 1
                ps_s1 = PoseStamped(); ps_s1.header.stamp = now
                ps_s1.pose.position.x = x1 - self.init_x
                ps_s1.pose.position.y = self.s1_y_pos - self.init_y
                self.path_s1.poses.append(ps_s1)

                # Sféra 2
                ps_s2 = PoseStamped(); ps_s2.header.stamp = now
                ps_s2.pose.position.x = x2 - self.init_x
                ps_s2.pose.position.y = self.s2_y_pos - self.init_y
                self.path_s2.poses.append(ps_s2)

                self.pub_path_robot.publish(self.path_robot)
                self.pub_path_s1.publish(self.path_s1)
                self.pub_path_s2.publish(self.path_s2)
                self.last_recorded_time = now

    def start_experiment(self):
        
        # 2. Skontrolujeme, či detektor už vidí prekážky
        start_wait = rospy.get_time()

        # 3. Spustíme nahrávanie
        bag_dir = os.path.expanduser("~/catkin_ws/src/TP_MRVK/mrvk_gazebo/bags")
        if not os.path.exists(bag_dir): os.makedirs(bag_dir)
        ts = datetime.now().strftime("%H%M%S")
        bag_path = os.path.join(bag_dir, f"dwa_1_ob.bag")
        
        rospy.loginfo(f"Nahrávam experiment do: {bag_path}")
        topics = [
            "/path_robot_fixed",
            "/path_sphere1_fixed",
            "/path_sphere2_fixed",
            "/moving_sphere_1/odom",
            "/moving_sphere_2/odom",
            "/obstacles"
        ]
        self.bag_process = subprocess.Popen(["rosbag", "record", "-O", bag_path] + topics)

        # 4. ŠTART! Odoslanie cieľa rozbehne sféry v model_states_callback
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = self.goal_x
        goal.target_pose.pose.position.y = self.goal_y
        goal.target_pose.pose.orientation.w = 1.0
        
        self.move_base_client.send_goal(goal)
        self.start_time = rospy.Time.now()
        self.active = True
        rospy.loginfo("--- ŠTART EXPERIMENTU ---")

if __name__ == '__main__':
    try:
        ExperimentOrchestrator()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass