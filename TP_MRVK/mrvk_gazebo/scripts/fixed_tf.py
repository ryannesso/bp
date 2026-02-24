#!/usr/bin/env python3
import rospy
from gazebo_msgs.msg import ModelStates
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped

class FinalPathSync:
    def __init__(self):
        rospy.init_node('final_path_sync')
        
        # --- KONFIGURÁCIA ---
        self.robot_name = "robot"         
        self.sphere_name = "moving_sphere" 
        
        # Realny štart robota v Gazebo
        self.init_x = 0.0
        self.init_y = -5.978
        
        # --- ОГРАНИЧЕНИЕ ЧАСТОТЫ ---
        self.last_recorded_time = rospy.Time.now()
        self.record_interval = 0.1  # Записывать каждые 0.1 сек (10 Hz)
        # ---------------------------

        self.path_robot = Path()
        self.path_robot.header.frame_id = "map"
        self.path_sphere = Path()
        self.path_sphere.header.frame_id = "map"

        self.pub_robot = rospy.Publisher('/path_robot_fixed', Path, queue_size=1)
        self.pub_sphere = rospy.Publisher('/path_sphere_fixed', Path, queue_size=1)

        rospy.Subscriber('/gazebo/model_states', ModelStates, self.callback)
        rospy.loginfo("Sync script bezi (10Hz). Nahrávajte témy /path_robot_fixed a /path_sphere_fixed")

    def callback(self, msg):
        # Проверяем, прошло ли достаточно времени с последней записи
        current_time = rospy.Time.now()
        if (current_time - self.last_recorded_time).to_sec() < self.record_interval:
            return

        if self.robot_name in msg.name and self.sphere_name in msg.name:
            r_idx = msg.name.index(self.robot_name)
            s_idx = msg.name.index(self.sphere_name)

            # Обновляем время последней записи
            self.last_recorded_time = current_time

            # 1. POLOHA ROBOTA
            rp = msg.pose[r_idx].position
            ps_r = PoseStamped()
            ps_r.header.stamp = current_time
            ps_r.header.frame_id = "map"
            ps_r.pose.position.x = rp.x - self.init_x
            ps_r.pose.position.y = rp.y - self.init_y
            self.path_robot.poses.append(ps_r)

            # 2. POLOHA SFÉRY
            sp = msg.pose[s_idx].position
            ps_s = PoseStamped()
            ps_s.header.stamp = current_time
            ps_s.header.frame_id = "map"
            ps_s.pose.position.x = sp.x - self.init_x
            ps_s.pose.position.y = sp.y - self.init_y
            self.path_sphere.poses.append(ps_s)

            # Публикуем обновленные пути
            self.path_robot.header.stamp = current_time
            self.path_sphere.header.stamp = current_time
            self.pub_robot.publish(self.path_robot)
            self.pub_sphere.publish(self.path_sphere)

if __name__ == '__main__':
    try:
        FinalPathSync()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass