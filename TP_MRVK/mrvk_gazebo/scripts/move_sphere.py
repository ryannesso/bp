#!/usr/bin/env python3
import rospy
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState
from nav_msgs.msg import Odometry
import math

class MoveSphere:
    def __init__(self):
        rospy.init_node('move_sphere_node', anonymous=True)
        self.pub_odom = rospy.Publisher('/moving_sphere/odom', Odometry, queue_size=10)
        self.set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self.rate = rospy.Rate(100)  # 100 Hz
        self.amplitude = 2.0  # Амплитуда движения (метры)
        self.frequency = 0.1  # Частота движения (Гц)

    def move_sphere(self):
        model_state = ModelState()
        model_state.model_name = 'moving_sphere'
        model_state.reference_frame = 'world'
        odom_msg = Odometry()
        odom_msg.header.frame_id = 'odom'
        odom_msg.child_frame_id = 'moving_sphere'

        while not rospy.is_shutdown():
            t = rospy.get_time()
            # Вычисляем смещение по X (синусоида)
            x_offset = self.amplitude * math.sin(2 * math.pi * self.frequency * t)
            vx = self.amplitude * 2 * math.pi * self.frequency * math.cos(2 * math.pi * self.frequency * t)

            # Обновление состояния модели в Gazebo
            model_state.pose.position.x = 0.0 + x_offset  # Центрируем вокруг X=2.0
            model_state.pose.position.y = 0.0             # Фиксированный Y
            model_state.pose.position.z = 0.5
            model_state.pose.orientation.w = 1.0
            
            model_state.twist.linear.x = vx               # Скорость по X
            model_state.twist.linear.y = 0.0
            model_state.twist.linear.z = 0.0

            # Публикация одометрии
            odom_msg.header.stamp = rospy.get_rostime()
            odom_msg.pose.pose.position.x = 2.0 + x_offset
            odom_msg.pose.pose.position.y = 0.0
            odom_msg.pose.pose.position.z = 0.5
            odom_msg.pose.pose.orientation.w = 1.0
            odom_msg.twist.twist.linear.x = vx
            odom_msg.twist.twist.linear.y = 0.0
            self.pub_odom.publish(odom_msg)

            try:
                self.set_state(model_state)
            except rospy.ServiceException as e:
                rospy.logerr("Service call failed: %s" % e)

            self.rate.sleep()

if __name__ == '__main__':
    try:
        mover = MoveSphere()
        mover.move_sphere()
    except rospy.ROSInterruptException:
        pass
