#!/usr/bin/env python3

import math

import rospy
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from nav_msgs.msg import Odometry


class MoveTwoSpheres:
    def __init__(self):
        rospy.init_node('move_two_spheres_node', anonymous=True)

        self.s1_name = rospy.get_param('~s1_name', 'moving_sphere')
        self.s2_name = rospy.get_param('~s2_name', 'moving_sphere_2')

        # Defaults mirror TP_MRVK/mrvk_gazebo/scripts/run_experiment.py
        self.amplitude = float(rospy.get_param('~amplitude', 3.0))
        self.frequency = float(rospy.get_param('~frequency', 0.08))
        self.frequency_s1 = float(rospy.get_param('~frequency_s1', self.frequency * 1.2))
        self.frequency_s2 = float(rospy.get_param('~frequency_s2', self.frequency))
        self.s1_y_pos = float(rospy.get_param('~s1_y_pos', 1.5))
        self.s2_y_pos = float(rospy.get_param('~s2_y_pos', 0.0))
        self.z_pos = float(rospy.get_param('~z_pos', 0.3))
        self.rate_hz = float(rospy.get_param('~rate', 100.0))

        rospy.wait_for_service('/gazebo/set_model_state')
        self.set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

        self.pub_odom1 = rospy.Publisher(f'/{self.s1_name}/odom', Odometry, queue_size=10)
        self.pub_odom2 = rospy.Publisher(f'/{self.s2_name}/odom', Odometry, queue_size=10)

        self.rate = rospy.Rate(self.rate_hz)
        self.start_time = rospy.Time.now()

    def _set_model(self, name: str, x: float, y: float, vx: float) -> None:
        st = ModelState()
        st.model_name = name
        st.reference_frame = 'world'
        st.pose.position.x = x
        st.pose.position.y = y
        st.pose.position.z = self.z_pos
        st.pose.orientation.w = 1.0
        st.twist.linear.x = vx
        st.twist.linear.y = 0.0
        st.twist.linear.z = 0.0

        self.set_state(st)

    @staticmethod
    def _publish_odom(pub: rospy.Publisher, stamp: rospy.Time, child_frame: str, x: float, y: float, z: float, vx: float) -> None:
        o = Odometry()
        o.header.stamp = stamp
        o.header.frame_id = 'odom'
        o.child_frame_id = child_frame
        o.pose.pose.position.x = x
        o.pose.pose.position.y = y
        o.pose.pose.position.z = z
        o.pose.pose.orientation.w = 1.0
        o.twist.twist.linear.x = vx
        pub.publish(o)

    def run(self) -> None:
        while not rospy.is_shutdown():
            now = rospy.Time.now()
            t = (now - self.start_time).to_sec()

            # Spheres move against each other (same as run_experiment.py)
            x1 = self.amplitude * math.sin(2 * math.pi * self.frequency_s1 * t)
            vx1 = self.amplitude * 2 * math.pi * self.frequency_s1 * math.cos(2 * math.pi * self.frequency_s1 * t)

            x2 = -self.amplitude * math.sin(2 * math.pi * self.frequency_s2 * t)
            vx2 = -self.amplitude * 2 * math.pi * self.frequency_s2 * math.cos(2 * math.pi * self.frequency_s2 * t)

            try:
                self._set_model(self.s1_name, x1, self.s1_y_pos, vx1)
                self._set_model(self.s2_name, x2, self.s2_y_pos, vx2)
            except rospy.ServiceException as e:
                rospy.logerr("/gazebo/set_model_state failed: %s", e)

            self._publish_odom(self.pub_odom1, now, self.s1_name, x1, self.s1_y_pos, self.z_pos, vx1)
            self._publish_odom(self.pub_odom2, now, self.s2_name, x2, self.s2_y_pos, self.z_pos, vx2)

            self.rate.sleep()


if __name__ == '__main__':
    try:
        MoveTwoSpheres().run()
    except rospy.ROSInterruptException:
        pass
