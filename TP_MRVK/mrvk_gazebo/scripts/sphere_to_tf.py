#!/usr/bin/env python3
import rospy
import tf
from gazebo_msgs.msg import ModelStates

class SphereTFBridge:
    def __init__(self):
        rospy.init_node('sphere_tf_bridge')
        self.br = tf.TransformBroadcaster()
        self.sphere_name = "moving_sphere"
        self.last_time = rospy.Time(0)
        
        rospy.Subscriber('/gazebo/model_states', ModelStates, self.callback)

    def callback(self, msg):
        # Získame aktuálny čas simulácie
        curr_time = rospy.Time.now()
        
        # Ak sa čas nepohol (napr. Gazebo posiela duplicitné stavy), ignorujeme to
        if curr_time <= self.last_time:
            return
            
        if self.sphere_name in msg.name:
            idx = msg.name.index(self.sphere_name)
            p = msg.pose[idx].position
            o = msg.pose[idx].orientation
            
            # Publikujeme sféru voči 'world' (Gazebo súradnice)
            self.br.sendTransform(
                (p.x, p.y, p.z),
                (o.x, o.y, o.z, o.w),
                curr_time,
                "sphere_link",
                "world"
            )
            self.last_time = curr_time

if __name__ == '__main__':
    try:
        SphereTFBridge()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass