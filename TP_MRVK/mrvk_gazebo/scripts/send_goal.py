#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import PoseStamped

def sender():
    rospy.init_node('goal_sender_node')
    # Načítame súradnice z parametrov (ktoré pošleme z launchu)
    gx = rospy.get_param('~x', 11.029)
    gy = rospy.get_param('~y', -0.899)
    
    pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=1)
    
    # Čakáme, kým sa simulácia poriadne rozbehne (napr. 10 sekúnd)
    rospy.loginfo("Čakám 10 sekúnd na inicializáciu simulácie...")
    rospy.sleep(6.8) #7, 6.8
    
    goal = PoseStamped()
    goal.header.frame_id = "map"
    goal.header.stamp = rospy.Time.now()
    goal.pose.position.x = gx
    goal.pose.position.y = gy
    goal.pose.orientation.w = 1.0
    
    rospy.loginfo(f"Posielam cieľ: x={gx}, y={gy}")
    pub.publish(goal)
    rospy.loginfo("Cieľ odoslaný. Uzol končí.")

if __name__ == '__main__':
    try:
        sender()
    except rospy.ROSInterruptException:
        pass