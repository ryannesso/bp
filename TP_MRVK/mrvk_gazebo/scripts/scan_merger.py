#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import LaserScan
import math

class ScanMerger:
    def __init__(self):
        rospy.init_node('simple_scan_merger')
        self.pub = rospy.Publisher('/scan_slam_ready', LaserScan, queue_size=1)
        self.scan1 = None
        self.scan2 = None
        rospy.Subscriber('/scan_cleaned', LaserScan, self.cb1)
        rospy.Subscriber('/grass_scan', LaserScan, self.cb2)
        rospy.loginfo("Scan Merger initializing... Waiting for /scan_cleaned and /grass_scan")

    def cb1(self, msg):
        self.scan1 = msg
        self.merge_and_publish()

    def cb2(self, msg):
        self.scan2 = msg
        self.merge_and_publish()

    def merge_and_publish(self):
        # We need the main lidar scan as the baseline
        if not self.scan1: 
            return
            
        msg = LaserScan()
        msg.header = self.scan1.header
        msg.angle_min = self.scan1.angle_min
        msg.angle_max = self.scan1.angle_max
        msg.angle_increment = self.scan1.angle_increment
        msg.time_increment = self.scan1.time_increment
        msg.scan_time = self.scan1.scan_time
        msg.range_min = self.scan1.range_min
        msg.range_max = self.scan1.range_max
        msg.ranges = list(self.scan1.ranges)
        
        # If we also have the grass scan, mix them by taking the minimum distance
        if self.scan2 and self.scan2.header.frame_id == self.scan1.header.frame_id:
            # We assume angle_min, angle_max and angle_increment are roughly similar.
            # Convert grass ranges into the main ranges array
            for i in range(len(self.scan2.ranges)):
                r2 = self.scan2.ranges[i]
                if r2 > self.scan2.range_min and r2 < self.scan2.range_max and not math.isinf(r2) and not math.isnan(r2):
                    angle2 = self.scan2.angle_min + i * self.scan2.angle_increment
                    
                    # Find corresponding index in main scan
                    idx1 = int((angle2 - msg.angle_min) / msg.angle_increment)
                    if 0 <= idx1 < len(msg.ranges):
                        r1 = msg.ranges[idx1]
                        if math.isnan(r1) or math.isinf(r1) or r2 < r1:
                            msg.ranges[idx1] = r2

        self.pub.publish(msg)

if __name__ == '__main__':
    try:
        ScanMerger()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
