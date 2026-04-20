#!/usr/bin/env python3
"""
slam_map_cleaner.py — Post-SLAM map cleaning node (Approach 2)

Subscribes to /map (from slam_toolbox) and /obstacles_tracked (from dynamic_filter).
For each tracked dynamic obstacle, erases its footprint from the OccupancyGrid
and publishes the cleaned map on /map_cleaned.
"""

import math
import rospy
from nav_msgs.msg import OccupancyGrid
from tracked_obstacle_msgs.msg import TrackedCircleArray


class SlamMapCleaner:
    def __init__(self):
        rospy.init_node("slam_map_cleaner", anonymous=False)

        self.clear_radius_buffer = rospy.get_param("~clear_radius_buffer", 0.5)
        self.clear_ghost = rospy.get_param("~clear_ghost", True)
        self.ghost_ttl = rospy.get_param("~ghost_ttl", 3.0)
        self.publish_rate = rospy.get_param("~publish_rate", 2.0)
        self.history_duration = rospy.get_param("~history_duration", 10.0)

        self.latest_map = None
        self.latest_tracked = None
        self.obstacle_history = []

        self.map_sub = rospy.Subscriber("/map", OccupancyGrid, self.map_callback, queue_size=1)
        self.tracked_sub = rospy.Subscriber(
            "/obstacles_tracked", TrackedCircleArray, self.tracked_callback, queue_size=1
        )
        self.map_pub = rospy.Publisher("/map_cleaned", OccupancyGrid, queue_size=1, latch=True)
        self.timer = rospy.Timer(rospy.Duration(1.0 / self.publish_rate), self.timer_callback)

        rospy.loginfo("[SlamMapCleaner] Started. buf=%.2f hist=%.1fs",
                      self.clear_radius_buffer, self.history_duration)

    def map_callback(self, msg):
        self.latest_map = msg

    def tracked_callback(self, msg):
        self.latest_tracked = msg
        now = rospy.Time.now()
        for c in msg.circles:
            if c.is_ghost and not self.clear_ghost:
                continue
            if c.is_ghost and c.time_since_seen > self.ghost_ttl:
                continue
            self.obstacle_history.append((
                c.center.x, c.center.y,
                max(0.1, c.radius) + self.clear_radius_buffer, now
            ))
        cutoff = now - rospy.Duration(self.history_duration)
        self.obstacle_history = [h for h in self.obstacle_history if h[3] > cutoff]

    def timer_callback(self, event):
        if self.latest_map is None:
            return

        cleaned = OccupancyGrid()
        cleaned.header = self.latest_map.header
        cleaned.header.stamp = rospy.Time.now()
        cleaned.info = self.latest_map.info
        cleaned.data = list(self.latest_map.data)

        res = cleaned.info.resolution
        ox = cleaned.info.origin.position.x
        oy = cleaned.info.origin.position.y
        w = cleaned.info.width
        h = cleaned.info.height
        if w == 0 or h == 0 or res <= 0:
            return

        zones = []
        if self.latest_tracked is not None:
            for c in self.latest_tracked.circles:
                if c.is_ghost and not self.clear_ghost:
                    continue
                if c.is_ghost and c.time_since_seen > self.ghost_ttl:
                    continue
                r = max(0.1, c.radius) + self.clear_radius_buffer
                zones.append((c.center.x, c.center.y, r))

        for hx, hy, hr, _ in self.obstacle_history:
            zones.append((hx, hy, hr))

        for cx, cy, r in zones:
            gx = (cx - ox) / res
            gy = (cy - oy) / res
            rc = int(math.ceil(r / res)) + 1
            i_min = max(0, int(gx) - rc)
            i_max = min(w - 1, int(gx) + rc)
            j_min = max(0, int(gy) - rc)
            j_max = min(h - 1, int(gy) + rc)
            r2 = r * r

            for j in range(j_min, j_max + 1):
                for i in range(i_min, i_max + 1):
                    wx = ox + (i + 0.5) * res
                    wy = oy + (j + 0.5) * res
                    if (wx - cx) ** 2 + (wy - cy) ** 2 <= r2:
                        idx = j * w + i
                        if 0 <= idx < len(cleaned.data) and cleaned.data[idx] > 0:
                            cleaned.data[idx] = 0

        self.map_pub.publish(cleaned)


if __name__ == "__main__":
    try:
        node = SlamMapCleaner()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
