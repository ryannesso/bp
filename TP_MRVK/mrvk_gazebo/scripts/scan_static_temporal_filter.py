#!/usr/bin/env python3
import math

import rospy
from sensor_msgs.msg import LaserScan


class TemporalStaticScanFilter:
    def __init__(self):
        rospy.init_node("scan_static_temporal_filter")

        self.input_topic = rospy.get_param("~input_topic", "/scan_filtered")
        self.output_topic = rospy.get_param("~output_topic", "/scan_static_slam")

        # Confidence model for static-vs-dynamic per laser beam.
        self.conf_threshold = rospy.get_param("~confidence_threshold", 3.0)
        self.conf_inc = rospy.get_param("~confidence_inc", 1.0)
        self.conf_dec = rospy.get_param("~confidence_dec", 1.4)
        self.conf_invalid_dec = rospy.get_param("~confidence_invalid_dec", 0.6)
        self.conf_max = rospy.get_param("~confidence_max", 12.0)
        self.reset_conf = rospy.get_param("~reset_confidence", 0.2)

        # Beam consistency threshold.
        self.tol_abs = rospy.get_param("~range_tolerance_abs", 0.12)
        self.tol_rel = rospy.get_param("~range_tolerance_rel", 0.08)

        # Exponential updates.
        self.ema_alpha = rospy.get_param("~ema_alpha", 0.82)
        self.outlier_blend = rospy.get_param("~outlier_blend", 0.25)

        # Bootstrap to avoid empty map in first seconds.
        self.warmup_frames = rospy.get_param("~warmup_frames", 8)

        self.debug_log = rospy.get_param("~debug_log", True)

        self.initialized = False
        self.n_beams = 0
        self.ema = []
        self.conf = []
        self.frame_count = 0

        self.pub = rospy.Publisher(self.output_topic, LaserScan, queue_size=1)
        self.sub = rospy.Subscriber(self.input_topic, LaserScan, self.cb, queue_size=1)

        rospy.loginfo(
            "[scan_static_filter] Started: in=%s out=%s threshold=%.2f",
            self.input_topic,
            self.output_topic,
            self.conf_threshold,
        )

    def _init_state(self, n_beams):
        self.n_beams = n_beams
        self.ema = [None] * n_beams
        self.conf = [0.0] * n_beams
        self.frame_count = 0
        self.initialized = True

    @staticmethod
    def _is_valid(r, rmin, rmax):
        return (not math.isnan(r)) and (not math.isinf(r)) and (r > rmin) and (r < rmax)

    def cb(self, msg):
        if (not self.initialized) or (len(msg.ranges) != self.n_beams):
            self._init_state(len(msg.ranges))

        self.frame_count += 1

        out = LaserScan()
        out.header = msg.header
        out.angle_min = msg.angle_min
        out.angle_max = msg.angle_max
        out.angle_increment = msg.angle_increment
        out.time_increment = msg.time_increment
        out.scan_time = msg.scan_time
        out.range_min = msg.range_min
        out.range_max = msg.range_max
        out.intensities = msg.intensities

        out_ranges = [float("inf")] * self.n_beams
        kept = 0

        for i in range(self.n_beams):
            r = msg.ranges[i]
            valid = self._is_valid(r, msg.range_min, msg.range_max)

            if not valid:
                self.conf[i] = max(0.0, self.conf[i] - self.conf_invalid_dec)
                if self.conf[i] <= self.reset_conf:
                    self.ema[i] = None
                continue

            if self.ema[i] is None:
                self.ema[i] = r
                self.conf[i] = min(self.conf_max, self.conf[i] + 0.5 * self.conf_inc)
            else:
                tol = self.tol_abs + self.tol_rel * max(r, self.ema[i])
                if abs(r - self.ema[i]) <= tol:
                    self.ema[i] = self.ema_alpha * self.ema[i] + (1.0 - self.ema_alpha) * r
                    self.conf[i] = min(self.conf_max, self.conf[i] + self.conf_inc)
                else:
                    self.conf[i] = max(0.0, self.conf[i] - self.conf_dec)
                    self.ema[i] = (1.0 - self.outlier_blend) * self.ema[i] + self.outlier_blend * r
                    if self.conf[i] <= self.reset_conf:
                        self.ema[i] = r

            if self.frame_count <= self.warmup_frames or self.conf[i] >= self.conf_threshold:
                out_ranges[i] = r
                kept += 1

        out.ranges = out_ranges
        self.pub.publish(out)

        if self.debug_log:
            rospy.loginfo_throttle(
                1.0,
                "[scan_static_filter] frame=%d kept=%d/%d (%.1f%%)",
                self.frame_count,
                kept,
                self.n_beams,
                100.0 * float(kept) / max(1, self.n_beams),
            )


if __name__ == "__main__":
    try:
        node = TemporalStaticScanFilter()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
