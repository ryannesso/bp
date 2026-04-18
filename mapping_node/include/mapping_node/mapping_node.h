#pragma once
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/OccupancyGrid.h>
#include <nav_msgs/Odometry.h>

nav_msgs::OccupancyGrid buildOccupancy(const sensor_msgs::PointCloud2::ConstPtr& cloud);
nav_msgs::OccupancyGrid buildESDF(const nav_msgs::OccupancyGrid& occ);
nav_msgs::OccupancyGrid buildDeltaESDF(const nav_msgs::OccupancyGrid& esdf);

class MappingNode {
public:
    MappingNode(ros::NodeHandle& nh);
    void cloudCallback(const sensor_msgs::PointCloud2::ConstPtr& cloud);
    void odomCallback(const nav_msgs::Odometry::ConstPtr& odom);
    void process();
private:
    ros::NodeHandle nh_;
    ros::Subscriber cloud_sub_;
    ros::Subscriber odom_sub_;
    ros::Publisher occ_pub_;
    ros::Publisher esdf_pub_;
    ros::Publisher delta_pub_;
    sensor_msgs::PointCloud2::ConstPtr last_cloud_;
    nav_msgs::Odometry::ConstPtr last_odom_;
};
