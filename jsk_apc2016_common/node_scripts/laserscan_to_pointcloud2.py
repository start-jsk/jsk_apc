#!/usr/bin/env python

import rospy
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import LaserScan
from laser_geometry import LaserProjection

def callback(laserscan_msg):
    laserscan_point = LaserProjection()
    pointcloud2_point = laserscan_point.projectLaser(laserscan_msg)
    pub = rospy.Publisher('~output', PointCloud2, queue_size=100)
    pub.publish(pointcloud2_point)

if __name__ == '__main__':
    rospy.init_node('laserscan_to_pointcloud2')
    sub = rospy.Subscriber('~input', LaserScan, callback)

    rospy.spin()
