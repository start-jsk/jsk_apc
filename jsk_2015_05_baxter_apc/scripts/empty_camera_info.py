#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

import rospy
from sensor_msgs.msg import CameraInfo


def main():
    rospy.init_node('empty_camera_info')
    pub_of_caminfo = rospy.Publisher('~output', CameraInfo,
                                     queue_size=1)
    rate = rospy.Rate(1)
    while not rospy.is_shutdown():
        pub_of_caminfo.publish()
        rate.sleep()


if __name__ == '__main__':
    main()
