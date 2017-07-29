#!/usr/bin/env python

import argparse

import cv_bridge
import jsk_arc2017_common
import rospkg
from sensor_msgs.msg import Image
import rospy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('object_name')
    args = parser.parse_args(rospy.myargv()[1:])

    object_name = args.object_name

    rospy.init_node('visualize_object')

    pub = rospy.Publisher('~output', Image, queue_size=1)
    rospy.sleep(1)

    rp = rospkg.RosPack()
    PKG_PATH = rp.get_path('jsk_arc2017_common')

    object_images = jsk_arc2017_common.get_object_images()

    if object_name not in object_images:
        rospy.logerr('Invalid object name: %s' % object_name)
        return

    img_obj = object_images[object_name]

    bridge = cv_bridge.CvBridge()
    imgmsg = bridge.cv2_to_imgmsg(img_obj, encoding='rgb8')
    imgmsg.header.stamp = rospy.Time.now()

    pub.publish(imgmsg)
    rospy.spin()


if __name__ == '__main__':
    main()
