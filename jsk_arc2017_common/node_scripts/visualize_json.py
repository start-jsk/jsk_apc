#!/usr/bin/env python

import os.path as osp

import cv_bridge
import jsk_arc2017_common
from sensor_msgs.msg import Image
import rospy


class VisualizeJSON(object):

    def __init__(self):
        super(VisualizeJSON, self).__init__()
        self._json_dir = rospy.get_param('~json_dir')
        self._types = rospy.get_param('~types')
        if not all(t in ['item_location', 'order'] for t in self._types):
            rospy.logfatal('Unsupported type is included: %s' % self._types)
            quit(1)
        self.pub_item_location = rospy.Publisher(
            '~output/item_location_viz', Image, queue_size=1)
        self.pub_order = rospy.Publisher(
            '~output/order_viz', Image, queue_size=1)
        rate = rospy.get_param('~rate', 1)
        self.timer = rospy.Timer(rospy.Duration(1. / rate), self._cb)

    def _cb(self, event):
        bridge = cv_bridge.CvBridge()
        if 'item_location' in self._types:
            filename = osp.join(self._json_dir, 'item_location_file.json')
            if osp.exists(filename):
                img = jsk_arc2017_common.visualize_item_location(filename)
                imgmsg = bridge.cv2_to_imgmsg(img, encoding='rgb8')
                imgmsg.header.stamp = rospy.Time.now()
                self.pub_item_location.publish(imgmsg)
            else:
                rospy.logwarn_throttle(10, '%s does not exists yet' % filename)
        if 'order' in self._types:
            filename = osp.join(self._json_dir, 'order_file.json')
            if osp.exists(filename):
                img = jsk_arc2017_common.visualize_order(filename)
                imgmsg = bridge.cv2_to_imgmsg(img, encoding='rgb8')
                imgmsg.header.stamp = rospy.Time.now()
                self.pub_order.publish(imgmsg)
            else:
                rospy.logwarn_throttle(10, '%s does not exists yet' % filename)


if __name__ == '__main__':
    rospy.init_node('visualize_json')
    app = VisualizeJSON()
    rospy.spin()
