#!/usr/bin/env python

import os.path as osp

import cv_bridge
import jsk_arc2017_common
from jsk_topic_tools import ConnectionBasedTransport
from sensor_msgs.msg import Image
from std_msgs.msg import String
import rospy


class VisualizeJSON(ConnectionBasedTransport):

    def __init__(self):
        super(VisualizeJSON, self).__init__()
        self._types = rospy.get_param('~types')
        if not all(t in ['item_location', 'order'] for t in self._types):
            rospy.logfatal('Unsupported type is included: %s' % self._types)
            quit(1)
        self.pub_item_location = self.advertise(
            '~output/item_location_viz', Image, queue_size=1)
        self.pub_order = self.advertise(
            '~output/order_viz', Image, queue_size=1)

    def subscribe(self):
        self.sub = rospy.Subscriber('~input/json_dir', String, self._cb)

    def unsubscribe(self):
        self.sub.unregister()

    def _cb(self, msg):
        json_dir = msg.data
        if not osp.isdir(json_dir):
            rospy.logfatal_throttle(
                10, 'Input json_dir is not directory: %s' % json_dir)
            return

        bridge = cv_bridge.CvBridge()
        if 'item_location' in self._types:
            filename = osp.join(json_dir, 'item_location_file.json')
            if osp.exists(filename):
                order_file = osp.join(json_dir, 'order_file.json')
                if not osp.exists(order_file):
                    order_file = None

                img = jsk_arc2017_common.visualize_item_location(
                    filename, order_file)
                imgmsg = bridge.cv2_to_imgmsg(img, encoding='rgb8')
                imgmsg.header.stamp = rospy.Time.now()
                self.pub_item_location.publish(imgmsg)
            else:
                rospy.logwarn_throttle(10, '%s does not exists yet' % filename)
        if 'order' in self._types:
            filename = osp.join(json_dir, 'order_file.json')
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
