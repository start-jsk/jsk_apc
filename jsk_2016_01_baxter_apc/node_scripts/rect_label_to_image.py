#!/usr/bin/env python

import cv2
import cv_bridge
import rospy
import message_filters
from sensor_msgs.msg import Image
from jsk_topic_tools import ConnectionBasedTransport, jsk_loginfo
from jsk_recognition_msgs.msg import RectArray, ClassificationResult
from geometry_msgs.msg import Point


class RectLabelToImage(ConnectionBasedTransport):
    def __init__(self):
        super(RectLabelToImage, self).__init__()
        self.pub = self.advertise("~output", Image, queue_size=10)

    def subscribe(self):
        self.sub_rect = message_filters.Subscriber('~input', RectArray)
        self.sub_img = message_filters.Subscriber('~input/image', Image)
        self.sub_class = message_filters.Subscriber('~input/class', ClassificationResult)
        # warn_no_remap('~input','~input/image','~input/class')
        use_async = rospy.get_param('~approximate_sync', False)
        queue_size = rospy.get_param('~queue_size', 100)
        jsk_loginfo('~approximate_sync: {} queue_size: {}'.format(use_async, queue_size))
        if use_async:
            slop = rospy.get_param('~slop', 0.1)
            jsk_loginfo('~slop: {}'.format(slop))
            sync = message_filters.ApproximateTimeSynchronizer(
                    [self.sub_rect, self.sub_img, self.sub_class],
                    queue_size, slop)
        else:
            sync = message_filters.TimeSynchronizer(
                    [self.sub_rect, self.sub_img, self.sub_class],
                    queue_size)
        sync.registerCallback(self.convert)

    def unsubscribe(self):
        self.sub_rect.unregister()
        self.sub_img.unregister()
        self.sub_class.unregister()

    def convert(self, rect_msg, img_msg, class_msg):
        bridge = cv_bridge.CvBridge()
        img = bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
        out = img.copy()
        for rect, label_name, label_proba in zip(rect_msg.rects, class_msg.label_names, class_msg.label_proba):
            cv2.rectangle(out, (rect.x, rect.y), (rect.x+rect.width, rect.y+rect.height), (0, 0, 255))
            text = '{:s} {:.3f}'.format(label_name, label_proba)
            cv2.putText(out, text, (int(rect.x), int(rect.y)-2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
        out_msg = bridge.cv2_to_imgmsg(out, encoding='bgr8')
        out_msg.header = img_msg.header
        self.pub.publish(out_msg)


if __name__ == '__main__':
    rospy.init_node('rect_label_to_image')
    rect_label_to_image = RectLabelToImage()
    rospy.spin()
