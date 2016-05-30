#!/usr/bin/env python

from sensor_msgs.msg import Image
import message_filters
import rospy
from cv_bridge import CvBridge
import numpy as np

from jsk_topic_tools import ConnectionBasedTransport

from skimage.color import gray2rgb


class CombineImages(ConnectionBasedTransport):
    """Publishes an RGB img where the color img is overlaid with the mono img

    Attributes:
        alpha (float): between [0, 1], Opacity of the mono img.
        image_alpha (float): between [0, 1].  Opacity of the color img.
    """
    def __init__(self):
        super(CombineImages, self).__init__()

        self.bridge = CvBridge()
        self.image_alpha = rospy.get_param('~image_alpha', 0.7)
        self.alpha = rospy.get_param('~alpha', 0.3)

        self.img_pub = self.advertise('~output', Image, queue_size=2)

    def subscribe(self):
        self.sub_color = message_filters.Subscriber('~color', Image)
        self.sub_mono = message_filters.Subscriber('~mono', Image)
        self.sync = message_filters.ApproximateTimeSynchronizer(
                [self.sub_color, self.sub_mono],
                queue_size=10, slop=0.3)
        self.sync.registerCallback(self._apply)

    def unsubscribe(self):
        self.sync.unregister()

    def _apply(self, color_img_msg, mono_img_msg):
        color_img = self.bridge.imgmsg_to_cv2(color_img_msg, "passthrough")
        mono_img = self.bridge.imgmsg_to_cv2(mono_img_msg, "passthrough")

        # normalize mono image
        mono_img = (255 * mono_img.astype(np.int32)) / np.max(mono_img)

        # reference skimage.color func: _label2rgb_overlay
        image = gray2rgb(mono_img) * self.image_alpha + (1 - self.image_alpha)
        result_img = color_img * self.alpha + image * (1 - self.alpha)

        # publish
        posterior_msg = self.bridge.cv2_to_imgmsg(
            result_img.astype(np.uint8), encoding='bgr8')
        self.img_pub.publish(posterior_msg)


if __name__ == '__main__':
    rospy.init_node('combine_images')
    combine_image = CombineImages()
    rospy.spin()
