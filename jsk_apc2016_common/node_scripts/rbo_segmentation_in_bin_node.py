#!/usr/bin/env python

from jsk_apc2016_common.msg import BinInfoArray
from jsk_topic_tools import ConnectionBasedTransport
from sensor_msgs.msg import Image
import rospy
from cv_bridge import CvBridge, CvBridgeError
import cv2
from image_geometry import cameramodels
import message_filters
import numpy as np
from jsk_apc2016_common.rbo_segmentation.apc_data import APCSample
import pickle


class RBOSegmentationInBinNode(ConnectionBasedTransport):
    def __init__(self):
        self.shelf = {}
        self.mask_img = None
        self.dist_img = None
        self._target_bin = None
        self.camera_model = cameramodels.PinholeCameraModel()

        ConnectionBasedTransport.__init__(self)

        self.load_trained(rospy.get_param('~trained_pkl_path'))

        bin_info_array_msg = rospy.wait_for_message(
                "~input/bin_info_array", BinInfoArray, timeout=50)
        self.bin_info_dict = self.bin_info_array_to_dict(bin_info_array_msg)

        self.bridge = CvBridge()
        self.img_pub = self.advertise('~target_mask', Image, queue_size=100)

    def subscribe(self):
        self.dist_img_sub = message_filters.Subscriber('~input/dist', Image)
        self.height_img_sub = message_filters.Subscriber(
                '~input/height', Image)
        self.img_sub = message_filters.Subscriber('~input/image', Image)
        self.mask_sub = message_filters.Subscriber('~input/mask', Image)

        self.sync = message_filters.ApproximateTimeSynchronizer(
                [self.dist_img_sub, self.height_img_sub, self.img_sub, self.mask_sub],
                queue_size=50,
                slop=1.0)
        self.sync.registerCallback(self._callback)

    def unsubscribe(self):
        self.sub.unregister()

    def _callback(self, dist_msg, height_msg, img_msg, mask_msg):
        rospy.loginfo('started')
        self.height = dist_msg.height
        self.width = dist_msg.width
        try:
            self.mask_img = self.bridge.imgmsg_to_cv2(mask_msg, "passthrough")
            self.mask_img = self.mask_img.astype('bool')
        except CvBridgeError as e:
            print "error"
        self.dist_img = self.bridge.imgmsg_to_cv2(dist_msg, "passthrough")
        self.height_img = self.bridge.imgmsg_to_cv2(height_msg, "passthrough")
        self.height_img = self.height_img.astype(np.float)/255.0

        try:
            img_color = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
            self.img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)
        except CvBridgeError as e:
            rospy.logerr('{}'.format(e))

        if rospy.get_param('~target_bin_name') not in 'abcdefghijkl':
            return

        self.target_bin_name = rospy.get_param('~target_bin_name')
        self.target_object = self.bin_info_dict[self.target_bin_name].target
        self.target_bin_info = self.bin_info_dict[self.target_bin_name]

        self.set_apc_sample()
        # generate a binary image
        self.segmentation()
        try:
            predict_msg = self.bridge.cv2_to_imgmsg(
                    self.predicted_segment, encoding="mono8")
            predict_msg.header = img_msg.header

            # This is a patch.
            # Later in the process of SIB, you need to synchronize
            # the current pointclouds and topics produced using this
            # topic. The synchronization fails if the timestamp is not
            # updated to the current time.
            predict_msg.header.stamp = rospy.Time.now()
            self.img_pub.publish(predict_msg)
        except CvBridgeError as e:
            rospy.logerr('{}'.format(e))
        rospy.loginfo('ended')

    def bin_info_array_to_dict(self, bin_info_array):
        bin_info_dict = {}
        for bin_ in bin_info_array.array:
            bin_info_dict[bin_.name] = bin_
        return bin_info_dict

    def set_apc_sample(self):
        assert self.target_object is not None
        # TODO: work on define_later later
        define_later = np.zeros((
            self.height, self.width))
        data = {}
        data['objects'] = self.target_bin_info.objects
        data['dist2shelf_image'] = self.dist_img
        data['depth_image'] = define_later
        data['has3D_image'] = define_later
        data['height3D_image'] = self.height_img
        data['height2D_image'] = define_later

        self.apc_sample = APCSample(
                image_input=self.img_color,
                bin_mask_input=self.mask_img,
                data_input=data,
                labeled=False,
                infer_shelf_mask=False,
                pickle_mask=False)

    def segmentation(self):
        zoomed_predicted_segment = self.trained_segmenter.predict(
                apc_sample=self.apc_sample,
                desired_object=self.target_object)
        self.predicted_segment = self.apc_sample.unzoom_segment(
                zoomed_predicted_segment)

        # Masked region needs to contain value 255.
        self.predicted_segment = 255 * self.predicted_segment.astype('uint8')

    def load_trained(self, path):
        with open(path, 'rb') as f:
            self.trained_segmenter = pickle.load(f)


if __name__ == '__main__':
    rospy.init_node('segmentation_in_bin')
    seg = RBOSegmentationInBinNode()
    rospy.spin()
