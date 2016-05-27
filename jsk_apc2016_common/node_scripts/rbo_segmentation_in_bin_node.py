#!/usr/bin/env python

from jsk_apc2016_common.msg import BinInfoArray, SegmentationInBinSync
from jsk_topic_tools import ConnectionBasedTransport
from sensor_msgs.msg import Image
import rospy
from cv_bridge import CvBridge, CvBridgeError
import cv2
from image_geometry import cameramodels
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
        self.posterior_pub = self.advertise('~posterior', Image, queue_size=20)
        self.masked_input_img_pub = self.advertise('~masked_input', Image, queue_size=20)

    def subscribe(self):
        self.subscriber = rospy.Subscriber('~input', SegmentationInBinSync, self._callback)

    def unsubscribe(self):
        self.sub.unregister()

    def _callback(self, sync_msg):
        rospy.loginfo('started')

        dist_msg = sync_msg.dist_msg
        height_msg = sync_msg.height_msg
        color_msg = sync_msg.color_msg
        mask_msg = sync_msg.mask_msg

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
            color_img = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")
            self.color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2HSV)
        except CvBridgeError as e:
            rospy.logerr('{}'.format(e))

        self.exist3d_img = (self.dist_img != 0)

        target_bin_name = rospy.get_param('~target_bin_name')
        if target_bin_name not in 'abcdefghijkl':
            rospy.logwarn('wrong target_bin_name')
            return
        if target_bin_name == '':
            rospy.logwarn('target_bin_name is empty string')
            return

        self.target_bin_name = target_bin_name
        self.target_object = self.bin_info_dict[self.target_bin_name].target
        self.target_bin_info = self.bin_info_dict[self.target_bin_name]

        self.set_apc_sample()
        # generate a binary image
        self.segmentation()
        if np.all(self.predicted_segment[self.exist3d_img] == 0):
            rospy.logwarn('Output of RBO does not contain any point clouds.')
            return
        try:
            predict_msg = self.bridge.cv2_to_imgmsg(
                    self.predicted_segment, encoding="mono8")
            predict_msg.header = color_msg.header
            self.img_pub.publish(predict_msg)
        except CvBridgeError as e:
            rospy.logerr('{}'.format(e))

        # for visualization
        masked_input_img = cv2.cvtColor(self.apc_sample.image, cv2.COLOR_HSV2BGR)
        masked_input_msg = self.bridge.cv2_to_imgmsg(
                masked_input_img)
        masked_input_msg.header = color_msg.header
        self.masked_input_img_pub.publish(masked_input_msg)

        try:
            posterior_img = self.trained_segmenter.\
                posterior_images_smooth[self.target_object]
            posterior_msg = self.bridge.cv2_to_imgmsg(
                posterior_img.astype(np.float32))
            posterior_msg.header = color_msg.header
            self.posterior_pub.publish(posterior_msg)
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
                image_input=self.color_img,
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
