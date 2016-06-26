#!/usr/bin/env python

from jsk_apc2016_common.msg import BinInfoArray, SegmentationInBinSync
import jsk_apc2016_common
from jsk_topic_tools import ConnectionBasedTransport
from sensor_msgs.msg import Image
import rospy

from fcn.models import FCN32s
import chainer.serializers as S
import numpy as np
import cv2

from cv_bridge import CvBridge
from jsk_topic_tools import log_utils
from chainer import Variable
from chainer import cuda
import os
import time
import message_filters


class FCNSegmentationInBinNode(ConnectionBasedTransport):
    mean_bgr = np.array((104.00698793, 116.66876762, 122.67891434))

    def __init__(self):
        self.mask_img = None
        self.dist_img = None
        self.target_bin = None
        self.bin_info_dict = {}
        self.bridge = CvBridge()
        ConnectionBasedTransport.__init__(self)

        self.gpu = rospy.get_param('~gpu', 0)
        self.load_model()

        self.objects = ['background'] +\
            [object_data['name'] for
             object_data in jsk_apc2016_common.get_object_data()]

        self.target_mask_pub = self.advertise(
            '~target_mask', Image, queue_size=3)
        self.masked_input_img_pub = self.advertise(
            '~masked_input', Image, queue_size=3)
        self.debug_output_pub = self.advertise(
            '~debug_output', Image, queue_size=3)

    def subscribe(self):
        self.bin_info_arr_sub = rospy.Subscriber(
            '~input/bin_info_array', BinInfoArray, self.bin_info_callback)
        self.sub = rospy.Subscriber(
            '~input', SegmentationInBinSync, self.callback)

    def unsubscribe(self):
        self.bin_info_arr_sub.unregister()
        self.sub.unregister()

    def load_model(self):
        chainermodel = rospy.get_param('~chainermodel', None)
        self.model = FCN32s(n_class=40)
        S.load_hdf5(chainermodel, self.model)
        if self.gpu != -1:
            self.model.to_gpu(self.gpu)

    def bin_info_callback(self, bin_info_array_msg):
        for bin_ in bin_info_array_msg.array:
            self.bin_info_dict[bin_.name] = bin_

    def callback(self, sync_msg):
        log_utils.loginfo_throttle(10, 'started')

        if self.bin_info_dict == {}:
            return
        if not hasattr(self, 'model'):
            return

        self.sync_msg_to_img(sync_msg)

        self.process_target_bin(rospy.get_param('~target_bin_name'))

        self.segmentation_publish()

        log_utils.loginfo_throttle(10, 'ended')

    def sync_msg_to_img(self, sync_msg):
        color_msg = sync_msg.color_msg
        height_msg = sync_msg.height_msg
        dist_msg = sync_msg.dist_msg
        mask_msg = sync_msg.mask_msg

        self.header = dist_msg.header
        self.height = dist_msg.height
        self.width = dist_msg.width

        # convert imgmsg to img
        self.mask_img = self.bridge.imgmsg_to_cv2(mask_msg, 'passthrough')
        self.mask_img = self.mask_img.astype('bool')
        self.color_img = self.bridge.imgmsg_to_cv2(color_msg, 'bgr8')

        # masked_input
        self.masked_input_img = self.color_img.copy()
        where = np.argwhere(self.mask_img)
        roi = where.min(0), where.max(0) + 1
        self.masked_input_img = self.masked_input_img[roi[0][0]:roi[1][0],
                                                      roi[1][0]:roi[1][1]]

        self.masked_input_msg = self.bridge.cv2_to_imgmsg(
            self.masked_input_img, 'bgr8')
        self.masked_input_img_pub.publish(self.masked_input_msg)

        # depth related
        self.dist_img = self.bridge.imgmsg_to_cv2(dist_msg, 'passthrough')
        self.height_img = self.bridge.imgmsg_to_cv2(height_msg, 'passthrough')
        self.height_img = self.height_img.astype(np.float) / 255.0
        self.exist3d_img = (self.dist_img != 0)

    def process_target_bin(self, target_bin_name):
        if target_bin_name not in 'abcdefghijkl':
            rospy.logwarn('wrong target_bin_name')
            return
        if target_bin_name == '':
            rospy.logwarn('target_bin_name is empty string')
            return
        self.target_bin_name = target_bin_name
        self.target_object = self.bin_info_dict[self.target_bin_name].target
        self.target_bin_info = self.bin_info_dict[self.target_bin_name]

    def segmentation_publish(self):
        try:
            self._segmentation()
            target_mask_msg = self.bridge.cv2_to_imgmsg(
                self.target_mask, encoding='mono8')
            target_mask_msg.header = self.header
            target_mask_msg.header.stamp = rospy.Time.now()
            if np.any(self.target_mask[self.exist3d_img] != 0):
                self.target_mask_pub.publish(target_mask_msg)
            else:
                rospy.logwarn(
                    'Output of RBO does not contain any point clouds.')
            # publish images which contain object probabilities
            # self.publish_predicted_results()
        except KeyError, e:
            rospy.loginfo(repr(e))

        # debug output
        self.debug_output_pub.publish(target_mask_msg)

    def _segmentation(self):
        """Predict and store the result in self.predicted_segment using RGB
        """
        datum = self.color_img - self.mean_bgr
        datum = datum.transpose((2, 0, 1))

        x_data = np.array([datum], dtype=np.float32)
        if self.gpu != -1:
            x_data = cuda.to_gpu(x_data, device=self.gpu)
        x = Variable(x_data, volatile=False)
        self.model(x)
        pred_datum = self.model.score.data[0]  # shape (40, 480, 640)

        pred_datum = cuda.to_cpu(pred_datum)

        candidate_labels =\
            np.array([self.objects.index(obj_name) for
                      obj_name in self.target_bin_info.objects + ['background']])

        # labels for all objects in the bin
        self.label_pred = pred_datum[candidate_labels].argmax(axis=0)
        for idx, label_val in enumerate(candidate_labels):
            self.label_pred[self.label_pred == idx] = label_val
        self.label_pred[self.mask_img == 0] = 0

        # for the target object
        self.target_mask =\
            self.label_pred == self.objects.index(self.target_object)
        self.target_mask = 255 * self.target_mask.astype(np.uint8)
        self.target_mask = self.extract_largest_connected_component(
            self.target_mask)

    def extract_largest_connected_component(self, mask):
        contours, hierarchy = cv2.findContours(
            mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contour_areas = [cv2.contourArea(cnt) for cnt in contours]
        if len(contour_areas) == 0:
            return np.zeros_like(mask)
        max_contour = contours[np.argmax(contour_areas)]
        extracted_mask = np.zeros_like(mask)
        cv2.drawContours(extracted_mask, [max_contour], -1, 255, -1)
        return extracted_mask


if __name__ == '__main__':
    rospy.init_node('segmentation_in_bin')
    seg = FCNSegmentationInBinNode()
    rospy.spin()
