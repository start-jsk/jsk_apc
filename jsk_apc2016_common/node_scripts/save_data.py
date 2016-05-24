#!/usr/bin/env python

from jsk_apc2016_common.msg import BinInfoArray, SegmentationInBinSync
from jsk_topic_tools import ConnectionBasedTransport
import rospy
from cv_bridge import CvBridge
import cv2
from image_geometry import cameramodels
import numpy as np
import pickle
from time import gmtime, strftime
import rospkg
import matplotlib.pyplot as plt



class SaveData(ConnectionBasedTransport):
    def __init__(self):
        self.shelf = {}
        self.mask_img = None
        self.dist_img = None
        self._target_bin = None
        self.camera_model = cameramodels.PinholeCameraModel()

        ConnectionBasedTransport.__init__(self)

        self.layout_name = rospy.get_param('~layout_name')

        bin_info_array_msg = rospy.wait_for_message(
                "~input/bin_info_array", BinInfoArray, timeout=50)
        self.bin_info_dict = self.bin_info_array_to_dict(bin_info_array_msg)

        self.bridge = CvBridge()

    def subscribe(self):
        self.subscriber = rospy.Subscriber(
            '~input', SegmentationInBinSync, self._callback)

    def unsubscribe(self):
        self.sub.unregister()

    def _callback(self, sync_msg):
        rospy.loginfo('started')

        dist_msg = sync_msg.dist_msg
        height_msg = sync_msg.height_msg
        color_msg = sync_msg.color_msg
        mask_msg = sync_msg.mask_msg

        self.mask_img = self.bridge.imgmsg_to_cv2(
            mask_msg, "passthrough").astype('bool')
        self.dist_img = self.bridge.imgmsg_to_cv2(dist_msg, "passthrough")
        self.height_img = self.bridge.imgmsg_to_cv2(
            height_msg, "passthrough").astype(np.float) / 255.0

        color_img = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")
        self.color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2HSV)

        self.target_bin_name = rospy.get_param('~target_bin_name')
        self.target_object = self.bin_info_dict[self.target_bin_name].target
        self.target_bin_info = self.bin_info_dict[self.target_bin_name]

        data, save_path = self.get_save_info()


        # save image


        with open(save_path, 'wb') as f:
            pickle.dump(data, f)
        rospy.loginfo('saved to {}'.format(save_path))

    def get_save_info(self):
        data = {}
        data['target_object'] = self.target_object
        data['objects'] = self.target_bin_info.objects
        data['dist_image'] = self.dist_img
        data['heigh_image'] = self.height_img
        data['color_img'] = self.color_img
        data['mask_img'] = self.mask_img

        time = strftime('%Y%m%d%H', gmtime())
        rospack = rospkg.RosPack()
        dir_path = rospack.get_path('jsk_apc2016_common') + '/data/tokyo_run/'
        save_path = (dir_path + self.layout_name + '_' + time + '_bin_' +
                     self.target_bin_name + '.pkl')
        return data, save_path

    def bin_info_array_to_dict(self, bin_info_array):
        bin_info_dict = {}
        for bin_ in bin_info_array.array:
            bin_info_dict[bin_.name] = bin_
        return bin_info_dict


if __name__ == '__main__':
    rospy.init_node('save_data')
    seg = SaveData()
    rospy.spin()
