#!/usr/bin/env python

from jsk_apc2016_common.msg import BinInfoArray, SegmentationInBinSync
import rospy
from cv_bridge import CvBridge
import cv2
import numpy as np
import pickle
from time import gmtime, strftime
from sensor_msgs.msg import Image
import os
import threading


class CollectSIBData(object):
    """
    """

    def __init__(self):
        self.mask_img = None
        self.dist_img = None
        self.depth_img = None
        self.bin_info_dict = {}

        self.lock = threading.Lock()

        self.bridge = CvBridge()

        self.debug_color_pub = rospy.Publisher(
            '~debug_color', Image, queue_size=1)
        self.debug_depth_pub = rospy.Publisher(
            '~debug_depth', Image, queue_size=1)

        self.bin_info_sub = rospy.Subscriber(
            '~input/bin_info_array', BinInfoArray, self.topic_cb)

        self.sync_sub = rospy.Subscriber(
            '~input', SegmentationInBinSync, self.save_data_callback)
        self.depth_sub = rospy.Subscriber(
            '~input/depth', Image, self.depth_cb)

    def topic_cb(self, bin_info_arr_msg):
        json_path = rospy.get_param('/publish_bin_info/json')
        self.layout_name = json_path.split('/')[-1][:-5]
        self.bin_info_dict = self.bin_info_array_to_dict(bin_info_arr_msg)

    def bin_info_array_to_dict(self, bin_info_array):
        bin_info_dict = {}
        for bin_ in bin_info_array.array:
            bin_info_dict[bin_.name] = bin_
        return bin_info_dict

    def depth_cb(self, depth_msg):
        self.lock.acquire()
        self.depth_msg = depth_msg
        self.depth_img = self.bridge.imgmsg_to_cv2(
            depth_msg, "passthrough")
        self.lock.release()

    def save_data_callback(self, sync_msg):
        if self.bin_info_dict == {}:
            rospy.loginfo('bin_info_dict is not stored yet')
            return

        if rospy.get_param('~ready_to_save') is True:
            self.lock.acquire()

            dist_msg = sync_msg.dist_msg
            height_msg = sync_msg.height_msg
            color_msg = sync_msg.color_msg
            mask_msg = sync_msg.mask_msg

            self.mask_img = self.bridge.imgmsg_to_cv2(
                mask_msg, "passthrough").astype(np.bool)
            self.dist_img = self.bridge.imgmsg_to_cv2(
                dist_msg, "passthrough")
            self.height_img = self.bridge.imgmsg_to_cv2(
                height_msg, "passthrough")

            self.color_img = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")
            # self.color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2HSV)

            self.target_bin_name = rospy.get_param('~target_bin_name')
            self.target_object =\
                self.bin_info_dict[self.target_bin_name].target

            self.target_bin_info = self.bin_info_dict[self.target_bin_name]

            # debug message
            self.debug_color_pub.publish(color_msg)
            self.debug_depth_pub.publish(self.depth_msg)

            self.save_data()

            self.lock.release()
        rospy.set_param('~ready_to_save', False)

    def save_data(self):
        """Save data

        1. Color: 3 channel uint8, BGR
        2. Mask Image: bool->uint8, 0 or 255
        3. Depth: uint8 (mm)
        4. Dist2Shelf: uint8 (mm)
        5. Height3D_image: uint8 (mm)
        """
        # path of files to save
        dir_path = os.path.expanduser(rospy.get_param('~save_dir'))
        print(dir_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        time = strftime('%Y%m%d%H%M%S', gmtime())
        save_path = (dir_path + '/' + self.layout_name + '_' + time + '_bin_' +
                     self.target_bin_name)

        # save data images
        mask_img = self.mask_img.astype(np.uint8) * 255
        cv2.imwrite(save_path + '.jpg', self.color_img)
        cv2.imwrite(save_path + '.pbm', mask_img)

        # save data pkl
        data = {}
        data['target_object'] = self.target_object
        data['objects'] = self.target_bin_info.objects
        data['dist2shelf_image'] = self.dist_img.astype(np.float16)
        data['height3D_image'] = self.height_img.astype(np.float16)
        data['depth_image'] = self.depth_img.astype(np.float16)
        with open(save_path + '.pkl', 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

        # log mesage
        rospy.loginfo('saved: {}\n target_object: {}'.format(
            save_path, self.target_object))


if __name__ == '__main__':
    rospy.init_node('save_data')
    # wait until gui button is pressed

    seg = CollectSIBData()
    rospy.spin()
