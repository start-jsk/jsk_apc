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
from jsk_rqt_plugins.srv import YesNo
from sensor_msgs.msg import Image
import os
import threading

# 1. TODO: make the class to accept service that is more general

class CollectSIBData(ConnectionBasedTransport):
    """
    """
    def __init__(self):
        self.shelf = {}
        self.mask_img = None
        self.dist_img = None
        self._target_bin = None
        self.camera_model = cameramodels.PinholeCameraModel()
        self.depth_img = None

        self.lock = threading.Lock()

        ConnectionBasedTransport.__init__(self)

        self.bridge = CvBridge()

    def subscribe(self):
        self.bin_info_sub = rospy.Subscriber(
            '~input/bin_info_array', BinInfoArray, self._topic_cb)
        self.synctopic_sub = rospy.Subscriber(
            '~input', SegmentationInBinSync, self._callback)
        self.depth_sub = rospy.Subscriber(
            '~input/depth', Image, self._depth_cb)

    def unsubscribe(self):
        self.sub.unregister()

    def set_layout_name(self, json):
        self.layout_name = json.split('/')[-1][:-5]

    def _topic_cb(self, bin_info_arr_msg):
        rospy.loginfo('get bin_info')
        self.set_layout_name(rospy.get_param('/set_bin_param/json'))
        self.bin_info_dict = self.bin_info_array_to_dict(bin_info_arr_msg)

    def _depth_cb(self, depth_msg):
        self.lock.acquire()
        rospy.loginfo('get depth')
        self.depth_img = self.bridge.imgmsg_to_cv2(
            depth_msg, "passthrough")
        self.lock.release()

    def _callback(self, sync_msg):
        rospy.loginfo('started')
        self.lock.acquire()

        # wait until yn_botton is pressed
        rospy.wait_for_service('~yes_no')
        try:
            client = rospy.ServiceProxy('~yes_no', YesNo)
        except rospy.ServiceException, e:
            print 'service {}'.format(e)
        yn = client.call()
        if not yn.yes:
            return

        if self.depth_img == None:
            rospy.loginfo('depth image not stored yet')
            return

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
        self.target_object = self.bin_info_dict[self.target_bin_name].target
        self.target_bin_info = self.bin_info_dict[self.target_bin_name]

        data, save_path = self.get_save_info()

        # save image
        self.save_images(save_path)

        with open(save_path + '.pkl', 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        self.lock.release()
        rospy.loginfo('saved to {}'.format(save_path))

    def save_images(self, save_path):
        """Save images

        1. Color: 3 channel uint8, BGR
        2. Mask Image: bool->uint8, 0 or 255
        3. Depth: uint8 (mm)
        4. Dist2Shelf: uint8 (mm)
        5. Height3D_image: uint8 (mm)
        """
        mask_img = self.mask_img.astype(np.uint8) * 255

        cv2.imwrite(save_path + '.jpg', self.color_img)
        cv2.imwrite(save_path + '.pbm', mask_img)

        

    def get_save_info(self):
        """prepare for saving
        """
        data = {}
        data['target_object'] = self.target_object
        data['objects'] = self.target_bin_info.objects
        data['dist2shelf_image'] = self.dist_img.astype(np.float16)
        data['height3D_image'] = self.height_img.astype(np.float16)


        # data['height2D_image'] = np.zeros_like(self.height_img)

        #data['mask_img'] = self.mask_img
        data['depth_image'] = self.depth_img.astype(np.float16)
        
        dir_path = rospy.get_param('~save_dir')
        print dir_path
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        time = strftime('%Y%m%d%H', gmtime())
        save_path = (dir_path + '/' + self.layout_name + '_' + time + '_bin_' +
                     self.target_bin_name)
        return data, save_path

    def bin_info_array_to_dict(self, bin_info_array):
        bin_info_dict = {}
        for bin_ in bin_info_array.array:
            bin_info_dict[bin_.name] = bin_
        return bin_info_dict


if __name__ == '__main__':
    rospy.init_node('save_data')
    # wait until gui button is pressed

    seg = CollectSIBData()
    rospy.spin()
