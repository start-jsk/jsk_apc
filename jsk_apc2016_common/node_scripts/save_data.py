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


class SaveData(ConnectionBasedTransport):
    def __init__(self):
        self.shelf = {}
        self.mask_img = None
        self.dist_img = None
        self._target_bin = None
        self.camera_model = cameramodels.PinholeCameraModel()
        self.depth_img = None

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
        # TODO: add Lock
        rospy.loginfo('get_bin_info')
        self.set_layout_name(rospy.get_param('~json'))
        self.try_dir = rospy.get_param('~try_dir')
        self.bin_info_dict = self.bin_info_array_to_dict(bin_info_arr_msg)

    def _depth_cb(self, depth_msg):
        self.depth_img = self.bridge.imgmsg_to_cv2(depth_msg, "passthrough")

    def _callback(self, sync_msg):
        rospy.loginfo('started')

        # wait until yn_botton is pressed
        rospy.wait_for_service('save_data/rqt_yn_btn')
        try:
            client = rospy.ServiceProxy('save_data/rqt_yn_btn', YesNo)
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
            mask_msg, "passthrough").astype('bool')
        self.dist_img = self.bridge.imgmsg_to_cv2(dist_msg, "passthrough")
        self.height_img = self.bridge.imgmsg_to_cv2(
            height_msg, "passthrough").astype(np.float) / 255.0

        self.color_img = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")
        # self.color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2HSV)

        self.target_bin_name = rospy.get_param('~target_bin_name')
        self.target_object = self.bin_info_dict[self.target_bin_name].target
        self.target_bin_info = self.bin_info_dict[self.target_bin_name]

        data, pkl_path, img_path = self.get_save_info()

        # save image
        # self.color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2HSV)
        # plt.imsave(img_path, data['color_img'])
        cv2.imwrite(img_path, self.color_img)

        with open(pkl_path, 'wb') as f:
            pickle.dump(data, f)
        rospy.loginfo('saved to {}'.format(pkl_path))

    def get_save_info(self):
        data = {}
        data['target_object'] = self.target_object
        data['objects'] = self.target_bin_info.objects
        data['dist2shelf'] = self.dist_img
        data['height3D'] = self.height_img
        data['color'] = self.color_img
        data['mask_img'] = self.mask_img
        data['depth'] = self.depth_img

        time = strftime('%Y%m%d%H', gmtime())
        rospack = rospkg.RosPack()
        dir_path = rospack.get_path('jsk_apc2016_common') + \
            '/data/tokyo_run/' + str(self.try_dir) + '/'
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        save_path = (dir_path + self.layout_name + '_' + time + '_bin_' +
                     self.target_bin_name)
        pkl_path = save_path + '.pkl'
        img_path = save_path + '.jpg'
        return data, pkl_path, img_path

    def bin_info_array_to_dict(self, bin_info_array):
        bin_info_dict = {}
        for bin_ in bin_info_array.array:
            bin_info_dict[bin_.name] = bin_
        return bin_info_dict


if __name__ == '__main__':
    rospy.init_node('save_data')
    # wait until gui button is pressed

    seg = SaveData()
    rospy.spin()
