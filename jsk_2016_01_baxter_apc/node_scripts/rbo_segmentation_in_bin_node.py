#!/usr/bin/env python

from jsk_apc2016_common.segmentation_in_bin.rbo_preprocessing \
        import get_mask_img, get_spatial_img
from jsk_apc2016_common.segmentation_in_bin.rbo_segmentation_in_bin\
        import RBOSegmentationInBin
from jsk_apc2016_common.msg import BinInfoArray
from jsk_topic_tools import ConnectionBasedTransport
import rospy
import message_filters
from sensor_msgs.msg import Image, CameraInfo
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import TransformStamped
from cv_bridge import CvBridge, CvBridgeError


class RBOSegmentationInBinNode(ConnectionBasedTransport, RBOSegmentationInBin):
    def __init__(self):
        RBOSegmentationInBin.__init__(self,
                trained_pkl_path=rospy.get_param('~trained_pkl_path'),
                target_bin_name=rospy.get_param('~target_bin_name'))
        ConnectionBasedTransport.__init__(self)
        bin_info_array_msg = rospy.wait_for_message(
                "~input/bin_info_array", BinInfoArray, timeout=50)
        self.from_bin_info_array(bin_info_array_msg)
        self.bridge = CvBridge()
        self.img_pub = self.advertise('~target_mask', Image, queue_size=100)

    def subscribe(self):
        self.pc_sub = message_filters.Subscriber('~input', PointCloud2)
        self.cam_info_sub = message_filters.Subscriber(
                '~input/info', CameraInfo)
        self.img_sub = message_filters.Subscriber('~input/image', Image)
        self.subs = [self.pc_sub, self.img_sub, self.cam_info_sub]
        self.sync = message_filters.ApproximateTimeSynchronizer(
                self.subs,
                queue_size=100,
                slop=0.5)
        self.sync.registerCallback(self._callback)

    def unsubscribe(self):
        for sub in self.subs:
            sub.unregister()

    def _callback(self, cloud, img_msg, camera_info):
        rospy.loginfo('started')
        self.target_bin_name = rospy.get_param('~target_bin_name')
        camera2bb_base = rospy.wait_for_message('~input/camera2bb_base', TransformStamped)
        bb_base2camera = rospy.wait_for_message('~input/bb_base2camera', TransformStamped)

        # mask image
        self.camera_info = camera_info
        try:
            img_color = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr('{}'.format(e))
        self.img_color = img_color

        # get mask_image
        self.mask_img = get_mask_img(
                camera2bb_base, self.target_bin, self.camera_model)

        self.dist_img, self.height_img = get_spatial_img(
                bb_base2camera, cloud, self.target_bin)

        self.set_apc_sample()
        # generate a binary image
        self.segmentation()
        try:
            mask_msg = self.bridge.cv2_to_imgmsg(self.predicted_segment, encoding="passthrough")
            mask_msg.header = img_msg.header
            self.img_pub.publish(mask_msg)
        except CvBridgeError as e:
            rospy.logerr('{}'.format(e))

        rospy.loginfo('ended')


if __name__ == '__main__':
    rospy.init_node('segmentation_in_bin')
    seg = RBOSegmentationInBinNode()
    rospy.spin()
