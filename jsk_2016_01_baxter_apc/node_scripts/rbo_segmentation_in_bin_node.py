#!/usr/bin/env python

from jsk_apc2016_common.segmentation_in_bin.rbo_preprocessing \
        import get_mask_img, get_spatial_img
from jsk_apc2016_common.segmentation_in_bin.rbo_segmentation_in_bin\
        import RBOSegmentationInBin
from jsk_apc2016_common.msg import BinInfoArray, SegmentationInBinSync
from jsk_topic_tools import ConnectionBasedTransport
from sensor_msgs.msg import Image
import rospy
import tf2_ros
from cv_bridge import CvBridge, CvBridgeError
import cv2


class RBOSegmentationInBinNode(ConnectionBasedTransport, RBOSegmentationInBin):
    def __init__(self):
        RBOSegmentationInBin.__init__(self)

        ConnectionBasedTransport.__init__(self)

        self.load_trained(rospy.get_param('~trained_pkl_path'))

        bin_info_array_msg = rospy.wait_for_message(
                "~input/bin_info_array", BinInfoArray, timeout=50)
        self.from_bin_info_array(bin_info_array_msg)

        self.buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.buffer)

        self.bridge = CvBridge()
        self.img_pub = self.advertise('~target_mask', Image, queue_size=100)

    def subscribe(self):
        self.sub = rospy.Subscriber('~input', SegmentationInBinSync, self._callback)

    def unsubscribe(self):
        self.sub .unregister()

    def _callback(self, sync):
        rospy.loginfo('started')

        if rospy.get_param('~target_bin_name') not in 'abcdefghijkl':
            return
        img_msg = sync.image_color
        cloud = sync.points

        self.target_bin_name = rospy.get_param('~target_bin_name')
        self.target_object = self.shelf[self.target_bin_name].target
        self.target_bin = self.shelf[self.target_bin_name]

        # mask image
        self.camera_info = sync.cam_info
        self.camera_model.fromCameraInfo(self.camera_info)
        try:
            img_color = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
            img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)
        except CvBridgeError as e:
            rospy.logerr('{}'.format(e))
        self.img_color = img_color

        # get transform
        camera2bb_base = self.buffer.lookup_transform(
                target_frame=self.camera_info.header.frame_id,
                source_frame=self.target_bin.bbox.header.frame_id,
                time=rospy.Time(0),
                timeout=rospy.Duration(10.0))

        bb_base2camera = self.buffer.lookup_transform(
                target_frame=self.target_bin.bbox.header.frame_id,
                source_frame=cloud.header.frame_id,
                time=rospy.Time(0),
                timeout=rospy.Duration(10.0))

        # get mask_image
        self.mask_img = get_mask_img(
                camera2bb_base, self.target_bin, self.camera_model)

        # dist image
        self.dist_img, self.height_img = get_spatial_img(
                bb_base2camera, cloud, self.target_bin)

        self.set_apc_sample()
        # generate a binary image
        self.segmentation()
        try:
            mask_msg = self.bridge.cv2_to_imgmsg(
                    self.predicted_segment, encoding="mono8")
            mask_msg.header = img_msg.header
            # This is a patch.
            # Later in the process of SIB, you need to synchronize
            # the current pointclouds and topics produced using this
            # topic. The synchronization fails if the timestamp is not
            # updated to the current time.
            mask_msg.header.stamp = rospy.Time.now()
            self.img_pub.publish(mask_msg)
        except CvBridgeError as e:
            rospy.logerr('{}'.format(e))
        rospy.loginfo('ended')


if __name__ == '__main__':
    rospy.init_node('segmentation_in_bin')
    seg = RBOSegmentationInBinNode()
    rospy.spin()
