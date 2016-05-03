#!/usr/bin/env python

from jsk_apc2016_common.segmentation_in_bin.rbo_preprocessing \
        import get_mask_img, get_spatial_img
from jsk_apc2016_common.segmentation_in_bin.rbo_segmentation_in_bin\
        import RBOSegmentationInBin
from jsk_apc2016_common.msg import BinInfoArray
import rospy
import tf2_ros
import message_filters
from sensor_msgs.msg import Image, CameraInfo
from sensor_msgs.msg import PointCloud2
from cv_bridge import CvBridge, CvBridgeError


class RBOSegmentationInBinNode(RBOSegmentationInBin):
    def __init__(self):
        super(self.__class__, self).__init__(
                trained_pkl_path=rospy.get_param('~trained_pkl_path'),
                target_bin_name=rospy.get_param('~target_bin_name'))

    def subscribe(self):
        bin_info_array_msg = rospy.wait_for_message(
                "~input/bin_info_array", BinInfoArray, timeout=50)
        self.from_bin_info_array(bin_info_array_msg)

        self.buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.buffer)
        self.tf_br = tf2_ros.TransformBroadcaster()

        self.bridge = CvBridge()

        self.img_pub = rospy.Publisher('~target_mask', Image, queue_size=100)

        self.pc_sub = message_filters.Subscriber('~input', PointCloud2)
        self.cam_info_sub = message_filters.Subscriber(
                '~input/info', CameraInfo)
        self.img_sub = message_filters.Subscriber('~input/image', Image)
        self.sync = message_filters.ApproximateTimeSynchronizer(
                [self.pc_sub,  self.img_sub, self.cam_info_sub],
                queue_size=100,
                slop=0.5)
        self.sync.registerCallback(self._callback)

    def unsubscribe(self):
        pass

    def _callback(self, cloud, img_msg, camera_info):
        rospy.loginfo('started')
        self.target_bin_name = rospy.get_param('~target_bin_name')

        # mask image
        self.camera_info = camera_info
        try:
            img_color = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr('{}'.format(e))
        self.img_color = img_color

        # get transform
        camera2bb_base = self.buffer.lookup_transform(
                target_frame=camera_info.header.frame_id,
                source_frame=self.target_bin.bbox.header.frame_id,
                time=rospy.Time.now(),
                timeout=rospy.Duration(1.0))
        # get mask_image
        self.mask_img = get_mask_img(
                camera2bb_base, self.target_bin, self.camera_model)

        # dist image
        bb_base2camera = self.buffer.lookup_transform(
                target_frame=self.target_bin.bbox.header.frame_id,
                source_frame=cloud.header.frame_id,
                time=rospy.Time.now(),
                timeout=rospy.Duration(1.0))
        self.dist_img, self.height_img = get_spatial_img(
                bb_base2camera, cloud, self.target_bin)

        self.set_apc_sample()
        # generate a binary image
        self.segmentation()
        try:
            self.img_pub.publish(self.bridge.cv2_to_imgmsg(
                    self.predicted_segment, encoding="passthrough"))
        except CvBridgeError as e:
            rospy.logerr('{}'.format(e))

        rospy.loginfo('ended')


if __name__ == '__main__':
    rospy.init_node('segmentation_in_bin')
    seg = RBOSegmentationInBinNode()
    seg.subscribe()
    rospy.spin()
