#!/usr/bin/env python

import rospy
import tf2_ros
from sensor_msgs.msg import Image, CameraInfo
from jsk_apc2016_common.msg import BinInfoArray
from image_geometry import cameramodels
from jsk_apc2016_common.segmentation_in_bin.bin_data import BinData
from tf2_geometry_msgs import do_transform_point
from matplotlib.path import Path
from jsk_topic_tools import ConnectionBasedTransport
from cv_bridge import CvBridge
import numpy as np


class TFBboxToMask(ConnectionBasedTransport):

    def __init__(self):
        super(TFBboxToMask, self).__init__()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.bridge = CvBridge()

        self.camera_model = cameramodels.PinholeCameraModel()
        bin_info_arr_msg = rospy.wait_for_message("~input/bin_info_array", BinInfoArray, timeout=50)
        self.shelf = {}
        for bin_info in bin_info_arr_msg.array:
            self.shelf[bin_info.name] = BinData(bin_info=bin_info)
        self.pub = self.advertise('~output', Image, queue_size=1)

    def subscribe(self):
        self.sub = rospy.Subscriber('~input', CameraInfo, self._callback, queue_size=3)

    def unsubscribe(self):
        self.sub.unregister()

    def _callback(self, camera_info):
        if rospy.get_param('~target_bin_name') not in 'abcdefghijkl':
            return

        target_bin_name = rospy.get_param('~target_bin_name')
        target_bin = self.shelf[target_bin_name]
        self.camera_model.fromCameraInfo(camera_info)

        # get transform
        camera2bb_base = self.tf_buffer.lookup_transform(
                target_frame=camera_info.header.frame_id,
                source_frame=target_bin.bbox.header.frame_id,
                time=rospy.Time(0),
                timeout=rospy.Duration(10.0))

        mask_img = self.get_mask_img(camera2bb_base, target_bin, self.camera_model)
        mask_msg = self.bridge.cv2_to_imgmsg(mask_img, encoding="passthrough")
        mask_msg.header = camera_info.header
        self.pub.publish(mask_msg)

    def get_mask_img(self, transform, target_bin, camera_model):
        """
        :param point: point that is going to be transformed
        :type point: PointStamped
        :param transform: camera_frame -> bbox_frame
        :type transform: Transform
        """
        # check frame_id of a point and transform just in case
        assert camera_model.tf_frame == transform.header.frame_id
        assert target_bin.bbox.header.frame_id == transform.child_frame_id

        transformed_list = [
                do_transform_point(corner, transform)
                for corner in target_bin.corners]
        projected_points = self.project_points(transformed_list, camera_model)

        # generate an polygon that covers the region
        path = Path(projected_points)
        x, y = np.meshgrid(
                np.arange(camera_model.width),
                np.arange(camera_model.height))
        x, y = x.flatten(), y.flatten()
        points = np.vstack((x, y)).T
        mask_img = path.contains_points(points).reshape(camera_model.height, camera_model.width)
        mask_img = (mask_img * 255).astype(np.uint8)
        return mask_img

    def project_points(self, points, camera_model):
        """
        :param points: list of geometry_msgs.msg.PointStamped
        :type list of stamped points :
        :param projected_points: list of camera_coordinates
        :type  projected_points: (u, v)

        The frames of the points and the camera_model are same.
        """
        # generate mask iamge
        for point in points:
            if point.header.frame_id != camera_model.tf_frame:
                raise ValueError('undefined')
        if len(points) != 4:
            raise ValueError('undefined')

        projected_points = []
        for point in points:
            projected_points.append(
                    camera_model.project3dToPixel(
                            [point.point.x, point.point.y, point.point.z]))
        return projected_points

if __name__ == '__main__':
    rospy.init_node('tf_bbox_to_mask')
    tfmask = TFBboxToMask()
    rospy.spin()
