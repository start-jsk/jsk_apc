#!/usr/bin/env python

from jsk_topic_tools import ConnectionBasedTransport
from jsk_apc2016_common.segmentation_in_bin.bin_data import BinData
from sensor_msgs.msg import CameraInfo, PointCloud2
from geometry_msgs.msg import TransformStamped
from jsk_apc2016_common.msg import BinInfoArray
import rospy
import tf2_ros

def rbo_transform_publisher():
    cloud_msg = rospy.wait_for_message("~input", PointCloud2, timeout=10)
    bin_info_array_msg = rospy.wait_for_message("~input/bin_info_array", BinInfoArray, timeout=10)
    camera_info_msg = rospy.wait_for_message("~input/info", CameraInfo, timeout=10)

    target_bin_name = rospy.get_param('~target_bin_name')
    shelf = {}
    for bin_info in bin_info_array_msg.array:
        shelf[bin_info.name] = BinData(bin_info=bin_info)
    if target_bin_name is not None:
        target_bin = shelf[target_bin_name]

    bbox_frame_id = target_bin.bbox.header.frame_id
    camera_info_frame_id = camera_info_msg.header.frame_id
    cloud_frame_id = cloud_msg.header.frame_id

    camera2bb_base_pub = rospy.Publisher('~camera2bb_base', TransformStamped, queue_size=10)
    bb_base2camera_pub = rospy.Publisher('~bb_base2camera', TransformStamped, queue_size=10)
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)
    
    rate = rospy.Rate(10.0)
    while not rospy.is_shutdown():
        camera2bb_base = tf_buffer.lookup_transform(
                target_frame=camera_info_frame_id,
                source_frame=bbox_frame_id,
                time=rospy.Time(0),
                timeout=rospy.Duration(10.0))
        bb_base2camera = tf_buffer.lookup_transform(
                target_frame=bbox_frame_id,
                source_frame=cloud_frame_id,
                time=rospy.Time(0),
                timeout=rospy.Duration(10.0))
        camera2bb_base_pub.publish(camera2bb_base)
        bb_base2camera_pub.publish(bb_base2camera)
        rate.sleep()

if __name__ == '__main__':
    rospy.init_node('rbo_transform_publisher')
    rbo_transform_publisher()
