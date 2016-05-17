#!/usr/bin/env python

import rospy
from jsk_apc2016_common.msg import BinInfoArray

import tf2_ros
from geometry_msgs.msg import TransformStamped


class PublishTF(object):
    def __init__(self):
        bin_info_array = rospy.wait_for_message(
                "~input/bin_info_array", BinInfoArray, timeout=50)

        bbox_dict = self.bin_info_array_to_bbox_dict(bin_info_array)

        # publish bbox tf
        self.publish_bbox_tf(bbox_dict)

    def publish_bbox_tf(self, bbox_dict):
        # publish bounding boxes' center as static tf
        broadcaster = tf2_ros.StaticTransformBroadcaster()
        for bin_, bbox in bbox_dict.iteritems():
            bin_tf = TransformStamped()

            bin_tf.header.stamp = rospy.Time.now()
            bin_tf.header.frame_id = bbox.header.frame_id
            bin_tf.child_frame_id = 'bin_' + bin_

            bin_tf.transform.translation.x = bbox.pose.position.x
            bin_tf.transform.translation.y = bbox.pose.position.y
            bin_tf.transform.translation.z = bbox.pose.position.z
            bin_tf.transform.rotation.x = bbox.pose.orientation.x
            bin_tf.transform.rotation.y = bbox.pose.orientation.y
            bin_tf.transform.rotation.z = bbox.pose.orientation.z
            bin_tf.transform.rotation.w = bbox.pose.orientation.w

            broadcaster.sendTransform(bin_tf)

    def bin_info_array_to_bbox_dict(self, bin_info_array):
        bbox_dict = {}
        for bin_ in bin_info_array.array:
            bbox_dict[bin_.name] = bin_.bbox
        return bbox_dict

if __name__ == '__main__':
    rospy.init_node('sib_publish_tf')
    pub_tf = PublishTF()
    rospy.spin()
