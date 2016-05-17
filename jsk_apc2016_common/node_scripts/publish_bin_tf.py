#!/usr/bin/env python

import rospy
from jsk_apc2016_common.msg import BinInfoArray

import roslaunch


class PublishTF(object):
    def __init__(self):
        bin_info_array = rospy.wait_for_message(
                "~input/bin_info_array", BinInfoArray, timeout=50)

        bbox_dict = self.bin_info_array_to_bbox_dict(bin_info_array)

        # publish bbox tf
        self.pub_dict = self.publish_bbox_tf(bbox_dict)

    def publish_bbox_tf(self, bbox_dict):
        # publish bounding boxes' center as static tf
        tf_nodes = {}
        for bin_, bbox in bbox_dict.iteritems():
            # something like 1 0 2 0 0 0 1
            transform_string = (
                    str(bbox_dict[bin_].pose.position.x) + ' ' +
                    str(bbox_dict[bin_].pose.position.y) + ' ' +
                    str(bbox_dict[bin_].pose.position.z) + ' ' +
                    str(bbox_dict[bin_].pose.orientation.x) + ' ' +
                    str(bbox_dict[bin_].pose.orientation.y) + ' ' +
                    str(bbox_dict[bin_].pose.orientation.z) + ' ' +
                    str(bbox_dict[bin_].pose.orientation.w) + ' ')

            tf_arg_string = (
                    transform_string +
                    str(bbox_dict[bin_].header.frame_id) + ' ' +
                    'bin_' + bin_ + ' ' + '100')  # last number is period

            tf_nodes[bin_] = roslaunch.core.Node(
                    package="tf",
                    name="pub_bin_" + bin_,
                    node_type="static_transform_publisher",
                    args=tf_arg_string,
                    respawn=True)
            launch = roslaunch.scriptapi.ROSLaunch()
            launch.start()
            launch.launch(tf_nodes[bin_])
        return tf_nodes

    def bin_info_array_to_bbox_dict(self, bin_info_array):
        bbox_dict = {}
        for bin_ in bin_info_array.array:
            bbox_dict[bin_.name] = bin_.bbox
        return bbox_dict

if __name__ == '__main__':
    rospy.init_node('sib_publish_tf')
    pub_tf = PublishTF()
    rospy.spin()
