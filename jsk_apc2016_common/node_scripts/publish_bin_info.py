#!/usr/bin/env python

import rospy
from jsk_apc2016_common.msg import BinInfo, BinInfoArray
from jsk_recognition_msgs.msg import BoundingBox
from geometry_msgs.msg import Pose
import jsk_apc2016_common.segmentation_in_bin.\
        segmentation_in_bin_helper as helper
from std_msgs.msg import Header

import json


def get_bin_contents_from_json(json_file):
    with open(json_file, 'r') as f:
        bin_contents = json.load(f)['bin_contents']
    for bin_, objects in bin_contents.items():
        bin_ = bin_.split('_')[1].lower()  # bin_A -> a
        yield (bin_, objects)


def get_target_from_json(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)['work_order']
    for order in data:
        bin_ = order['bin'].split('_')[1].lower()  # bin_A -> a
        target_object = order['item']
        yield (bin_, target_object)


class BinInfoArrayPublisher(object):
    def __init__(self):
        self.bbox_dict = {}
        self.bin_contents_dict = {}
        self.targets_dict = {}
        self.cam_direction_dict = {}

        self.json_file = rospy.get_param('~json', None)

        # get bbox from rosparam
        self.from_shelf_param('upper')
        self.from_shelf_param('lower')

        # get contents of bin from json
        self.get_bin_contents()
        self.get_targets()

        # create bin_msg
        self.create_bin_info_arr()

        pub_bin_info_arr = rospy.Publisher('~bin_array', BinInfoArray, queue_size=1)
        rate = rospy.Rate(rospy.get_param('rate', 1))
        while not rospy.is_shutdown():
            pub_bin_info_arr.publish(self.bin_info_arr)
            rate.sleep()

    def from_shelf_param(self, upper_lower):
        upper_lower = upper_lower + '_shelf'
        initial_pos_list = rospy.get_param(
                '~' + upper_lower + '/initial_pos_list')
        initial_quat_list = rospy.get_param(
                '~' + upper_lower + '/initial_quat_list')
        dimensions = rospy.get_param(
                '~' + upper_lower + '/dimensions')
        frame_id_list = rospy.get_param(
                '~' + upper_lower + '/frame_id_list')
        prefixes = rospy.get_param(
                '~' + upper_lower + '/prefixes')
        camera_directions = rospy.get_param(
                '~' + upper_lower + '/camera_directions')

        for i, bin_ in enumerate(prefixes):
            bin_ = bin_.split('_')[1].lower()  # bin_A -> a
            header = Header(
                    stamp=rospy.Time.now(),
                    frame_id=frame_id_list[i])
            self.bbox_dict[bin_] = BoundingBox(
                    header=header,
                    pose=Pose(
                            position=helper.point(initial_pos_list[i]),
                            orientation=helper.quaternion(initial_quat_list[i])),
                    dimensions=helper.vector3(dimensions[i]))
            self.cam_direction_dict[bin_] = camera_directions[i]

    def get_bin_contents(self):
        if self.json_file is None:
            rospy.logerr('must set json file path to ~json')
            return
        bin_contents = get_bin_contents_from_json(
                json_file=self.json_file)
        bin_contents = list(bin_contents)

        for bin_, objects in bin_contents:
            self.bin_contents_dict[bin_.encode('ascii')] = objects

    def get_targets(self):
        if self.json_file is None:
            rospy.logerr('must set json file path to ~json')
            return
        targets = get_target_from_json(
                json_file=self.json_file)
        targets = list(targets)

        for bin_, target in targets:
            self.targets_dict[bin_.encode('ascii')] = target

    def create_bin_info_arr(self):

        self.bin_info_arr = BinInfoArray()
        for bin_ in self.bin_contents_dict.keys():
            self.bin_info_arr.array.append(BinInfo(
                    name=bin_,
                    objects=self.bin_contents_dict[bin_],
                    target=self.targets_dict[bin_],
                    bbox=self.bbox_dict[bin_],
                    camera_direction=self.cam_direction_dict[bin_]))


if __name__ == '__main__':
    rospy.init_node('set_bin_param')
    bin_publisher = BinInfoArrayPublisher()
    rate = rospy.Rate(rospy.get_param('rate', 1))
    while not rospy.is_shutdown():
        rate.sleep()
