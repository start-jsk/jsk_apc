#!/usr/bin/env python

import rospy
from jsk_apc2016_common.msg import BinInfo, BinInfoArray
from jsk_recognition_msgs.msg import BoundingBox
from geometry_msgs.msg import (
    Pose,
    Quaternion,
    Vector3,
    Point)
import jsk_apc2016_common
from std_msgs.msg import Header

import os


class BinInfoArrayPublisher(object):
    def __init__(self):
        self.bbox_dict = {}
        self.bin_contents_dict = {}
        self.targets_dict = {}
        self.cam_direction_dict = {}
        self.json_file = None

        self.pub_bin_info_arr = rospy.Publisher(
            '~bin_array', BinInfoArray, queue_size=1)

    def main(self):
        duration = rospy.Duration(rospy.get_param('~duration', 1))
        rospy.Timer(duration, self.bin_info_publlish)
        rospy.spin()

    def bin_info_publlish(self, event):
        json = rospy.get_param('~json', None)

        # update bin_info_arr only when rosparam: json is changd
        if self.json_file != json:
            if not os.path.isfile(json) or json[-4:] != 'json':
                rospy.logwarn('wrong json file name')
                return
            self.json_file = json

            # get bbox from rosparam
            self.from_shelf_param('upper')
            self.from_shelf_param('lower')

            # get contents of bin from json
            self.bin_contents_dict = jsk_apc2016_common.get_bin_contents(
                self.json_file)
            self.targets_dict = jsk_apc2016_common.get_work_order(
                self.json_file)

            # create bin_msg
            self.create_bin_info_arr()

        self.bin_info_arr.header.stamp = rospy.Time.now()
        self.pub_bin_info_arr.publish(self.bin_info_arr)

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
                            position=Point(*initial_pos_list[i]),
                            orientation=Quaternion(*initial_quat_list[i])),
                    dimensions=Vector3(*dimensions[i]))
            self.cam_direction_dict[bin_] = camera_directions[i]

    def create_bin_info_arr(self):
        self.bin_info_arr = BinInfoArray()
        for bin_ in 'abcdefghijkl':
            self.bin_info_arr.array.append(BinInfo(
                    header=Header(
                            stamp=rospy.Time(0),
                            seq=0,
                            frame_id='bin_'+bin_),
                    name=bin_,
                    objects=self.bin_contents_dict[bin_],
                    target=self.targets_dict[bin_],
                    bbox=self.bbox_dict[bin_],
                    camera_direction=self.cam_direction_dict[bin_]))


if __name__ == '__main__':
    rospy.init_node('publish_bin_info')
    bin_publisher = BinInfoArrayPublisher()
    bin_publisher.main()
