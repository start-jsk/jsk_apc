#!/usr/bin/env python

import rospy
import jsk_apc2016_common
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray
from std_msgs.msg import Header
from geometry_msgs.msg import Pose, Quaternion, Vector3, Point


class BinBoundingBoxPublisher():
    def __init__(self):
        self.bbox_dict = {}
        self.pub = rospy.Publisher('~boxes', BoundingBoxArray, queue_size=1)
        for shelf in ['~upper_shelf', '~lower_shelf']:
            initial_pos_list = rospy.get_param(shelf + '/initial_pos_list')
            initial_quat_list = rospy.get_param(shelf + '/initial_quat_list')
            dimensions = rospy.get_param(shelf + '/dimensions')
            frame_id_list = rospy.get_param(shelf + '/frame_id_list')
            prefixes = rospy.get_param(shelf + '/prefixes')
            camera_directions = rospy.get_param(shelf + '/camera_directions')

            for i, bin_ in enumerate(prefixes):
                bin_ = bin_.split('_')[1].lower()  # bin_A -> a
                self.bbox_dict[bin_] = {
                        'initial_pos': initial_pos_list[i],
                        'initial_quat': initial_quat_list[i],
                        'dimensions': dimensions[i],
                        'frame_id': frame_id_list[i]
                        }

    def main(self):
        duration = rospy.get_param('~duration', 1.0)
        rospy.Timer(rospy.Duration(duration), self.bbox_publish)
        rospy.spin()

    def bbox_publish(self, event):
        bbox_list = []
        stamp = rospy.Time.now()
        for bin_ in 'abcdefghijkl':
            bbox = self.bbox_dict[bin_]
            bbox_ = BoundingBox(
                    header=Header(
                        stamp=stamp,
                        frame_id=bbox['frame_id']
                        ),
                    pose=Pose(
                        position=Point(
                            x=bbox['initial_pos'][0],
                            y=bbox['initial_pos'][1],
                            z=bbox['initial_pos'][2]
                            ),
                        orientation=Quaternion(
                            x=bbox['initial_quat'][0],
                            y=bbox['initial_quat'][1],
                            z=bbox['initial_quat'][2],
                            w=bbox['initial_quat'][3]
                            )
                        ),
                    dimensions=Vector3(
                            x=bbox['dimensions'][0],
                            y=bbox['dimensions'][1],
                            z=bbox['dimensions'][2]
                            )
                    )
            bbox_list.append(bbox_)
        bbox_array = BoundingBoxArray(
                header=Header(
                    stamp=stamp,
                    frame_id='kiva_pod_base'
                    ),
                boxes=bbox_list
            )
        self.pub.publish(bbox_array)


if __name__ == '__main__':
    rospy.init_node('publish_bin_bbox')
    bbox = BinBoundingBoxPublisher()
    bbox.main()
