#!/usr/bin/env python

from jsk_recognition_msgs.msg import BoundingBoxArray
from jsk_recognition_msgs.msg import BoundingBox
import rospy


class ShelfBinBboxPublisher(object):
    def __init__(self):
        self.pub = rospy.Publisher('~boxes', BoundingBoxArray, queue_size=1)
        self.shelf_bin = rospy.get_param('~shelf_bin')
        self.duration = rospy.get_param('~duration', 1.0)
        self.bbox_dict = {}
        for bin_name in 'abc':
            bin_dict = self.shelf_bin['bin_' + bin_name.upper()]
            bbox = BoundingBox()
            bbox.header.frame_id = bin_dict['frame_id']
            bbox.pose.position.x = bin_dict['position']['x']
            bbox.pose.position.y = bin_dict['position']['y']
            bbox.pose.position.z = bin_dict['position']['z']
            bbox.pose.orientation.x = bin_dict['orientation']['x']
            bbox.pose.orientation.y = bin_dict['orientation']['y']
            bbox.pose.orientation.z = bin_dict['orientation']['z']
            bbox.pose.orientation.w = bin_dict['orientation']['w']
            bbox.dimensions.x = bin_dict['dimensions']['x']
            bbox.dimensions.y = bin_dict['dimensions']['y']
            bbox.dimensions.z = bin_dict['dimensions']['z']
            self.bbox_dict[bin_name] = bbox

    def main(self):
        rospy.Timer(rospy.Duration(self.duration), self.bbox_array_publish)
        rospy.spin()

    def bbox_array_publish(self, event):
        bbox_list = []
        stamp = rospy.Time.now()
        for bin_name in 'abc':
            bbox = self.bbox_dict[bin_name]
            bbox.header.stamp = stamp
            bbox_list.append(bbox)
        bbox_array = BoundingBoxArray()
        bbox_array.header.frame_id = bbox_list[0].header.frame_id
        bbox_array.header.stamp = stamp
        bbox_array.boxes = bbox_list
        self.pub.publish(bbox_array)


if __name__ == '__main__':
    rospy.init_node('publish_bin_bbox')
    bbox_publisher = ShelfBinBboxPublisher()
    bbox_publisher.main()
