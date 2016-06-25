#!/usr/bin/env python

from jsk_recognition_msgs.msg import BoundingBoxArray
import rospy


class RemoveInfBoundingBoxes(object):
    def __init__(self):
        self.bbox_sub = rospy.Subscriber('~input', BoundingBoxArray, self.cb)
        self.bbox_pub = rospy.Publisher(
            '~output', BoundingBoxArray, queue_size=3)

    def cb(self, bbox_arr):
        boxes = []
        for box in bbox_arr.boxes:
            if (box.dimensions.x > 0 and
                    box.dimensions.y > 0 and
                    box.dimensions.z > 0):
                boxes.append(box)
        bbox_arr.boxes = boxes
        self.bbox_pub.publish(bbox_arr)


if __name__ == '__main__':
    rospy.init_node('remove_inf_bounding_boxes')
    remove_inf_bboxes = RemoveInfBoundingBoxes()
    rospy.spin()
