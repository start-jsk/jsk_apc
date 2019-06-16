#!/usr/bin/env python

import rospy
import tf

from jsk_recognition_msgs.msg import BoundingBox


class BoundingBoxToTf(object):

    def __init__(self):
        self.tf_frame = rospy.get_param('~tf_frame', 'bounding_box')

        self.broadcaster = tf.TransformBroadcaster()

        self.sub = rospy.Subscriber('~input', BoundingBox, self._cb)

    def _cb(self, bbox):
        pos = bbox.pose.position
        ornt = bbox.pose.orientation
        self.broadcaster.sendTransform((pos.x, pos.y, pos.z),
                                       (ornt.x, ornt.y, ornt.z, ornt.w),
                                       rospy.Time.now(),
                                       self.tf_frame,
                                       bbox.header.frame_id)


if __name__ == '__main__':
    rospy.init_node(' bounding_box_to_tf')
    app = BoundingBoxToTf()
    rospy.spin()
