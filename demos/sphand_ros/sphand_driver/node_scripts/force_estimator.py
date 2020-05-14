#!/usr/bin/env python

import numpy as np

from geometry_msgs.msg import Wrench
from sphand_driver_msgs.msg import IntensityProxCalibInfoArray
import rospy


# TODO: Add estimation in fingers
# TODO: Add torque estimation
class ForceEstimator(object):

    def __init__(self):
        # Unit: mm
        self.rubber_t = rospy.get_param('~rubber_thickness', None)
        if self.rubber_t is not None:
            self.rubber_t = np.array(self.rubber_t)
        # Unit: N/mm
        self.rubber_k = rospy.get_param('~rubber_stiffness', None)
        if self.rubber_k is not None:
            self.rubber_k = np.array(self.rubber_k)

        self.pub_palm = rospy.Publisher(
            '~output/palm', Wrench,
            queue_size=1)
        self.sub = rospy.Subscriber(
            '~input', IntensityProxCalibInfoArray, self._cb)

    def _cb(self, msg):
        if self.rubber_t is None:
            self.rubber_t = np.zeros(len(msg.data))
        if self.rubber_k is None:
            self.rubber_k = np.zeros(len(msg.data))
        assert len(msg.data) == len(self.rubber_t) == len(self.rubber_k)

        dist = np.array([d.distance for d in msg.data])
        force = self.rubber_k * (self.rubber_t - dist)
        force = np.where(force < 0, 0, force)
        pub_palm_msg = Wrench()
        pub_palm_msg.force.x = 0
        pub_palm_msg.force.y = 0
        pub_palm_msg.force.z = force[6]
        self.pub_palm.publish(pub_palm_msg)


if __name__ == '__main__':
    rospy.init_node('force_estimator')
    app = ForceEstimator()
    rospy.spin()
