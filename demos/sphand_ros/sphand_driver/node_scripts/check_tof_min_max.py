#!/usr/bin/env python

# Use this script when ToF sensor is in stable state:
# rosrun sphand_driver check_tof_min_max.py __name:=check_tof_ave \
#   ~input:=/gripper_front/limb/left/tof_average/output

import numpy as np

from vl53l0x_mraa_ros.msg import RangingMeasurementDataStampedArray
import rospy


class CheckTofMinMax(object):

    def __init__(self):
        self.idx = rospy.get_param('~index', 0)
        self.min_range = np.inf
        self.max_range = -np.inf

        self.sub = rospy.Subscriber(
            '~input', RangingMeasurementDataStampedArray, self._cb)

    def _cb(self, msg):
        is_updated = False
        curr_range = msg.array[self.idx].data.range_millimeter
        if curr_range < self.min_range:
            self.min_range = curr_range
            is_updated = True
        if curr_range > self.max_range:
            self.max_range = curr_range
            is_updated = True
        if is_updated:
            rospy.loginfo('Current max range: %d, min range: %d',
                          self.max_range, self.min_range)


if __name__ == '__main__':
    rospy.init_node('check_tof_min_max')
    app = CheckTofMinMax()
    rospy.spin()
