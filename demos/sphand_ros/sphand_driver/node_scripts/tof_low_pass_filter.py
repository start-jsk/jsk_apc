#!/usr/bin/env python

import numpy as np

from vl53l0x_mraa_ros.msg import RangingMeasurementDataStamped
from vl53l0x_mraa_ros.msg import RangingMeasurementDataStampedArray
import rospy


# TODO: Exclude duplicated element from filter
class TofLowPassFilter(object):

    def __init__(self):
        # Default cut-off frequency is 9Hz
        self.input_coeff = rospy.get_param('~input_coeff', 0.825)
        self.ranges_filtered = None

        self.pub = rospy.Publisher(
            '~output', RangingMeasurementDataStampedArray,
            queue_size=1)
        self.sub = rospy.Subscriber(
            '~input', RangingMeasurementDataStampedArray, self._cb)

    def _cb(self, msg):
        ranges_raw = np.array([e.data.range_millimeter for e in msg.array])
        if self.ranges_filtered is None:
            self.ranges_filtered = ranges_raw.copy()
        self.ranges_filtered = self.input_coeff * ranges_raw \
            + (1 - self.input_coeff) * self.ranges_filtered
        pub_msg = RangingMeasurementDataStampedArray()
        pub_msg.header.stamp = msg.header.stamp
        for sub_e, range_filtered in zip(msg.array, self.ranges_filtered):
            pub_e = RangingMeasurementDataStamped()
            pub_e.header.stamp = sub_e.header.stamp
            pub_e.data.range_millimeter = range_filtered
            pub_msg.array.append(pub_e)
        self.pub.publish(pub_msg)


if __name__ == '__main__':
    rospy.init_node('tof_low_pass_filter')
    app = TofLowPassFilter()
    rospy.spin()
