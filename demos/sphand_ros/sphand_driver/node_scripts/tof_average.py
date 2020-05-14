#!/usr/bin/env python

import numpy as np

from vl53l0x_mraa_ros.msg import RangingMeasurementDataStamped
from vl53l0x_mraa_ros.msg import RangingMeasurementDataStampedArray
import rospy


# TODO: Exclude duplicated element from averaging
class TofAverage(object):

    def __init__(self):
        self.queue_size = rospy.get_param('~queue_size', 5)
        self.queue = []

        self.pub = rospy.Publisher(
            '~output', RangingMeasurementDataStampedArray,
            queue_size=1)
        self.sub = rospy.Subscriber(
            '~input', RangingMeasurementDataStampedArray, self._cb)

    def _cb(self, msg):
        self.queue.append([e.data.range_millimeter for e in msg.array])
        while len(self.queue) > self.queue_size:
            self.queue.pop(0)
        averages = np.array(self.queue).mean(axis=0)
        pub_msg = RangingMeasurementDataStampedArray()
        pub_msg.header.stamp = msg.header.stamp
        for sub_e, average in zip(msg.array, averages):
            pub_e = RangingMeasurementDataStamped()
            pub_e.header.stamp = sub_e.header.stamp
            pub_e.data.range_millimeter = average
            pub_msg.array.append(pub_e)
        self.pub.publish(pub_msg)


if __name__ == '__main__':
    rospy.init_node('tof_average')
    app = TofAverage()
    rospy.spin()
