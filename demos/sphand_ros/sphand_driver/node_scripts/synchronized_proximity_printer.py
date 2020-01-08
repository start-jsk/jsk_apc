#!/usr/bin/env python

# Usage:
# rosrun jsk_arc2017_baxter synchronized_proximity_printer.py \
#   ~input/tof:=/gripper_front/limb/left/tof_low_pass_filter/output \
#   ~input/prox:=/gripper_front/limb/left/intensity_prox_calibrator/output \
#   > test.log
# In another terminal:
# rosservice call /synchronized_proximity_printer/start_printing

from __future__ import print_function

import pprint

import message_filters
import rospy
from sphand_driver_msgs.msg import IntensityProxCalibInfoArray
from std_srvs.srv import Empty
from std_srvs.srv import EmptyResponse
from vl53l0x_mraa_ros.msg import RangingMeasurementDataStampedArray


class SynchronizedProximityPrinter(object):

    def __init__(self):
        self.print_num = rospy.get_param('~print_num', 10)
        # Default target is right tip sensor
        self.target_idx = rospy.get_param('~target_index', 3)
        self.current_num = 0

        self.start_printing_srv = rospy.Service(
            '~start_printing', Empty, self.start_printing)

    def subscribe(self):
        self.sub_tof = message_filters.Subscriber(
            '~input/tof', RangingMeasurementDataStampedArray, queue_size=1,
            buff_size=2**24
        )
        self.sub_prox = message_filters.Subscriber(
            '~input/prox', IntensityProxCalibInfoArray, queue_size=1,
            buff_size=2**24
        )
        sync = message_filters.ApproximateTimeSynchronizer(
            [self.sub_tof, self.sub_prox],
            queue_size=50,
            slop=0.1,
        )
        sync.registerCallback(self.printer)

    def unsubscribe(self):
        self.sub_tof.unregister()
        self.sub_prox.unregister()

    def start_printing(self, req):
        self.current_num = 0
        self.subscribe()
        return EmptyResponse()

    def printer(self, tof, prox):
        if self.current_num == 0:
            rospy.loginfo('Printing starts')
        print('Target sensor index: {}'.format(self.target_idx))
        print('========================================')
        print('# [Data {}]'.format(self.current_num))
        print('## [ToF]')
        pprint.pprint(tof.array[self.target_idx])
        print('## [Proximity]')
        pprint.pprint(prox.data[self.target_idx])
        self.current_num = self.current_num + 1
        if self.current_num >= self.print_num:
            rospy.loginfo('Printing ends')
            self.unsubscribe()
            self.current_num = 0


if __name__ == '__main__':
    rospy.init_node('synchronized_proximity_printer')
    app = SynchronizedProximityPrinter()
    rospy.spin()
