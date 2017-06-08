#!/usr/bin/env python

import rospy

from jsk_arc2017_baxter.msg import GripperSensorStates
from force_proximity_ros.msg import ProximityArray
from std_msgs.msg import UInt16
from std_msgs.msg import Float64
from jsk_topic_tools import ConnectionBasedTransport


class RepublishGripperSensorStates(ConnectionBasedTransport):

    def __init__(self):
        super(RepublishGripperSensorStates, self).__init__()
        self.pub_prox = self.advertise(
            'gripper_front/limb/right/proximity_array', ProximityArray,
            queue_size=1)
        self.pub_pressure = self.advertise(
            'gripper_front/limb/right/pressure/state', Float64,
            queue_size=1)
        self.pub_r_finger_flex = self.advertise(
            'gripper_front/limb/right/flex/right/state', UInt16,
            queue_size=1)
        self.pub_l_finger_flex = self.advertise(
            'gripper_front/limb/right/flex/left/state', UInt16,
            queue_size=1)

    def subscribe(self):
        self.sub = rospy.Subscriber('rgripper_sensors', GripperSensorStates,
                                    self._cb)

    def unsubscribe(self):
        self.sub.unregister()

    def _cb(self, gripper_sensor_states):
        proximity_array = ProximityArray()
        proximity_array.proximities = gripper_sensor_states.proximities

        pressure = Float64()
        pressure = gripper_sensor_states.pressure

        r_finger_flex = UInt16()
        r_finger_flex = gripper_sensor_states.r_finger_flex

        l_finger_flex = UInt16()
        l_finger_flex = gripper_sensor_states.l_finger_flex

        self.pub_prox.publish(proximity_array)
        self.pub_pressure.publish(pressure)
        self.pub_r_finger_flex.publish(r_finger_flex)
        self.pub_l_finger_flex.publish(l_finger_flex)


if __name__ == '__main__':
    rospy.init_node('republish_gripper_sensor_states')
    app = RepublishGripperSensorStates()
    rospy.spin()
