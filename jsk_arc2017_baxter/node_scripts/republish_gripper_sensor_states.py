#!/usr/bin/env python

import rospy

from jsk_arc2017_baxter.msg import GripperSensorStates
from force_proximity_ros.msg import ProximityArray
from force_proximity_ros.msg import Proximity
from std_msgs.msg import UInt16
from std_msgs.msg import Float64
from jsk_topic_tools import ConnectionBasedTransport


class RepublishGripperSensorStates(ConnectionBasedTransport):

    def __init__(self):
        super(RepublishGripperSensorStates, self).__init__()
        self.pub_prox = self.advertise(
            '~proximity_array', ProximityArray,
            queue_size=1)
        self.pub_pressure = self.advertise(
            '~pressure/state', Float64,
            queue_size=1)
        self.pub_r_finger_flex = self.advertise(
            '~flex/right/state', UInt16,
            queue_size=1)
        self.pub_l_finger_flex = self.advertise(
            '~flex/left/state', UInt16,
            queue_size=1)
        # low-pass filtered proximity reading
        self.average_value = []
        # FA-II value
        self.fa2 = []
        # Sensitivity of touch/release detection
        self.sensitivity = 1000
        # exponential average weight parameter / cut-off frequency for high-pass filter
        self.ea = 0.3

    def subscribe(self):
        self.sub = rospy.Subscriber('~input', GripperSensorStates,
                                    self._cb)

    def unsubscribe(self):
        self.sub.unregister()

    def _cb(self, gripper_sensor_states):
        proximity_array = ProximityArray()
        proximity_array.header.stamp = rospy.Time.now()
        for i, raw in enumerate(gripper_sensor_states.proximities):
            proximity = Proximity()
            try:
                self.average_value[i]
            except IndexError:
                self.average_value.append(raw)
            try:
                self.fa2[i]
            except IndexError:
                self.fa2.append(0)
            proximity.proximity = raw
            proximity.average = self.average_value[i]
            proximity.fa2derivative = self.average_value[i] - raw - self.fa2[i]
            self.fa2[i] = self.average_value[i] - raw
            proximity.fa2 = self.fa2[i]
            if self.fa2[i] < -self.sensitivity:
                proximity.mode = "T"
            elif self.fa2[i] > self.sensitivity:
                proximity.mode = "R"
            else:
                proximity.mode = "0"

            self.average_value[i] = self.ea * raw + (1 - self.ea) * self.average_value[i]
            proximity_array.proximities.append(proximity)

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
