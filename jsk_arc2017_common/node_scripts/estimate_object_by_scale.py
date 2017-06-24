#!/usr/bin/env python

import rospy
from std_msgs.msg import Empty
from std_msgs.msg import Float32
from jsk_arc2017_common.msg import ObjectCandidates
from jsk_topic_tools import ConnectionBasedTransport

class EstimateObjectByScale(ConnectionBasedTransport):

    def __init__(self):
        super(EstimateObjectByScale, self).__init__()
        self.scale_inputs = rospy.get_param('~scale_inputs')
        self.object_weights = rospy.get_param('~object_weights')
        self.error = rospy.get_param('~error', 1.0)
        self.scale_values = [0.0] * len(self.scale_inputs)
        self.init_sum = 0.0
        self.weight_sum_pub = self.advertise('~weight_sum', Float32, queue_size=1)
        self.picked_pub = self.advertise('~picked_object_candidates', ObjectCandidates, queue_size=1)
        self.stowed_pub = self.advertise('~stowed_object_candidates', ObjectCandidates, queue_size=1)

    def subscribe(self):
        self.scale_subs = []
        for i, scale_input in enumerate(self.scale_inputs):
            self.scale_subs.append(rospy.Subscriber(scale_input, Float32, self._scale_cb, callback_args=i))
        self.init_sub = rospy.Subscriber('~initialize', Empty, self._initialize)

    def unsubscribe(self):
        self.scale_subs.unregister()
        self.init_sub.unregister()

    def _scale_cb(self, value, index):
        self.scale_values[index] = value.data
        weight_sum = sum(self.scale_values)
        weight_diff = weight_sum - self.init_sum
        picked_object = ObjectCandidates()
        stowed_object = ObjectCandidates()
        for obj, w in self.object_weights.items():
            if (weight_diff - self.error) < w < (weight_diff + self.error):
                stowed_object.candidates.append(obj)
            if (weight_diff - self.error) < -w < (weight_diff + self.error):
                picked_object.candidates.append(obj)

        self.weight_sum_pub.publish(Float32(weight_sum))
        self.picked_pub.publish(picked_object)
        self.stowed_pub.publish(stowed_object)

    def _initialize(self, msg):
        self.init_sum = sum(self.scale_values)

if __name__ == '__main__':
    rospy.init_node('estimate_object_by_scale')
    eobs = EstimateObjectByScale()
    rospy.spin()
