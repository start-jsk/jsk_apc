#!/usr/bin/env python

from jsk_arc2017_common.msg import WeightStamped
from jsk_recognition_msgs.msg import Label
from jsk_recognition_msgs.msg import LabelArray
from jsk_topic_tools import ConnectionBasedTransport
import message_filters
import rospy
from std_srvs.srv import Trigger
from std_srvs.srv import TriggerResponse


class EstimateObjectByScale(ConnectionBasedTransport):

    def __init__(self):
        super(EstimateObjectByScale, self).__init__()
        self.scale_inputs = rospy.get_param('~scale_inputs')
        self.object_weights = rospy.get_param('~object_weights')
        self.error = rospy.get_param('~error', 1.0)

        self.scale_values = [0.0] * len(self.scale_inputs)
        self.init_sum = 0.0
        self.weight_sum_pub = self.advertise(
            '~output/weight_sum', WeightStamped, queue_size=1)
        self.picked_pub = self.advertise(
            '~output/candidates/picked', LabelArray, queue_size=1)
        self.placed_pub = self.advertise(
            '~output/candidates/placed', LabelArray, queue_size=1)
        self.init_srv = rospy.Service(
            '~initialize', Trigger, self._initialize)

    def subscribe(self):
        use_async = rospy.get_param('~approximate_sync', False)
        queue_size = rospy.get_param('~queue_size', 10)
        # add candidates subscriber
        self.sub_candidates = rospy.Subscriber(
            '~input/candidates', LabelArray, self._candidates_cb)
        # add scale subscriber
        self.subs = []
        for scale_input in self.scale_inputs:
            sub = message_filters.Subscriber(scale_input, WeightStamped)
            self.subs.append(sub)
        if use_async:
            slop = rospy.get_param('~slop', 0.1)
            sync = message_filters.ApproximateTimeSynchronizer(
                self.subs, queue_size=queue_size, slop=slop)
        else:
            sync = message_filters.TimeSynchronizer(
                self.subs, queue_size=queue_size)
        sync.registerCallback(self._scale_cb)

    def unsubscribe(self):
        self.sub_candidates.unregister()
        for sub in self.subs:
            sub.unregister()

    def _candidates_cb(self, labels_msg):
        candidates = {}
        for label_msg in labels_msg.labels:
            candidates[label_msg.name] = label_msg.id

        weight_sum = sum(self.scale_values)
        weight_diff = weight_sum - self.init_sum
        sum_msg = WeightStamped()
        sum_msg.header = labels_msg.header
        sum_msg.weight.value = weight_sum
        sum_msg = WeightStamped()
        sum_msg.header = labels_msg.header
        sum_msg.weight.value = weight_sum

        pick_msg = LabelArray()
        place_msg = LabelArray()
        pick_msg.header = labels_msg.header
        place_msg.header = labels_msg.header
        diff_lower = weight_diff - self.error
        diff_upper = weight_diff + self.error
        for obj, w in self.object_weights.items():
            if obj not in candidates.keys():
                continue
            if diff_upper > w and w > diff_lower:
                label = Label()
                label.id = candidates[obj]
                label.name = obj
                place_msg.labels.append(label)
            elif diff_upper > -w and -w > diff_lower:
                label = Label()
                label.id = candidates[obj]
                label.name = obj
                pick_msg.labels.append(label)

        self.weight_sum_pub.publish(sum_msg)
        self.picked_pub.publish(pick_msg)
        self.placed_pub.publish(place_msg)

    def _scale_cb(self, *args):
        assert len(args) == len(self.scale_values)
        for i, weight_msg in enumerate(args):
            self.scale_values[i] = weight_msg.weight.value

    def _initialize(self, req):
        is_success = True
        try:
            self.init_sum = sum(self.scale_values)
        except Exception:
            is_success = False
        return TriggerResponse(success=is_success)

if __name__ == '__main__':
    rospy.init_node('estimate_object_by_scale')
    eobs = EstimateObjectByScale()
    rospy.spin()
