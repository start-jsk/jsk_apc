#!/usr/bin/env python

from jsk_arc2017_common.msg import WeightStamped
from jsk_recognition_msgs.msg import BoolStamped
import message_filters
import rospy
from std_srvs.srv import Trigger
from std_srvs.srv import TriggerResponse


class WeightChangeDetector(object):

    def __init__(self):
        self.input_topics = rospy.get_param('~input_topics')
        self.error = rospy.get_param('~error', 1.0)

        self.weight_sum_at_reset = 0.0
        self.prev_weight_values = [0] * len(self.input_topics)

        self.weight_sum_pub = rospy.Publisher(
            '~debug/weight_sum', WeightStamped, queue_size=1)
        self.weight_sum_at_reset_pub = rospy.Publisher(
            '~debug/weight_sum_at_reset', WeightStamped, queue_size=1)
        self.changed_pub = rospy.Publisher(
            '~output/changed_from_reset', BoolStamped, queue_size=1)
        self.can_reset = False
        self.subscribe()

    def subscribe(self):
        use_async = rospy.get_param('~approximate_sync', False)
        queue_size = rospy.get_param('~queue_size', 10)
        # add scale subscriber
        self.subs = []
        for input_topic in self.input_topics:
            sub = message_filters.Subscriber(
                input_topic, WeightStamped, queue_size=1)
            self.subs.append(sub)
        if use_async:
            slop = rospy.get_param('~slop', 0.1)
            sync = message_filters.ApproximateTimeSynchronizer(
                self.subs, queue_size=queue_size, slop=slop)
        else:
            sync = message_filters.TimeSynchronizer(
                self.subs, queue_size=queue_size)
        sync.registerCallback(self._scale_cb)

    # def unsubscribe(self):
    #     for sub in self.subs:
    #         sub.unregister()

    def _scale_cb(self, *weight_msgs):
        assert len(weight_msgs) == len(self.prev_weight_values)

        # Publish debug info
        weight_values = [w.weight.value for w in weight_msgs]
        weight_sum = sum(weight_values)
        sum_msg = WeightStamped()
        sum_msg.header = weight_msgs[0].header
        sum_msg.weight.value = weight_sum
        sum_msg.weight.stable = all(msg.weight.stable for msg in weight_msgs)
        sum_at_reset_msg = WeightStamped()
        sum_at_reset_msg.header = weight_msgs[0].header
        sum_at_reset_msg.weight.value = self.weight_sum_at_reset
        sum_at_reset_msg.weight.stable = True
        self.weight_sum_at_reset_pub.publish(sum_at_reset_msg)
        self.weight_sum_pub.publish(sum_msg)

        if not sum_msg.weight.stable:
            return  # unstable

        # Store stable weight and enable resetting
        self.prev_weight_values = weight_values
        if not self.can_reset:
            self.reset_srv = rospy.Service('~reset', Trigger, self._reset)
            self.can_reset = True

        # Judge if scale value is changed
        weight_diff = weight_sum - self.weight_sum_at_reset
        changed_msg = BoolStamped()
        changed_msg.header = weight_msgs[0].header
        if -self.error < weight_diff < self.error:
            changed_msg.data = False
        else:
            changed_msg.data = True
        self.changed_pub.publish(changed_msg)

    def _reset(self, req):
        is_success = True
        try:
            self.weight_sum_at_reset = sum(self.prev_weight_values)
        except Exception:
            is_success = False
        return TriggerResponse(success=is_success)


if __name__ == '__main__':
    rospy.init_node('weight_change_detector')
    app = WeightChangeDetector()
    rospy.spin()
