#!/usr/bin/env python

import subprocess
import sys

import rosgraph
import rospy

from jsk_arc2017_baxter.msg import GripperSensorStates


class ArduinoChecker(object):
    def __init__(self):
        super(ArduinoChecker, self).__init__()
        sys.stdout = sys.__stderr__
        self.master = rosgraph.Master('/rostopic')
        self.topic_names = [
            '/lgripper_sensors',
            '/rgripper_sensors',
        ]
        self.topic_times = {}
        self.start_time = rospy.Time.now()
        self.start_duration = rospy.get_param('~start_duration', 5)
        self.duration = rospy.get_param('~duration', 1)
        subs = []
        for topic_name in self.topic_names:
            self.topic_times[topic_name] = None
            subs.append(
                rospy.Subscriber(
                    topic_name, GripperSensorStates, self._sub_cb,
                    callback_args=topic_name))
        self.timer = rospy.Timer(rospy.Duration(1.0), self._timer_cb)

    def _sub_cb(self, msg, topic_name):
        self.topic_times[topic_name] = rospy.Time.now()

    def _timer_cb(self, event):
        for topic_name in self.topic_names:
            now = rospy.Time.now()
            if self.topic_times[topic_name] is None:
                last = self.start_time
                duration = self.start_duration
            else:
                last = self.topic_times[topic_name]
                duration = self.duration
            time_elapsed = (now - last).to_sec()
            rospy.logdebug(
                '{} received {} [sec] ago.'
                .format(topic_name, time_elapsed))
            if time_elapsed > duration:
                rospy.logerr(
                    '{} topic does not come in {} [sec]'
                    .format(topic_name, time_elapsed))
                res, node_name = self._kill_nodes(topic_name)
                if res:
                    rospy.logerr('node {} is killed'.format(node_name))
                else:
                    rospy.logerr('fail to kill node of {}'.format(topic_name))

    def _kill_nodes(self, topic_name):
        try:
            pubs = self.master.getSystemState()[0]
            pub = [pub for pub in pubs if pub[0] == topic_name][0]
            if pub[0] == topic_name:
                res = subprocess.call('rosnode kill {}'.format(pub[1][0]))
            return res, pub[1][0]
        except Exception:
            return False, None


if __name__ == '__main__':
    rospy.init_node('check_arduino')
    checker = ArduinoChecker()
    rospy.spin()
