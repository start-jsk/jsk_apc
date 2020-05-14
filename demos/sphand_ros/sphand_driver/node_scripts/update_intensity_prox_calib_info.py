#!/usr/bin/env python

# Based on https://github.com/jsk-ros-pkg/jsk_robot/blob/2a4395761c50b03b0e039c8838184a647388c08f/jsk_robot_common/jsk_robot_utils/scripts/marker_msg_from_indigo_to_kinetic.py  # NOQA

import rospy
from sphand_driver_msgs.msg import IntensityProxCalibInfo
from sphand_driver_msgs.msg import IntensityProxCalibInfoArray
from sphand_driver_msgs.msg import IntensityProxCalibInfoOldArray


class UpdateIntensityProxCalibInfo(object):
    def __init__(self):
        rate = rospy.get_param('~rate', 1.0)
        self.suffix = rospy.get_param('~suffix', 'latest_type')

        self.publishers = dict()
        self.subscribers = dict()

        self.timer = rospy.Timer(rospy.Duration(rate), self.timer_cb)

    def msg_cb(self, msg, topic_name):
        try:
            msg_latest = IntensityProxCalibInfoArray()
            msg_latest.header = msg.header
            for info in msg.data:
                info_latest = IntensityProxCalibInfo()
                info_latest.header = info.header
                info_latest.distance = info.distance
                info_latest.diff_from_init = info.diff_from_init
                info_latest.reflectance_param = info.prop_const
                info_latest.init_value = info.init_value
                msg_latest.data.append(info_latest)
            self.publishers[topic_name].publish(msg_latest)
            rospy.logdebug('Relayed :{}'.format(topic_name))
        except Exception as e:
            rospy.logerr(e)

    def timer_cb(self, event=None):
        all_topics = rospy.get_published_topics()
        topic_names = []
        for topic in all_topics:
            if topic[0].split('/')[-2] == self.suffix:
                continue
            if topic[1] == 'sphand_driver_msgs/IntensityProxCalibInfoArray':
                topic_names.append(topic[0])

        for topic_name in topic_names:
            if topic_name not in self.publishers:
                new_topic_name = '/'.join(
                    topic_name.split('/')[:-1] +
                    [self.suffix, topic_name.split('/')[-1]])
                self.publishers[topic_name] = rospy.Publisher(
                    new_topic_name, IntensityProxCalibInfoArray, queue_size=1)
                rospy.logdebug(
                    'Advertised: {0} -> {1}'.format(
                        topic_name, new_topic_name))

        for topic_name, pub in self.publishers.items():
            # clean old topics
            if topic_name not in topic_names:
                try:
                    self.subscribers.pop(topic_name).unregister()
                    rospy.logdebug('Removed subscriber: {}'.format(topic_name))
                except Exception:
                    pass
                try:
                    self.publishers.pop(topic_name).unregister()
                    rospy.logdebug('Removed publisher: {}'.format(topic_name))
                except Exception:
                    pass
            # subscribe topics subscribed
            elif pub.get_num_connections() > 0:
                if topic_name not in self.subscribers:
                    self.subscribers[topic_name] = rospy.Subscriber(
                        topic_name,
                        IntensityProxCalibInfoOldArray,
                        self.msg_cb,
                        topic_name
                    )
                    rospy.logdebug('Subscribed {}'.format(topic_name))
            # unsubscribe topics unsubscribed
            else:
                if topic_name in self.subscribers:
                    self.subscribers.pop(topic_name).unregister()
                    rospy.logdebug('Unsubscribed {}'.format(topic_name))


if __name__ == '__main__':
    rospy.init_node('update_intensity_prox_calib_info')
    app = UpdateIntensityProxCalibInfo()
    rospy.spin()
