#!/usr/bin/env python

import rospy
from std_srvs.srv import Trigger
from std_srvs.srv import TriggerResponse
from sensor_msgs.msg import Image
import message_filters


class ImageBuffer(object):

    def __init__(self):
        self.stamp = None
        self.pub_imgs = None
        rospy.loginfo("Subscribing image.")
        self.save_service = rospy.Service('~save', Trigger, self._save)
        input_topics = rospy.get_param('~input_topics', [])
        rate = rospy.get_param('~rate', 1.)
        approximate_sync = rospy.get_param('~approximate_sync', False)
        queue_size = rospy.get_param('~queue_size', 10)
        self.timer = rospy.Timer(rospy.Duration(1. / rate), self.publish)
        if len(input_topics) < 1:
            rospy.logwarn('rosparam ~input_topics is not set.')
            return

        self.pubs = []
        for i in range(0, len(input_topics)):
            pub = rospy.Publisher(
                '~output_{}'.format(str(i)), Image, queue_size=1)
            self.pubs.append(pub)

        if len(input_topics) == 1:
            sub = rospy.Subscriber(
                input_topics[0], Image, self._cb, queue_size=1)
        else:
            subs = []
            for input_topic in input_topics:
                sub = message_filters.Subscriber(input_topic, Image, queue_size=1)
                subs.append(sub)
            if approximate_sync:
                slop = rospy.get_param('~slop', 0.1)
                sync = message_filters.ApproximateTimeSynchronizer(
                    subs, queue_size=queue_size, slop=slop)
            else:
                sync = message_filters.TimeSynchronizer(
                    subs, queue_size=queue_size)
            sync.registerCallback(self._cb)

    def _cb(self, *sub_imgs):
        if self.stamp and self.stamp < sub_imgs[0].header.stamp:
            rospy.logerr('Time diff: %.2f[s]' %
                         (sub_imgs[0].header.stamp - self.stamp).secs)
            self.pub_imgs = sub_imgs
            self.stamp = None

    def _save(self, req):
        rospy.logerr('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        rospy.logerr('Save request is called to image buffer.')
        rospy.logerr('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        self.stamp = rospy.Time.now()
        self.pub_imgs = None
        res = TriggerResponse()
        res.success = True
        return res

    def publish(self, event):
        if self.pub_imgs is None:
            rospy.logwarn_throttle(5, "Input topic is not published yet.")
            return
        for pub, pub_img in zip(self.pubs, self.pub_imgs):
            pub_img.header.stamp = event.current_real
            pub.publish(pub_img)


if __name__ == '__main__':
    rospy.init_node('image_buffer')
    app = ImageBuffer()
    rospy.spin()
