#!/usr/bin/python
#-*- coding:utf-8 -*-
import os
import sys
import argparse

import roslib; roslib.load_manifest('jsk_2015_05_baxter_apc')
import rospy

from sound_play.msg import *

from zbar_ros.msg import Marker

def send_audio(speech, lang='en'):
    speech = speech.replace('_', '+')
    speech = speech.replace(':', '+')
    speech = speech.replace(' ', '+')
    pub = rospy.Publisher('robotsound', SoundRequest, queue_size=100)
    # print('http://translate.google.com/translate_tts?tl='+lang+'&q='+speech)
    req = SoundRequest(sound=SoundRequest.PLAY_FILE, command=SoundRequest.PLAY_ONCE, arg='http://translate.google.com/translate_tts?tl='+lang+'&q='+speech)
    # print(req)
    rospy.loginfo(speech)
    pub.publish(req)

detected_markers = {}
def callback(data):
    # rospy.loginfo(data.data)
    if data.data in detected_markers.keys():
        return
    detected_markers[data.data] = 1
    send_audio("marker detected", lang='en')
    rospy.sleep(1.)
    send_audio(data.data, lang='en')


def main():
    rospy.init_node("speak_qrcode_contents")
    rospy.Subscriber('markers', Marker, callback)
    rospy.spin()

if __name__ == '__main__':
    main()
