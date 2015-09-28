#!/usr/bin/python
#-*- coding:utf-8 -*-
import os
import sys
import argparse

import roslib; roslib.load_manifest('jsk_2015_05_baxter_apc')
import rospy

import time

from sound_play.msg import *

def send_audio(speech, lang='en'):
    speech = speech.replace(' ', '+')

    time.sleep(1.0)
    pub = rospy.Publisher('robotsound', SoundRequest, queue_size=1)
    while pub.get_num_connections() < 1:
        print(pub.get_num_connections())
        print("waiting...")
    time.sleep(1.0)

    # print('http://translate.google.com/translate_tts?tl='+lang+'&q='+speech)
    req = SoundRequest(sound=SoundRequest.PLAY_FILE, command=SoundRequest.PLAY_ONCE, arg='http://translate.google.com/translate_tts?tl='+lang+'&q='+speech)
    # print(req)
    pub.publish(req)

def main():
    """Baxter can speak."""
    arg_fmt = argparse.RawTextHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=arg_fmt,
                                     description=main.__doc__)
    required = parser.add_argument_group('required arguments')
    required.add_argument(
        '-s', '--speak', required=True,
        help='What you want Baxter to speak.'
    )
    parser.add_argument(
        '-l', '--lang', type=str, default='en',
        help = 'select language'
    )
    args = parser.parse_args(rospy.myargv()[1:])

    rospy.init_node('robotspeaker', anonymous=True)
    # print(args.speak)
    send_audio(args.speak, lang=args.lang)

    return 0

if __name__ == '__main__':
   sys.exit(main())
