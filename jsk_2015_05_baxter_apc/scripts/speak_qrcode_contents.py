#!/usr/bin/python
#-*- coding:utf-8 -*-
import urllib, pycurl, os

import roslib; roslib.load_manifest('jsk_2014_picking_challenge')
import rospy

def downloadFile(url, fileName):
    fp = open(fileName, "wb")
    curl = pycurl.Curl()
    curl.setopt(pycurl.URL, url)
    curl.setopt(pycurl.WRITEDATA, fp)
    curl.perform()
    curl.close()
    fp.close()

def getGoogleSpeechURL(phrase, select_english=True):
    if select_english:
        googleTranslateURL = "http://translate.google.com/translate_tts?tl=en&"
    else:
        googleTranslateURL = "http://translate.google.com/translate_tts?tl=ja&"
    parameters = {'q': phrase}
    data = urllib.urlencode(parameters)
    googleTranslateURL = "%s%s" % (googleTranslateURL,data)
    return googleTranslateURL

def speakSpeechFromText(phrase):
    googleSpeechURL = getGoogleSpeechURL(phrase, select_english=False)
    downloadFile(googleSpeechURL,"tts.mp3")
    os.system("mplayer tts.mp3 &")

from sound_play.msg import *

def main():
    pub = rospy.Publisher('robotsound', SoundRequest, queue_size=100)
    rospy.init_node('robotspeaker', anonymous=True)
    req = SoundRequest(sound=SoundRequest.PLAY_FILE, command=SoundRequest.PLAY_ONCE, arg='http://translate.google.com/translate_tts?tl=ja&q=' + "test")
    print(req)
    pub.publish(req)

if __name__ == '__main__':
    main()

#   speakSpeechFromText("testing, testing, 1 2 3."
