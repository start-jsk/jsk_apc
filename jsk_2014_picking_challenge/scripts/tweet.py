#!/usr/bin/env python
#-*- coding:utf-8 -*-

import rospy
from std_msgs.msg import String
from std_srvs.srv import Empty

def saver_client():
    rospy.wait_for_service('image_saver/save')
    try:
        save = rospy.ServiceProxy('image_saver/save', Empty)
        resp = save()
        print "----",resp
        return resp
    except rospy.ServiceException, e:
        print "Service call faild: %s" % e

def talker():
    pub = rospy.Publisher('/tweet', String)
    rospy.init_node('baxter_tweet')
    str = "Hello /tmp/baxter_camera.png"
    #str = "Test"
    msg = String()
    msg.data = str
    rospy.loginfo(str)
    pub.publish(msg)

if __name__ == '__main__':
    print"Requesting ..."
    saver_client()
    try:
        talker()
    except rospy.ROSInterruptException: pass
