#!/usr/bin/env python
#-*- coding:utf-8 -*-
"""
カメラの画像をとるテストファイル
"""

import rospy
from std_srvs.srv import Empty


def saver_client():
    rospy.wait_for_service('image_saver2/save')
    try:
        save = rospy.ServiceProxy('image_saver2/save', Empty)
        resp = save()
        print resp
        return resp
    except rospy.ServiceException, e:
        print "Service call faild: %s" % e

if __name__ == "__main__":
    print "Requesting ..."
    saver_client()
