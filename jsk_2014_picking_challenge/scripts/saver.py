#!/usr/bin/env python
import sys

import rospy
from std_srvs.srv import Empty

def saver_client():
	rospy.wait_for_service('image_saver/save')
	try:
		save = rospy.ServiceProxy('image_saver/save', Empty)
		resp = save()
		return resp
	except rospy.ServiceException, e:
		print "Service call faild: %s"%e

if __name__=="__main__":
	print "Requesting"
	saver_client()
