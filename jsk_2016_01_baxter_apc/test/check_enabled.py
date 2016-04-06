#!/usr/bin/env python
PKG = 'jsk_2016_01_baxter_apc'

import sys, unittest, time
import rospy, rostest

from baxter_core_msgs.msg import AssemblyState

## A sample python unit test
class TestCheckEnabled(unittest.TestCase):
    
    def __init__(self, *args):
        super(TestCheckEnabled, self).__init__(*args)
        self.success = False

    def callback(self, msg):
        print(rospy.get_caller_id(), "status.enabled %s"%msg.enabled)
        self.success = msg.enabled

    def test_robot_state(self):
        rospy.init_node('check_enabled', anonymous=True)
        rospy.Subscriber("/robot/state", AssemblyState, self.callback)
        timeout_t = rospy.Time.now() + rospy.Duration(20.0) #20 seconds
        while not rospy.is_shutdown() and not self.success and rospy.Time.now() < timeout_t:
            rospy.sleep(1)
        self.assert_(self.success)

if __name__ == '__main__':
    import rostest
    rostest.rosrun(PKG, 'check_enabled', TestCheckEnabled)
