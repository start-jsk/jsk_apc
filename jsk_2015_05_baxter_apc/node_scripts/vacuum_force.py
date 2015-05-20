#!/usr/bin/env python

from std_msgs.msg import Bool
from geometry_msgs.msg import WrenchStamped

import rospy
import math
rospy.init_node("vacuum_force")

g_force = {}
g_force['left'] = 0
g_force['right'] = 0
def vacuum_state_cb(msg, arm):
    print("state call back  %8s %7.2f %s"%(arm, g_force[arm], msg.data))
    pub_msg = Bool(False)
    if g_force[arm] > 15 and msg.data is True:
        pub_msg.data = True
    if arm == "left":
        left_vacuum_pub.publish(pub_msg)
    elif arm == "right":
        right_vacuum_pub.publish(pub_msg)

def wrench_cb(msg, arm):
    force = math.sqrt(msg.wrench.force.x ** 2 + msg.wrench.force.y ** 2 + msg.wrench.force.z ** 2)
    g_force[arm] = 0.8 * force + 0.2 * g_force[arm]
    print("wrench call back %8s %7.1f %7.1f(%7.2f %7.2f %7.2f)"%(arm, g_force[arm], force, msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z))

if __name__ == "__main__":
    left_vacuum_pub = rospy.Publisher("/gripper_grabbed/limb/left/state2", Bool,  queue_size=3)
    right_vacuum_pub = rospy.Publisher("/gripper_grabbed/limb/right/state2", Bool,  queue_size=3)

    rospy.Subscriber('/gripper_grabbed/limb/right/state', Bool, vacuum_state_cb, 'right')
    rospy.Subscriber('/gripper_grabbed/limb/right/state', Bool, vacuum_state_cb, 'left')

    rospy.Subscriber('/right_endeffector/wrench', WrenchStamped, wrench_cb, 'right')
    rospy.Subscriber('/left_endeffector/wrench', WrenchStamped, wrench_cb, 'left')

    rospy.spin()
