#!/usr/bin/env python
from check_common import *

def check_vacuum(arm):
    topic_name = "/vacuum_gripper/limb/"+arm
    print(HEADER+BOLD+"=== Check " +topic_name + " ===" +ENDC)
    pub = rospy.Publisher(topic_name, Bool, queue_size=1)
    msg = Bool()
    msg.data = True
    time.sleep(3)
    print(INFO,"Start " + arm + " Vacuum for 5 seconds...")
    pub.publish(msg)
    time.sleep(5)

    print(INFO,"Stop " + arm + " Vacuum")
    msg = Bool()
    msg.data = False
    pub.publish(msg)

if __name__ == "__main__":

    print(HEADER+"= Check Start APC Setup =", ENDC)

    index_print("== Check MASTER ==")
    check_rosmaster()

    rospy.init_node("check_state")

    index_print("== Check NODES ==")
    check_node("/baxter_joint_trajectory", True)
    check_node("/robot_state_publisher", True)
    check_node("/kinect2_bridge", True)
    check_node("/kinect2_points_xyzrgb_highres", True)

    index_print("== Check TOPICS ==")
    check_topic("/robot/state", True, 3)
    check_topic("/vacuum_gripper/limb/left/state", True, 3)
    check_topic("/vacuum_gripper/limb/right/state", True, 3)
    check_topic("/left_hand/output", True, 3)
    check_topic("/right_hand/output", True, 3)

    index_print("== Check PARAMETERS ==")

    index_print("== Check OTHER ==")
    check_vacuum("left")
    check_vacuum("right")
