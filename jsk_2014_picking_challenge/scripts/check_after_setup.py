#!/usr/bin/env python
from check_common import *

def check_vacuum(arm):
    topic_name = "/vacuum_gripper/limb/"+arm
    print HEADER+BOLD+"=== Check " +topic_name + " ===" +ENDC
    print INFO,"Start " + arm + " Vacuum for 5 seconds..."
    pub = rospy.Publisher(topic_name, Bool, queue_size=1)
    pub.publish(Bool(True))
    time.sleep(5)

    print INFO,"Stop " + arm + " Vacuum"
    pub.publish(Bool(False))

if __name__ == "__main__":

    print HEADER+"= Check Start APC Setup =", ENDC

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

    index_print("== Check PARAMETERS ==")
    check_param("/left_process/target", "")
    check_param("/right_process/target", "")
    check_param("/left_process/state", "")
    check_param("/right_process/state", "")

    index_print("== Check OTHER ==")
    check_vacuum("left")
    check_vacuum("right")
