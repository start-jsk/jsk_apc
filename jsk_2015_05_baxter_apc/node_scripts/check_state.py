#!/usr/bin/env python
from check_common import *

if __name__ == "__main__":

    print(HEADER+"= Check Start APC Setup =", ENDC)

    index_print("== Check MASTER ==")
    check_rosmaster()

    rospy.init_node("check_state")

    index_print("== Check NODES ==")
    check_node("/baxter_joint_trajectory", True, sub_success="baxter.launch launch")
    check_node("/kinect2_bridge", True, sub_success="baxter.launch launch")
    check_node("/kinect2_points_xyzrgb_highres", True, sub_success="baxter.launch launch")
    check_node("/robot_state_publisher", True, sub_success="setup.launch launch", sub_fail="setup.launch seems not appear")

    index_print("== Check TOPICS ==")
    check_topic("/robot/state", True, 3)
    check_topic("/vacuum_gripper/limb/left/state", True, 3)
    check_topic("/vacuum_gripper/limb/right/state", True, 3)
    check_topic("/left_hand/output", True, 3)
    check_topic("/right_hand/output", True, 3)
    check_topic("/kinect2/depth_highres/points", False, 5)

    index_print("== Check PARAMETERS ==")
    check_param("/left_process/target", "")
    check_param("/right_process/target", "")
    check_param("/left_process/state", "")
    check_param("/right_process/state", "")
    for param_type in ["trajectory", "goal"]:
        for arm in ["right", "left"]:
            for joint in ["e0", "e1", "s0", "s1", "w0", "w1", "w2"]:
                check_param("/baxter_joint_trajectory/"+"_".join([arm, joint, param_type]),  -1, True)

    index_print("== Check OTHER ==")
    check_cameras()
