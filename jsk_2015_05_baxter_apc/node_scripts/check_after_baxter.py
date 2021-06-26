#!/usr/bin/env python
from check_common import *

if __name__ == "__main__":

    print(HEADER+"= Check Start After Baxter.launch =", ENDC)

    index_print("== Check MASTER ==")
    check_rosmaster()

    rospy.init_node("check_after_baxter")

    index_print("== Check NODES ==")
    check_node("/baxter_joint_trajectory", True)
    check_node("/kinect2_bridge", True)
    check_node("/kinect2_points_xyzrgb_highres", True)

    index_print("== Check TOPICS ==")
    check_topic("/robot/state", True, 3)

    index_print("== Check PARAMETERS ==")
    for param_type in ["goal"]:
        for arm in ["right", "left"]:
            for joint in ["e0", "e1", "s0", "s1", "w0", "w1", "w2"]:
                check_param("/baxter_joint_trajectory/"+"_".join([arm, joint, param_type]),  -1, True)

