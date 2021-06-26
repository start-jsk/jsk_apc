#!/usr/bin/env python
from check_common import *


if __name__ == "__main__":

    print(HEADER+"= Check Start AFTER BOOT =", ENDC)

    index_print("== Check MASTER ==")
    check_rosmaster()

    rospy.init_node("check_after_boot")

    # index_print("== Check NODES ==")

    index_print("== Check TOPICS ==")
    check_topic("/robot/state", True, 3)

    # index_print("== Check PARAMETERS ==")

    index_print("== Check OTHER ==")
    check_cameras()
