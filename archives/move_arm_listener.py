#!/usr/bin/env python
# -*- coding:utf-8 -*-
# move_arm_listener.py
"""
topic: semi/move_arm_chatter
message: geometry_msgs/PoseStamped
function: PoseStampedで指定された位置へ右腕を移動させる。

How to use
---

    $ rosrun jsk_2015_05_baxter_apc move_arm_listener.py
    # in another terminal
    $ rosrun jsk_2015_05_baxter_apc test_move_arm.py

"""
import struct

import rospy

from geometry_msgs.msg import (
    PoseStamped,
    Pose,
    Point,
    Quaternion,
)
from std_msgs.msg import Header

import baxter_interface

from baxter_core_msgs.srv import (
    SolvePositionIK,
    SolvePositionIKRequest,
)


class MoveArmListener(object):
    def __init__(self):
        rospy.init_node('move_arm_listener')
        rospy.Subscriber('semi/move_right_arm', PoseStamped, self._rarm_callback)
        rospy.Subscriber('semi/move_left_arm', PoseStamped, self._larm_callback)

    def _rarm_callback(self, pose_stamp):
        self._solve_ik(limb_nm='right', pose_stamp=pose_stamp)

    def _larm_callback(self, pose_stamp):
        self._solve_ik(limb_nm='left', pose_stamp=pose_stamp)

    def _solve_ik(self, limb_nm, pose_stamp):
        ns = "ExternalTools/" + limb_nm + "/PositionKinematicsNode/IKService"
        iksvc = rospy.ServiceProxy(ns, SolvePositionIK)
        ikreq = SolvePositionIKRequest()

        ikreq.pose_stamp.append(pose_stamp)  # store poses to solve ik
        try:
            rospy.wait_for_service(ns, 5.)
            resp = iksvc(ikreq)
        except (rospy.ServiceException, rospy.ROSException) as e:
            rospy.logerr("Service call failed: %s" % (e,))
            return 1

        # Check if result valid, and type of seed ultimately used to get solution
        # convert rospy's string representation of uint8[]'s to int's
        resp_seeds = struct.unpack('<%dB' % len(resp.result_type),
                                resp.result_type)
        if (resp_seeds[0] == resp.RESULT_INVALID):
            print("INVALID POSE - No Valid Joint Solution Found.")
        else:
            seed_str = {
                        ikreq.SEED_USER: 'User Provided Seed',
                        ikreq.SEED_CURRENT: 'Current Joint Angles',
                        ikreq.SEED_NS_MAP: 'Nullspace Setpoints',
                    }.get(resp_seeds[0], 'None')
            print("SUCCESS - Valid Joint Solution Found from Seed Type: %s" %
                (seed_str,))
            # Format solution into Limb API-compatible dictionary
            limb_joints = dict(zip(resp.joints[0].name, resp.joints[0].position))
            print("\nIK Joint Solution:\n", limb_joints)
            print("------------------")
            print("Response Message:\n", resp)

        # leave log information about head pose
        log = rospy.get_name() + ": I heard {}".format(pose_stamp)
        rospy.loginfo(log)

        # actually move arm
        limb = baxter_interface.Limb(limb_nm)
        lj = limb.joint_names()
        if len(resp.joints[0].position) !=0:
            joint_command = {jt_nm: resp.joints[0].position[i]
                    for i, jt_nm in enumerate(lj)}
            limb.move_to_joint_positions(joint_command)


def main():
    m = MoveArmListener()
    rospy.spin()


if __name__ == '__main__':
    main()
