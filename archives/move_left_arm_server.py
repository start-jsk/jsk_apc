#!/usr/bin/env python
#-*- coding:utf-8 -*-
import roslib; roslib.load_manifest('jsk_2015_05_baxter_apc')

import struct
import rospy

from geometry_msgs.msg import (
    PoseStamped,
    Pose,
    Point,
    Quaternion,
)
from std_msgs.msg import Header
from sensor_msgs.msg import JointState
from jsk_2015_05_baxter_apc.srv import (
    MoveArm,
    MoveArmResponse,
)

import baxter_interface
from baxter_core_msgs.srv import (
    SolvePositionIK,
    SolvePositionIKRequest,
)

def handle_move_arm(pose_req):
    # calculate arm IK
    pose = pose_req.pose_arm
    ns = "ExternalTools/" + 'left' + "/PositionKinematicsNode/IKService"
    iksvc = rospy.ServiceProxy(ns, SolvePositionIK)
    ikreq = SolvePositionIKRequest()
    ikreq.pose_stamp.append(pose)

    # Check if service call is successfull
    try:
        rospy.wait_for_service(ns, 5.)
        resp = iksvc(ikreq)
    except (rospy.ServiceException, rospy.ROSException) as e:
        rospy.logerr("Service call failed: %s" % (e,))
        return False

    # Check if result valid, and type of seed ultimately used to get solution
    # convert rospy's string representation of uint8[]'s to int's
    resp_seeds = struct.unpack('<%dB' % len(resp.result_type),
                               resp.result_type)
    if (resp_seeds[0] == resp.RESULT_INVALID):
        # if the ik solution is invalid return false
        return MoveArmResponse(False)

    # print the input pose
    log = rospy.get_name() + ": I heard {}".format(pose)
    rospy.loginfo(log)

    # get arm info
    left = baxter_interface.Limb('left')
    rj = left.joint_names()
    # move the arm
    joint_command = {jt_nm: resp.joints[0].position[i]
            for i, jt_nm in enumerate(rj)}
    left.move_to_joint_positions(joint_command)
    # if the move is successful return True
    return MoveArmResponse(succeeded=True)


def move_arm_server():
    rospy.init_node('move_left_arm_server')
    s = rospy.Service('semi/move_left_arm', MoveArm, handle_move_arm)
    print("Ready to move left arm.")
    rospy.spin()


if __name__ == '__main__':
    move_arm_server()
