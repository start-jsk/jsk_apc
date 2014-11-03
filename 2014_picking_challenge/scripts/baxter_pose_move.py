#!/usr/bin/env python
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


def callback(pose):
    ns = "ExternalTools/" + 'right' + "/PositionKinematicsNode/IKService"
    iksvc = rospy.ServiceProxy(ns, SolvePositionIK)
    ikreq = SolvePositionIKRequest()

    ikreq.pose_stamp.append(pose)
    try:
        rospy.wait_for_service(ns, 5.0)
        resp = iksvc(ikreq)
    except (rospy.ServiceException, rospy.ROSException), e:
        rospy.logerr("Service call failed: %s" % (e,))
        return 1

    # Check if result valid, and type of seed ultimately used to get solution
    # convert rospy's string representation of uint8[]'s to int's
    resp_seeds = struct.unpack('<%dB' % len(resp.result_type),
                               resp.result_type)
    if (resp_seeds[0] != resp.RESULT_INVALID):
        seed_str = {
                    ikreq.SEED_USER: 'User Provided Seed',
                    ikreq.SEED_CURRENT: 'Current Joint Angles',
                    ikreq.SEED_NS_MAP: 'Nullspace Setpoints',
                   }.get(resp_seeds[0], 'None')
        print("SUCCESS - Valid Joint Solution Found from Seed Type: %s" %
              (seed_str,))
        # Format solution into Limb API-compatible dictionary
        limb_joints = dict(zip(resp.joints[0].name, resp.joints[0].position))
        print "\nIK Joint Solution:\n", limb_joints
        print "------------------"
        print "Response Message:\n", resp
    else:
        print("INVALID POSE - No Valid Joint Solution Found.")

    log = rospy.get_name() + ": I heard {}".format(pose)
    rospy.loginfo(log)

    right = baxter_interface.Limb('right')
    rj = right.joint_names()
    for i, joint_name in enumerate(rj):
        current_position = right.joint_angle(joint_name)
        print "*", current_position, resp.joints[0].position[i]

        joint_command = {joint_name: resp.joints[0].position[i]}
        right.set_joint_positions(joint_command)
        rospy.sleep(0.1)

    print right.endpoint_pose()



def listener():
    rospy.init_node('baxter_pose_listener', anonymous=True)

    # initialize arm position
    right = baxter_interface.Limb('right')
    right.move_to_neutral()

    rospy.Subscriber('baxter_pose_chatter', PoseStamped, callback)
    rospy.spin()


if __name__ == '__main__':
    listener()
