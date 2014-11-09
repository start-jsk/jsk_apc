#!/usr/bin/env python
#-*- coding:utf-8 -*-
"""
move_arm.pyのテスト用
"""
import rospy

from geometry_msgs.msg import (
    PoseStamped,
    Pose,
    Point,
    Quaternion,
)
from std_msgs.msg import Header


def talker():
    rospy.init_node('joint_angle_talker')
    pub = rospy.Publisher('semi/joint_angle_chatter', PoseStamped)

    right_pose = []
    hdr = Header(stamp=rospy.Time.now(), frame_id='base')

    position=Point(
            x=0.9743684236857688,
            y=-0.400063802027193,
            z=0.45121855877314805,
            )
    orientation=Quaternion(
            x=0.05532824006190543,
            y=0.8002393257489635,
            z=0.09079879412889569,
            w=0.5901791137961719,
            )
    pose = Pose(position=position, orientation=orientation)
    right_pose.append(PoseStamped(header=hdr, pose=pose))

    position = Point(
            x=1.1257550223179955,
            y=-0.2874995035594442,
            z=0.4497770857427736)
    orientation = Quaternion(
            x=-0.4546518238906712,
            y=0.47559726095679095,
            z=-0.5269588476157249,
            w=0.5379715023333403)
    pose = Pose(position=position, orientation=orientation)
    right_pose.append(PoseStamped(header=hdr, pose=pose))

    while not rospy.is_shutdown():
        log = "hello, baxter. I'm talking right_arm_pose: {}".format(right_pose[0])
        rospy.loginfo(log)
        pub.publish(right_pose[0])
        rospy.sleep(1.)


if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
