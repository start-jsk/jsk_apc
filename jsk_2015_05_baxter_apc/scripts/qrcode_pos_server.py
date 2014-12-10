#!/usr/bin/python
#-*- coding:utf-8 -*-
import pprint

import roslib
roslib.load_manifest('jsk_2014_picking_challenge')
import rospy

import baxter_interface

from zbar_ros.msg import Marker
from jsk_2014_picking_challenge.srv import *
from jsk_2014_picking_challenge.msg import *

from std_msgs.msg import Header
from geometry_msgs.msg import (
    PoseStamped,
    Pose,
    Point,
    Quaternion
)

baxter_arm_angles = [
    dict(zip(['right_s0', 'right_s1', 'right_w0', 'right_w1', 'right_w2', 'right_e0', 'right_e1'],
             [1.0688011127380372, 1.0492428577148438, 3.0587576875488285, 0.24697090656738283, 3.0595246779418948, -3.046869336456299, 1.0580632472351075]
         )),
    dict(zip(['right_s0', 'right_s1', 'right_w0', 'right_w1', 'right_w2', 'right_e0', 'right_e1'],
             [0.872068076916504, 1.0312185834777832, 3.0587576875488285, 0.6906748489562988, 3.0587576875488285, -3.0457188508666992, 1.2168302585998536]
         ))
]

marker_id = ""
def callback(mark):
    print(mark)
    global marker_id
    marker_id = mark.data
    print(marker_id)

def handle_qrcode_pos(req):
    qrstamps = QrStamp()
    right = baxter_interface.Limb('right')
    rj = right.joint_names()

    # print("joint_names()-->")
    # print(rj)
    # rospy.Rate(1)
    for angle in baxter_arm_angles:
        # print('right joint_angles=>')
        # print(right.joint_angles())
        right.move_to_joint_positions(angle)

        global marker_id
        hdr = Header(stamp=rospy.Time.now(), frame_id=marker_id[:5])
        lim = baxter_interface.Limb('right')
        pprint.pprint(lim.endpoint_pose())
        end_pose = lim.endpoint_pose()
        position = end_pose['position']
        orientation = end_pose['orientation']
        pose = Pose(position=position, orientation=orientation)
        right_pose = PoseStamped(header=hdr, pose=pose)
        qrstamps.qrcode_poses.append(right_pose)
        # rospy.sleep()
    return QrStampsrvResponse(qrstamps)

def qrcode_pos_server():
    rospy.init_node("qrcode_pos_server")
    s = rospy.Service('semi/qrcode_pos', QrStampsrv, handle_qrcode_pos)
    print("Ready to return qrcode_pos")
    rospy.Subscriber('markers', Marker, callback)
    rospy.spin()

if __name__ == '__main__':
    qrcode_pos_server()
