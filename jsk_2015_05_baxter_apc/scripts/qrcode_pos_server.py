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
             [0.35856800875854494, -0.23546605067138673, 0.11581554935302735, 0.14802914586181642, 2.8907867914672853, 0.45482530308837893, 1.0611312088073732]
         )),
    dict(zip(['right_s0', 'right_s1', 'right_w0', 'right_w1', 'right_w2', 'right_e0', 'right_e1'],
[0.7873156384826661, -0.681470964239502, -1.089126358154297, -0.3819612157470703, 1.742602173046875, -0.4816699668457032, 0.5506991022216797]
)),
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

    print("joint_names()-->")
    print(rj)
    # rospy.Rate(1)
    for angle in baxter_arm_angles:
        print('right joint_angles=>')
        print(right.joint_angles())
        right.move_to_joint_positions(angle)

        global marker_id
        # rospy.sleep()
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
    s = rospy.Service('qrcode_pos', QrStampsrv, handle_qrcode_pos)
    print("Ready to return qrcode_pos")
    rospy.Subscriber('markers', Marker, callback)
    rospy.spin()

if __name__ == '__main__':
    qrcode_pos_server()
