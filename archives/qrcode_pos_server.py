#!/usr/bin/python
#-*- coding:utf-8 -*-
import pprint
import time

import roslib
roslib.load_manifest('jsk_2015_05_baxter_apc')
import rospy

import baxter_interface

from zbar_ros.msg import Marker
from jsk_2015_05_baxter_apc.srv import *
from jsk_2015_05_baxter_apc.msg import *

from std_msgs.msg import Header
from geometry_msgs.msg import (
    PoseStamped,
    Pose,
    Point,
    Quaternion
)

baxter_arm_angles = [
    dict(zip(['right_s0', 'right_s1', 'right_w0', 'right_w1', 'right_w2', 'right_e0', 'right_e1'],
             [0.6864564017944337, -0.16336895372314456, -0.8471408891418457, 0.3627864559204102, 0.777728258569336, 0.5414952175048828, 0.556835025366211]
         )),
    dict(zip(['right_s0', 'right_s1', 'right_w0', 'right_w1', 'right_w2', 'right_e0', 'right_e1'],
             [0.7554855371704102, -0.3179175179260254, -1.3468351302246095, 0.6178107616149903, 0.8421554515869141, 0.6757185362915039, 0.9200049764831544]
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

    # print("joint_names()-->")
    # print(rj)
    # rospy.Rate(1)
    for angle in baxter_arm_angles:
        # print('right joint_angles=>')
        # print(right.joint_angles())
        right.move_to_joint_positions(angle, timeout=20.0)

        # time.sleep(5)
        # for i in range(5):
        #     rospy.sleep()
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
