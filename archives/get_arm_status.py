#!/usr/bin/env python
#-*- coding:utf-8 -*-
"""
function: get arm joints and end effector positions

How to use
---

    $ rosrun jsk_2015_05_baxter_apc get_arm_status.py

"""

import math
import pprint

import rospy

import baxter_interface


def cprint(sentence, color):
    colors = dict(pink='\033[95m',
                  blue='\033[94m',
                  green='\033[92m')
    if color not in colors:
        raise ValueError('Unexpected color input')
    endc = '\033[0m'
    print(colors[color] + sentence + endc)


def get_arm_position():
    rospy.init_node('get_arm_status', anonymous=True)

    for limb_nm in ['left', 'right']:
        cprint('### {0} arm'.format(limb_nm), 'pink')
        limb = baxter_interface.Limb(limb_nm)
        # end point pose
        cprint('==> end_point_pose', 'green')
        pprint.pprint(limb.endpoint_pose())
        # joint angles [rad]
        cprint('==> joint_angle', 'green')
        cprint('==> [rad]', 'blue')
        joint_angles_rad = limb.joint_angles()
        print(limb.joint_names())
        print([joint_angles_rad[jt_nm] for jt_nm in limb.joint_names()])
        # joint angles [deg]
        cprint('==> [deg]', 'blue')
        joint_angles_deg = {}
        if joint_angles_rad != {}:
            for jt_nm, jt_angle in joint_angles_rad.items():
                joint_angles_deg[jt_nm] = jt_angle / math.pi * 180
        print(limb.joint_names())
        print([joint_angles_deg[jt_nm] for jt_nm in limb.joint_names()])


if __name__ == '__main__':
    get_arm_position()
