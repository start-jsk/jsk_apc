#!/usr/bin/env python
#-*- coding:utf-8 -*-
import math
import pprint

import rospy

import baxter_interface

bcolors = dict(
   header='\033[95m',
   okblue='\033[94m',
   okgreen='\033[92m',
   endc='\033[0m')

def get_arm_position():
    rospy.init_node('get_arm_status', anonymous=True)

    for lim in ['left', 'right']:
        print bcolors['header'] + "### {0} arm".format(lim) + bcolors['endc']
        lim = baxter_interface.Limb(lim)
        #
        print bcolors['okgreen'] + "==> end_point_pose" + bcolors['endc']
        pprint.pprint(lim.endpoint_pose())
        #
        print bcolors['okgreen'] + "==> joint_angle" + bcolors['endc']
        print bcolors['okblue'] + "==> [rad]" + bcolors['endc'] 
        joint_angles_rad = lim.joint_angles()
        print joint_angles_rad.keys()
        print joint_angles_rad.values()
        print bcolors['okblue'] + "==> [deg]" + bcolors['endc'] 
        joint_angles_deg = {}
        if joint_angles_rad != {}:
            for jt_nm, jt_angle in joint_angles_rad.items():
                joint_angles_deg[jt_nm] = jt_angle / math.pi * 180
        print joint_angles_deg.keys()
        print joint_angles_deg.values()

if __name__ == '__main__':
    get_arm_position()
