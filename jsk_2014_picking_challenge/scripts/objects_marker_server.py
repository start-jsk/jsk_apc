#!/usr/bin/env python

"""
Copyright (c) 2011, Willow Garage, Inc.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the Willow Garage, Inc. nor the names of its
      contributors may be used to endorse or promote products derived from
      this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES LOSS OF USE, DATA, OR PROFITS OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""

import roslib; roslib.load_manifest("interactive_markers")
import rospy
import copy

from interactive_markers.interactive_marker_server import *
from interactive_markers.menu_handler import *
from geometry_msgs.msg import *
from visualization_msgs.msg import *
from tf import transformations
from tf import TransformListener
from jsk_2014_picking_challenge.msg import SetObjectPositionArray, WorkOrderArray
import rospkg
import yaml
server = None
menu_handler = MenuHandler()
counter = 0
objects_name_list = None
base_frame_id = "/base"

def frameCallback( msg ):
    global counter, objects_name_list
    counter+=1

    if objects_name_list:
        for object_name in objects_name_list:
            int_marker = server.get(object_name)
            if int_marker:
                cur_pose = int_marker.pose
                quat = transformations.quaternion_about_axis(2 * 3.14159286 * (1.0 / 1000), (0,0,1))
                quat = transformations.quaternion_multiply([cur_pose.orientation.x, cur_pose.orientation.y, cur_pose.orientation.z, cur_pose.orientation.w], quat)
                cur_pose.orientation.x = quat[0]
                cur_pose.orientation.y = quat[1]
                cur_pose.orientation.z = quat[2]
                cur_pose.orientation.w = quat[3]
                server.setPose(int_marker.name, cur_pose);
        server.applyChanges()

def processFeedback( feedback ):
    server.applyChanges()

def makeTargetObject( msg, object_name ):
    marker = Marker()

    marker.type = Marker.MESH_RESOURCE
    marker.scale.x = msg.scale * 2
    marker.scale.y = msg.scale * 2
    marker.scale.z = msg.scale * 2
    marker.mesh_resource = "package://jsk_2014_picking_challenge/meshes/" + object_name +"/" + object_name + ".dae"
    marker.mesh_use_embedded_materials = True
    return marker

def makeBoardObject( msg , object_name):
    marker = Marker()
    marker.type = Marker.CUBE
    marker.scale.x = 1
    marker.scale.y = 5
    marker.scale.z = 0.1
    if object_name == "left_board":
        marker.color.r = 1
        marker.color.g = 0.2
        marker.color.b = 0.2
        marker.color.a = 0.5
    else:
        marker.color.r = 0.2
        marker.color.g = 0.2
        marker.color.b = 1
        marker.color.a = 0.5
        
    return marker

def makeTargetObjectControl( msg, object_name ):
    control =  InteractiveMarkerControl()
    control.always_visible = True
    control.markers.append( makeTargetObject(msg, object_name) )
    msg.controls.append( control )
    return control

def makeBoardObjectControl( msg, object_name ):
    control =  InteractiveMarkerControl()
    control.always_visible = True
    control.markers.append( makeBoardObject(msg, object_name) )
    msg.controls.append( control )
    return control

def makeInteractiveBoardObject( object_name, position, quaternion):
    global base_frame_id
    int_marker = InteractiveMarker()
    int_marker.header.frame_id = base_frame_id
    int_marker.pose.position = position
    int_marker.pose.orientation = quaternion
    int_marker.scale = 1

    makeBoardObjectControl(int_marker, object_name)
    int_marker.controls[0].interaction_mode = InteractiveMarkerControl.NONE

    int_marker.name  = object_name 
    int_marker.description = object_name
    server.insert(int_marker, processFeedback)
    

def make6DofMarker( object_name, position, quaternion ):
    global base_frame_id
    int_marker = InteractiveMarker()
    int_marker.header.frame_id = base_frame_id
    int_marker.pose.position = position
    int_marker.pose.orientation = quaternion
    int_marker.scale = 1

    makeTargetObjectControl(int_marker, object_name)
    int_marker.controls[0].interaction_mode = InteractiveMarkerControl.MOVE_3D

    int_marker.name  = object_name 
    int_marker.description = object_name
    server.insert(int_marker, processFeedback)

def setObjectPoses(msg):
    for target_object in msg.objects:
        int_marker = server.get(target_object.object_name)
        if int_marker:
            tl = TransformListener()
            transed_point = tl.transformPoint(base_frame_id, target_object.position)
            cur_pose = int_marker.pose
            cur_pose.position = transed_point.point
            server.setPose(int_marker.name, cur_pose);        
    server.applyChanges()

def setWithWorkOrder(msg, callback_args):
    arm = callback_args["arm"]
    counter = 0
    for target_object in msg.array:
        int_marker = server.get(target_object.object)
        if int_marker:
            tl = TransformListener()
            cur_pose = Pose()
            cur_pose.position.x = -1 if arm == "left" else 1
            cur_pose.position.y = 0.5 + counter * 0.5
            cur_pose.position.z = 0
            cur_pose.orientation = int_marker.pose.orientation
            server.setPose(int_marker.name, cur_pose);
    server.applyChanges()    

if __name__=="__main__":
    rospy.init_node("target_object_marker_server")
    
    rospy.Subscriber('~set_pose', SetObjectPositionArray, setObjectPoses)

    rospy.Subscriber('/left/work_order_list', WorkOrderArray, setWithWorkOrder, {"arm":"left"})
    rospy.Subscriber('/right/work_order_list', WorkOrderArray, setWithWorkOrder, {"arm":"right"})

    rospy.Timer(rospy.Duration(0.01), frameCallback)
    server = InteractiveMarkerServer("target_object_marker_server")    
    rospack = rospkg.RosPack()
    with open(rospack.get_path("jsk_2014_picking_challenge") + "/data/object_list.yml", 'rb') as f:
        objects_name_list = yaml.load(f)
    
    for i,object_name in enumerate(objects_name_list):
        position = Point( - i / 5 - 5 , i % 5 - 2.5, 0)
        quat = transformations.quaternion_about_axis(2 * 3.14159286 * (i * 1.0 / len(objects_name_list)), (0,0,1))
        quaternion = Quaternion(quat[0], quat[1], quat[2], quat[3])
        make6DofMarker( object_name, position, quaternion )

    position = Point( -2.8, -0.8, 0)
    quat = transformations.quaternion_about_axis(3.14159286/2, (0,0,1))
    quaternion = Quaternion(quat[0], quat[1], quat[2], quat[3])
    makeInteractiveBoardObject("left_board", position, quaternion)
    position = Point( -2.8, 0.8,  0)
    makeInteractiveBoardObject("right_board", position, quaternion)

    server.applyChanges()
    rospy.spin()

