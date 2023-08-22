#!/usr/bin/env python
import rospy
import rosnode
import rostopic
from std_msgs.msg import Bool
import os
import xmlrpclib
import time
import socket
import rosgraph


HEADER = '\033[95m'
OKBLUE = '\033[94m'
OKGREEN = '\033[92m'
WARNING = '\033[93m'
INFO = '\033[36m'
FAIL = '\033[91m'
ENDC = '\033[0m'
BOLD = '\033[1m'
UNDERLINE = '\033[4m'

is_topic_published=False
first_time_callback=True
def success_print(msg):
    print(OKGREEN, "[OK]   : " , msg, ENDC)

def warning_print(msg):
    print(WARNING , "[WARN] : " , msg , ENDC)

def fail_print(msg):
    print(FAIL, "[FAIL] : " , msg , ENDC)

def index_print(msg):
    print(OKBLUE+BOLD+msg, ENDC)

def check_param(param_name, expected, must=False):
    if not rospy.has_param(param_name):
        if must:
            fail_print("Paremeter " + param_name + " doesn't exist" )
        else:
            warning_print("Parameter " + param_name + " doesn't exist" )
        return

    target_param = rospy.get_param(param_name)
    if target_param == expected:
        success_print("Parameter " + param_name + " is " + str(expected))
    else:
        fail_print("Parameter " + param_name + " is " + str(target_param) + ". Doesn't match with exepcted value : " + str(expected))

def check_publish_callback(msg, callback_args):
    global is_topic_published, first_time_callback
    is_topic_published = True

    if callback_args["print_data"] and first_time_callback:
        print(INFO+"---")
        first_time_callback = False
        field_filter_fn = rostopic.create_field_filter(False, True)
        callback_echo = rostopic.CallbackEcho(callback_args["topic"], None, plot=False,
                                              filter_fn=None,
                                              echo_clear=False, echo_all_topics=False,
                                              offset_time=False, count=None,
                                              field_filter_fn=field_filter_fn)
        callback_echo.callback(msg, callback_args)
        print(ENDC)

def check_publishers(topic_name):
    master = rosgraph.Master('/rostopic')
    state = master.getSystemState()

    pubs, subs, _ = state
    subs = [x for x in subs if x[0] == topic]
    pubs = [x for x in pubs if x[0] == topic]
    return pubs

def check_topic(topic_name,
                print_data = False,
                timeout = 1):
    print(HEADER+BOLD+"=== Check " +topic_name + " ===" +ENDC)
    global is_topic_published, first_time_callback
    first_time_callback = True

    topic_list = rospy.get_published_topics()
    for topic in topic_list:
        if topic[0] == topic_name:
            break
    else:
        fail_print("No Publishers for " + topic_name + " found")
        return

    is_topic_published = False
    msg_class, real_topic, msg_eval = rostopic.get_topic_class(topic_name, blocking=True)

    s = rospy.Subscriber(topic_name, msg_class,
                         check_publish_callback,
                         {"topic":topic_name, "print_data":print_data, "type_infomation":None})
    start_time = rospy.Time.now()
    rate = rospy.Rate(1)
    while not rospy.is_shutdown():
        now = rospy.Time.now()
        diff = (now - start_time).to_sec()
        if diff > timeout:
            break
        if is_topic_published:
            break
        rate.sleep()
    try:
        if is_topic_published:
            success_print("%s is published" % (topic_name))
        else:
            fail_print("%s is not published" % (topic_name))
        return is_topic_published
    finally:
        is_topic_published = False
        s.unregister()

def check_node(target_node_name, needed, sub_success="", sub_fail=""):
    nodes = rosnode.get_node_names()
    if target_node_name in nodes:
        if needed:
            success_print("Node " + target_node_name + " exists")
            if sub_success:
                print(OKGREEN+"    "+sub_success,ENDC)
        else:
            fail_print("Node " + target_node_name + " exists unexpecetedly. This should be killed with rosnode kill")
            if sub_fail:
                print(FAIL+"    "+sub_fail,ENDC)
    else:
        if needed:
            fail_print("Node " + target_node_name + " doesn't exists. This node is NEEDED")
            if sub_fail:
                print(FAIL+"    "+sub_fail,ENDC)
        else:
            success_print("Node " + target_node_name + " doesn't exists")
            if sub_success:
                print(OKGREEN+"    "+sub_success,ENDC)

def check_rosmaster():
    try:
        nodes = rosnode.get_node_names()
    except rosnode.ROSNodeIOException:
        fail_print("ROS MASTER doesn't exist or can't communicate")
        exit(-1)
    else:
        success_print("Able to Communate With ROS MASTER")

def check_vacuum(arm):
    topic_name = "/vacuum_gripper/limb/"+arm
    print(HEADER+BOLD+"=== Check " +topic_name + " ===" +ENDC)
    print(INFO,"Start " + arm + " Vacuum for 5 seconds...")
    pub = rospy.Publisher(topic_name, Bool, queue_size=1)
    pub.publish(Bool(True))
    time.sleep(5)

    print(INFO,"Stop " + arm + " Vacuum")
    pub.publish(Bool(False))

from baxter_core_msgs.srv import (
    ListCameras,
)

def check_cameras():
    print(HEADER+BOLD+"=== Check CAMERAS ===" +ENDC)
    ls = rospy.ServiceProxy('cameras/list', ListCameras)
    print(INFO+" wait for server..." +ENDC)
    try:
        rospy.wait_for_service('cameras/list', timeout=5)
    except rospy.exceptions.ROSException:
        fail_print("cameras/list doesn't seems to appear")
        exit()
    resp = ls()
    if len(resp.cameras):
        # Find open (publishing) cameras
        master = rosgraph.Master('/rostopic')
        resp.cameras
        cam_topics = dict([(cam, "/cameras/%s/image" % cam)
                               for cam in resp.cameras])
        open_cams = dict([(cam, False) for cam in resp.cameras])
        try:
            topics = master.getPublishedTopics('')
            for topic in topics:
                for cam in resp.cameras:
                    if topic[0] == cam_topics[cam]:
                        open_cams[cam] = True
        except socket.error:
            raise ROSTopicIOException("Cannot communicate with master.")


    fail=False
    for arm in ["right", "left"]:
        if arm+"_hand_camera" in resp.cameras and open_cams[arm+"_hand_camera"]:
            success_print(arm + " camera was found and open!")
        else:
            fail_print(arm + " camera was not found...")
            fail=True
            break

    if fail:
        warning_print("The open cameras are below")
        for cam in resp.cameras:
            print(INFO+"    "+cam+ ("  -  (open)" if open_cams[cam] else "")+ENDC % ())
