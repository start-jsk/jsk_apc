#!/usr/bin/env python
try:
  from jsk_rviz_plugins.msg import *
except:
  import roslib;roslib.load_manifest("jsk_rviz_plugins")
  from jsk_rviz_plugins.msg import *

from std_msgs.msg import ColorRGBA, Float32, String
from jsk_2014_picking_challenge.msg import ObjectRecognition
import rospy
import math
rospy.init_node("debug_info_overlay_text")

text_pub = rospy.Publisher("text_sample", OverlayText)

base_top = 800
common_width = 1000

def publish_overlay_text(text_pub, text_content, width, height, left, top, fg_color = ColorRGBA(25 / 255.0, 1.0, 240.0 / 255.0, 1.0), bg_color = ColorRGBA(0.0, 0.0, 0.0, 0.2)):
    text = OverlayText()
    text.width = width
    text.height = height
    text.left = left
    text.top = top
    text.text_size = 18
    text.line_width = 2
    text.font = "DejaVu Sans Mono"
    text.fg_color = fg_color
    text.bg_color = bg_color
    text.text = text_content
    text_pub.publish(text)

def vacuum_state_cb(msg, callback_args):
    text = callback_args["arm"] + " vacuum : " + ("ON" if msg.data == "ON" else "OFF")
    width = common_width
    height = 50
    left = 0
    top = base_top + 100 if callback_args["arm"] == "left" else base_top + 100 + height
    fg_color = ColorRGBA(25 / 255.0, 1.0, 240.0 / 255.0, 1.0) if msg.data == "OFF" else ColorRGBA(1.0, 0.0, 0.0, 1.0)
    publish_overlay_text(callback_args["pub"], text, width, height ,left, top, fg_color)

def recog_cb(msg, callback_args):
    matched = msg.matched
    probability = msg.probability
    text = callback_args["arm"] + " verify result : " + matched + "(" + str(probability) +")"
    width = common_width
    height = 70
    left = 0
    top = base_top+200 if callback_args["arm"] == "left" else base_top+200 + height
    publish_overlay_text(callback_args["pub"], text, width, height ,left, top)

def tweet_cb(msg, callback_args):
    text = "Tweet Now : " +  msg.data
    width = common_width
    height = 50
    left = 0
    top = base_top+350
    fg_color = ColorRGBA(25 / 255.0, 1.0, 20.0 / 255.0, 1.0)
    publish_overlay_text(callback_args["pub"], text, width, height ,left, top, fg_color)

if __name__ == "__main__":
    left_vacuum_pub = rospy.Publisher("left_vacuum_state", OverlayText)
    right_vacuum_pub = rospy.Publisher("right_vacuum_state", OverlayText)
    left_recog_pub = rospy.Publisher("left_recog_state", OverlayText)
    right_recog_pub = rospy.Publisher("right_recog_state", OverlayText)
    tweet_pub = rospy.Publisher("tweet_text", OverlayText)


    rospy.Subscriber('/vacuum_gripper/limb/left/state', String, vacuum_state_cb, {"arm":"left", "pub":left_vacuum_pub})
    rospy.Subscriber('/vacuum_gripper/limb/right/state', String, vacuum_state_cb, {"arm":"right", "pub":right_vacuum_pub})

    rospy.Subscriber('/left_process/object_verification/output', ObjectRecognition, recog_cb, {"arm":"left", "pub":left_recog_pub})
    rospy.Subscriber('/right_process/object_verification/output', ObjectRecognition, recog_cb, {"arm":"right", "pub":right_recog_pub})

    rospy.Subscriber('/tweet', String, tweet_cb, {"pub":tweet_pub})

    rospy.spin()
