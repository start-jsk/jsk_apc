#!/usr/bin/env python
import rospy

from geometry_msgs.msg import (
    PoseStamped,
    Pose,
    Point,
    Quaternion,
)
from std_msgs.msg import Header


def talker():
    rospy.init_node('semi/joint_angle_talker')
    pub = rospy.Publisher('semi/joint_angle_chatter', PoseStamped)

    hdr = Header(stamp=rospy.Time.now(), frame_id='base')
    right_pose = PoseStamped(
        header=hdr,
        pose=Pose(
            position=Point(
                x=0.9543684236857687,
                y=-0.2900063802027193,
                z=0.40121855877314805,
            ),
            orientation=Quaternion(
                x=0.05532824006190543,
                y=0.8002393257489635,
                z=0.09079879412889569,
                w=0.5901791137961719,
            ),
        ),
    )

    while not rospy.is_shutdown():
        log = "hello, baxter. I'm talking right_arm_pose: {}".format(right_pose)
        rospy.loginfo(log)
        pub.publish(right_pose)
        rospy.sleep(1.0)


if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
