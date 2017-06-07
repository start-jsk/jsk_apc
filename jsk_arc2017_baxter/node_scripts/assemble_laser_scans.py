#!/usr/bin/env python

from jsk_topic_tools import ConnectionBasedTransport
from laser_assembler.srv import AssembleScans2
import rospy
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import PointCloud2


class AssembleLaserScans(ConnectionBasedTransport):

    def __init__(self):
        super(AssembleLaserScans, self).__init__()
        self.pub = self.advertise('~output', PointCloud2, queue_size=1)
        self.interval = rospy.get_param('~assemble_interval', 10.0)
        rospy.wait_for_service('assemble_scans2')
        self.srv = rospy.ServiceProxy('assemble_scans2', AssembleScans2)

    def subscribe(self):
        self.sub = rospy.Subscriber('scan', LaserScan, self.callback)

    def unsubscribe(self):
        self.sub.unregister()

    def callback(self, msg):
        try:
            now = rospy.Time.now()
            response = self.srv(now - rospy.Duration(self.interval), now)
            rospy.loginfo(
                'Got cloud with %u points' % len(response.cloud.data))
            self.pub.publish(response.cloud)
        except rospy.ServiceException as e:
            rospy.logerr('Service call failed: %s' % e)


if __name__ == '__main__':
    rospy.init_node('assemble_laser_scans')
    app = AssembleLaserScans()
    rospy.spin()
