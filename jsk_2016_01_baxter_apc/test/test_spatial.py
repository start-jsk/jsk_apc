#!/usr/bin/env python
PKG = 'jsk_2016_01_baxter_apc'

import sys
import os
home = os.path.expanduser("~")
sys.path.insert(0,home + '/ros/indigo/src/start-jsk/jsk_apc/jsk_2016_01_baxter_apc')
import unittest
import rospy
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointField 
from jsk_recognition_msgs.msg import BoundingBox
from geometry_msgs.msg import Vector3,Point,Pose, Quaternion, TransformStamped 
from std_msgs.msg import Header 
import struct
import tf2_ros 

from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud 
#from node_scripts.dist2shelf import dist2bbox
from node_scripts.pipeline import Pipeline 

from test.helper import separator 


class Dist2Bbox(unittest.TestCase):
    """
    two test setups
    one that has a orientation identical to the origin
    the other has a different orientation
    """
    def setUp(self):
        # init TF listener
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)

        #internal transformations for two test setups
        self.tfsetUp()
        
        # initialize clouds
        self.cloud = point_cloud2.PointCloud2() 
        self.cloud.header = Header(frame_id='cloud_base')
        self.cloud.fields = [
            PointField(name='x',offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1)
            ]
        self.cloud.point_step = 4*4
        self.pipeline = Pipeline()

    def tfsetUp(self):
        t1 = self.tfBuffer.lookup_transform('base','cloud_base',rospy.Time.now(), rospy.Duration(1.0))
        self.failUnlessEqual(t1.transform.translation.x, 0)
        self.failUnlessEqual(t1.transform.translation.y, 0)
        self.failUnlessEqual(t1.transform.translation.z, 1)
        self.failUnlessEqual(t1.transform.rotation.x, 0)
        self.failUnlessEqual(t1.transform.rotation.y, 0)
        self.failUnlessEqual(t1.transform.rotation.z, 0)
        self.failUnlessEqual(t1.transform.rotation.w, 1)

        t3 = self.tfBuffer.lookup_transform('base','bbox_center_orient',rospy.Time.now(), rospy.Duration(1.0))
        self.failUnlessEqual(t3.transform.translation.x, 1)
        self.failUnlessEqual(t3.transform.translation.y, 0)
        self.failUnlessEqual(t3.transform.translation.z, 0)
        self.failUnlessEqual(t3.transform.rotation.x, 0.7071067811)
        self.failUnlessEqual(t3.transform.rotation.y, 0)
        self.failUnlessEqual(t3.transform.rotation.z, 0)
        self.failUnlessEqual(t3.transform.rotation.w, 0.7071067811)
        
        self.bbox = BoundingBox()  
        self.bbox.header = Header(frame_id='base') 
        self.bbox.pose = Pose(
            Point(
                x=t3.transform.translation.x,
                y=t3.transform.translation.y,
                z=t3.transform.translation.z), 
            Quaternion(
                x=t3.transform.rotation.x, 
                y=t3.transform.rotation.y, 
                z=t3.transform.rotation.z,
                w=t3.transform.rotation.w))

    def test_dist2bbox(self):
        transform = self.tfBuffer.lookup_transform('bbox_center_orient', 'cloud_base', rospy.Time.now())
        self.bbox.dimensions = Vector3(x=2, y=1, z=1)

        # coordinates relative to cloud_base
        points = [
            1, 0, -1, 0xff0000,
            1, 0, -0.5, 0x000000
            ]
        self.cloud.data = struct.pack('3fI3fI', *points)
        self.cloud.height = 1
        self.cloud.width = 2
        self.cloud.row_step = self.cloud.point_step * self.cloud.width

        dists = self.pipeline._get_spatial(self.cloud, self.bbox, transform, 'x')
        self.failUnlessEqual(dists[0], [0.5, 0])


if __name__ == '__main__':
    rospy.init_node("test_dist2bbox")
    import rostest
    rostest.rosrun(PKG, 'test_dist2bbox', Dist2Bbox)
