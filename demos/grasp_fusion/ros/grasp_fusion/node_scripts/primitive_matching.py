#!/usr/bin/env python

from threading import Lock

import cv_bridge
import glob
import message_filters
import numpy as np
import os.path as osp
import rospy
import skimage.draw
import skimage.morphology
import tf
import yaml

from geometry_msgs.msg import Pose
from geometry_msgs.msg import PoseArray
from jsk_recognition_msgs.msg import BoundingBox
from jsk_recognition_msgs.msg import BoundingBoxArray
from jsk_topic_tools import ConnectionBasedTransport
from sensor_msgs.msg import Image
from std_srvs.srv import SetBool
from std_srvs.srv import SetBoolResponse

import grasp_fusion_lib
from grasp_fusion_lib.contrib.grasp_fusion.utils import get_primitives_poses


class PrimitiveMatching(ConnectionBasedTransport):

    def __init__(self):
        super(PrimitiveMatching, self).__init__()

        self.br = cv_bridge.CvBridge()

        self.instance_bg_label = rospy.get_param('~instance_bg_label')
        self.heightmap_frame = rospy.get_param('~heightmap_frame')
        # Size[m] of each height map pixel
        self.voxel_size = rospy.get_param('~voxel_size')
        self.cluster_tolerance = rospy.get_param('~cluster_tolerance', 0.02)
        self.cluster_max_size = rospy.get_param('~cluster_max_size')
        self.cluster_min_size = rospy.get_param('~cluster_min_size')
        self.prob_threshold = rospy.get_param('~prob_threshold', 0.5)
        self.reliable_pts_ratio = rospy.get_param('~reliable_pts_ratio', 0.25)
        # Directory of grasp primitives
        self.primitive_dir = rospy.get_param('~primitive_dir')

        self.primitives = []
        if not osp.isdir(self.primitive_dir):
            err = 'Input primitive_dir is not directory: %s' \
                % self.primitive_dir
            rospy.logfatal(err)
            rospy.signal_shutdown(err)
            return
        filenames = sorted(glob.glob(self.primitive_dir + "/*"))
        for fname in filenames:
            with open(fname) as f:
                self.primitives.append(yaml.load(f))

        # ROS publishers
        self.pubs_poses = []
        self.pubs_boxes = []
        for prim in self.primitives:
            self.pubs_poses.append(
                self.advertise('~output/' + prim['label'] + '/poses',
                               PoseArray, queue_size=1))
            self.pubs_boxes.append(
                self.advertise('~output/' + prim['label'] + '/boxes',
                               BoundingBoxArray, queue_size=1))
        self.pub_debug = self.advertise('~output/debug', Image, queue_size=1)

        self.lock = Lock()
        self.ignore_ins = False
        self.srv_ignore_ins = rospy.Service(
            '~ignore_instance', SetBool, self.ignore_ins_cb)

    def subscribe(self):
        self.sub_rgb = message_filters.Subscriber(
            '~input/rgb', Image, queue_size=1, buff_size=2**24
        )
        self.sub_depth = message_filters.Subscriber(
            '~input/depth', Image, queue_size=1, buff_size=2**24
        )
        self.sub_lbl_ins = message_filters.Subscriber(
            '~input/label_instance', Image, queue_size=1, buff_size=2**24
        )
        self.sub_prob_pinch_aff = message_filters.Subscriber(
            '~input/prob_pinch_affordance', Image,
            queue_size=1, buff_size=2**24
        )
        self.sub_prob_pinch_sole_aff = message_filters.Subscriber(
            '~input/prob_pinch_sole_affordance', Image,
            queue_size=1, buff_size=2**24
        )
        self.sub_prob_suc_aff = message_filters.Subscriber(
            '~input/prob_suction_affordance', Image,
            queue_size=1, buff_size=2**24
        )
        sync = message_filters.TimeSynchronizer([
            self.sub_rgb,
            self.sub_depth,
            self.sub_lbl_ins,
            self.sub_prob_pinch_aff,
            self.sub_prob_pinch_sole_aff,
            self.sub_prob_suc_aff
        ], queue_size=100)
        sync.registerCallback(self._cb)

    def unsubscribe(self):
        self.sub_depth.unregister()
        self.sub_lbl_ins.unregister()
        self.sub_prob_pinch_aff.unregister()
        self.sub_prob_pinch_sole_aff.unregister()
        self.sub_prob_suc_aff.unregister()

    def _cb(
        self,
        imgmsg,
        depthmsg,
        lbl_ins_msg,
        prob_pinch_aff_msg,
        prob_pinch_sole_aff_msg,
        prob_suc_aff_msg,
    ):
        img = self.br.imgmsg_to_cv2(imgmsg, desired_encoding='rgb8')
        depth = self.br.imgmsg_to_cv2(depthmsg, desired_encoding='32FC1')
        lbl_ins = self.br.imgmsg_to_cv2(
            lbl_ins_msg, desired_encoding='passthrough'
        )
        prob_pinch_aff = self.br.imgmsg_to_cv2(
            prob_pinch_aff_msg, desired_encoding='passthrough'
        )
        prob_pinch_sole_aff = self.br.imgmsg_to_cv2(
            prob_pinch_sole_aff_msg, desired_encoding='passthrough'
        )
        prob_suc_aff = self.br.imgmsg_to_cv2(
            prob_suc_aff_msg, desired_encoding='passthrough'
        )

        with self.lock:
            if self.ignore_ins:
                lbl_ins = np.ones((lbl_ins.shape[0], lbl_ins.shape[1]),
                                  dtype=lbl_ins.dtype)

        prim_posess = get_primitives_poses(
            self.primitives,
            depth,
            [prob_pinch_aff, prob_pinch_sole_aff, prob_suc_aff],
            ['pinch', 'pinch_sole', 'suction'],
            self.cluster_tolerance,
            self.cluster_max_size,
            self.cluster_min_size,
            voxel_size=self.voxel_size,
            instance_label=lbl_ins,
            instance_bg_label=self.instance_bg_label,
            prob_threshold=self.prob_threshold,
            reliable_pts_ratio=self.reliable_pts_ratio,
        )

        # Correction values for padding in get_heightmap
        corr_x_m = 10 * self.voxel_size
        corr_y_m = 12 * self.voxel_size

        for i, poses in enumerate(prim_posess):
            poses_msg = PoseArray()
            poses_msg.header.stamp = depthmsg.header.stamp
            poses_msg.header.frame_id = self.heightmap_frame
            bboxes_msg = BoundingBoxArray()
            bboxes_msg.header.stamp = depthmsg.header.stamp
            bboxes_msg.header.frame_id = self.heightmap_frame
            for pose in poses:
                # Pose
                pos_xy_pix = np.round(pose[1]).astype(int)
                pos_xy_m = pose[1] * self.voxel_size
                rad = np.radians(pose[2])
                quat = tf.transformations.quaternion_about_axis(
                    rad, (0, 0, 1)
                )
                pos_z_m = depth[pos_xy_pix[1], pos_xy_pix[0]]
                pose_msg = Pose()
                pose_msg.position.x = pos_xy_m[0] - corr_x_m
                pose_msg.position.y = pos_xy_m[1] - corr_y_m
                pose_msg.position.z = pos_z_m
                pose_msg.orientation.x = quat[0]
                pose_msg.orientation.y = quat[1]
                pose_msg.orientation.z = quat[2]
                pose_msg.orientation.w = quat[3]
                poses_msg.poses.append(pose_msg)

                # Bounding box of instance
                ins_mask = (lbl_ins == pose[0]) * (depth > 0)
                # Denoise mask
                skimage.morphology.remove_small_objects(
                    ins_mask, min_size=50, connectivity=1, in_place=True)
                # array([[y, x], [y, x], ...])
                ins_pts = np.array(np.where(ins_mask)).T
                # array([[x, y], [x, y], ...])
                pts_xy = ins_pts[:, [1, 0]]
                rot = np.array([[np.cos(rad), -np.sin(rad)],
                                [np.sin(rad), np.cos(rad)]])
                pts_aligned = np.dot(pts_xy, rot)
                pts_center = np.mean(pts_xy, axis=0) * self.voxel_size
                ins_depth = depth[ins_mask]
                pts_center_z \
                    = (np.max(ins_depth) + np.min(ins_depth)) / 2
                bbox_msg = BoundingBox()
                bbox_msg.header.stamp = depthmsg.header.stamp
                bbox_msg.header.frame_id = self.heightmap_frame
                xs = pts_aligned[:, 0]
                bbox_msg.dimensions.x \
                    = (np.max(xs) - np.min(xs)) * self.voxel_size
                ys = pts_aligned[:, 1]
                bbox_msg.dimensions.y \
                    = (np.max(ys) - np.min(ys)) * self.voxel_size
                bbox_msg.dimensions.z \
                    = np.max(depth[ins_mask]) - np.min(depth[ins_mask])
                bbox_msg.pose.position.x = pts_center[0] - corr_x_m
                bbox_msg.pose.position.y = pts_center[1] - corr_y_m
                bbox_msg.pose.position.z = pts_center_z
                bbox_msg.pose.orientation = pose_msg.orientation
                bboxes_msg.boxes.append(bbox_msg)

            self.pubs_poses[i].publish(poses_msg)
            self.pubs_boxes[i].publish(bboxes_msg)

        # Publish image for debug
        vizs = []
        vizs.append(img)
        vizs.append(grasp_fusion_lib.image.colorize_depth(
            depth, min_value=0, max_value=0.3))
        vizs.append(
            grasp_fusion_lib.image.label2rgb(lbl_ins + 1, img, alpha=0.7)
        )
        viz = grasp_fusion_lib.image.colorize_heatmap(prob_suc_aff)
        viz = grasp_fusion_lib.image.overlay_color_on_mono(
            img_color=viz, img_mono=img, alpha=0.7
        )
        vizs.append(viz)
        for c in range(prob_pinch_aff.shape[2]):
            prob_c = prob_pinch_aff[:, :, c]
            viz = grasp_fusion_lib.image.colorize_heatmap(prob_c)
            viz = grasp_fusion_lib.image.overlay_color_on_mono(
                img_color=viz, img_mono=img, alpha=0.7
            )
            vizs.append(viz)
        for c in range(prob_pinch_sole_aff.shape[2]):
            prob_c = prob_pinch_sole_aff[:, :, c]
            viz = grasp_fusion_lib.image.colorize_heatmap(prob_c)
            viz = grasp_fusion_lib.image.overlay_color_on_mono(
                img_color=viz, img_mono=img, alpha=0.7
            )
            vizs.append(viz)
        # vizs.extend([np.zeros_like(img)] * 2)
        for poses in prim_posess:
            vizs.append(self._primitive_poses2rgb(poses, img))
        viz = grasp_fusion_lib.image.tile(
            vizs, (-(-len(vizs) // 4), 4), boundary=True
        )
        debug_msg = self.br.cv2_to_imgmsg(viz, encoding='rgb8')
        debug_msg.header.stamp = depthmsg.header.stamp
        debug_msg.header.frame_id = self.heightmap_frame
        self.pub_debug.publish(debug_msg)

    def _primitive_poses2rgb(self, poses, img):
        lbl = np.zeros(img.shape[:2], dtype=int)
        for pose in poses:
            rr, cc = skimage.draw.circle(
                int(round(pose[1][1])), int(round(pose[1][0])), 5)
            # Bug of skimage?
            rr = np.where(rr < 0, 0, rr)
            rr = np.where(rr >= lbl.shape[0], lbl.shape[0] - 1, rr)
            cc = np.where(cc < 0, 0, cc)
            cc = np.where(cc >= lbl.shape[1], lbl.shape[1] - 1, cc)
            lbl[rr, cc] = pose[0] + 1
        return grasp_fusion_lib.image.label2rgb(lbl, img, alpha=0.7)

    def ignore_ins_cb(self, req):
        with self.lock:
            self.ignore_ins = req.data
        return SetBoolResponse(success=True)


if __name__ == '__main__':
    rospy.init_node('primitive_matching')
    node = PrimitiveMatching()
    rospy.spin()
