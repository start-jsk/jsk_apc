#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import easydict
import matplotlib
import numpy as np
import os.path as osp
# import time
import yaml

matplotlib.use('Agg')  # NOQA

import chainer
import cv_bridge
from dynamic_reconfigure.server import Server
from jsk_recognition_msgs.msg import ClassificationResult
from jsk_recognition_msgs.msg import ClusterPointIndices
from jsk_recognition_msgs.msg import Label
from jsk_recognition_msgs.msg import LabelArray
from jsk_recognition_msgs.msg import Rect
from jsk_recognition_msgs.msg import RectArray
from jsk_topic_tools import ConnectionBasedTransport
import matplotlib.pyplot as plt
import message_filters
from pcl_msgs.msg import PointIndices
import rospy
import scipy
from sensor_msgs.msg import Image
from sklearn.cluster import KMeans
from std_srvs.srv import Trigger
from std_srvs.srv import TriggerResponse

from dualarm_grasping.models import OccludedGraspMaskRCNNResNet101
from dualarm_grasping.visualizations import vis_occluded_grasp_instance_segmentation

from dualarm_grasping.cfg \
    import DualarmOccludedGraspInstanceSegmentationConfig
from dualarm_grasping.msg import GraspClassificationResult
from dualarm_grasping.srv import GetAnother
from dualarm_grasping.srv import GetAnotherResponse


filepath = osp.dirname(osp.realpath(__file__))


class DualarmOccludedGraspInstanceSegmentation(ConnectionBasedTransport):

    def __init__(self):
        super(self.__class__, self).__init__()
        self.gpu = rospy.get_param('~gpu', -1)
        model_file = rospy.get_param('~model_file')

        self.label_names = rospy.get_param('~label_names')
        self.bg_index = rospy.get_param('~bg_index', -1)
        if self.bg_index >= 0:
            bg_label_name = self.label_names[self.bg_index]
            if bg_label_name != '__background__':
                rospy.logerr('bg_label_name is not __background__: {}'
                             .format(bg_label_name))

        self.sampling = rospy.get_param('~sampling', False)
        self.sampling_weighted = rospy.get_param('~sampling_weighted', False)
        cfgpath = rospy.get_param(
            '~config_yaml', osp.join(filepath, '../yaml/config.yaml'))
        with open(cfgpath, 'r') as f:
            config = easydict.EasyDict(yaml.load(f))

        tote_contents = rospy.get_param('~tote_contents', None)
        self.candidates = [self.bg_index]
        if tote_contents is None:
            self.candidates += range(len(self.label_names))
        else:
            self.candidates += [
                self.label_names.index(x) for x in tote_contents]
        self.candidates = sorted(list(set(self.candidates)))

        self.giveup_ins_ids = {
            'single': [],
            'dual': [],
        }

        self.target_grasp = rospy.get_param('~target_grasp', False)
        target_names = rospy.get_param('~target_names', None)
        if self.target_grasp and target_names is None:
            target_names = self.label_names
        self.target_ids = [
            self.label_names.index(x) for x in target_names]

        # chainer global config
        chainer.global_config.train = False
        chainer.global_config.enable_backprop = False

        # mask rcnn
        if 'rotate_angle' not in config:
            self.rotate_angle = None
        else:
            self.rotate_angle = config.rotate_angle
        self.model = OccludedGraspMaskRCNNResNet101(
            n_fg_class=len(self.label_names),
            anchor_scales=config.anchor_scales,
            min_size=config.min_size,
            max_size=config.max_size,
            rpn_dim=config.rpn_dim,
            rotate_angle=self.rotate_angle)
        chainer.serializers.load_npz(model_file, self.model)

        if self.gpu != -1:
            chainer.cuda.get_device_from_id(self.gpu).use()
            self.model.to_gpu()

        # input
        self.pub_net_input = self.advertise(
            '~debug/net_input', Image, queue_size=1)
        self.pub_vis_output = self.advertise(
            '~debug/vis_output', Image, queue_size=1)
        # vis
        self.pub_vis_cpi = self.advertise(
            '~output/vis/cluster_indices', ClusterPointIndices, queue_size=1)
        self.pub_vis_labels = self.advertise(
            '~output/vis/labels', LabelArray, queue_size=1)
        self.pub_vis_cls_lbl = self.advertise(
            '~output/vis/cls_label', Image, queue_size=1)
        self.pub_vis_ins_lbl = self.advertise(
            '~output/vis/ins_label', Image, queue_size=1)

        # occ
        self.pub_occ_cpi = self.advertise(
            '~output/occ/cluster_indices', ClusterPointIndices, queue_size=1)
        self.pub_occ_labels = self.advertise(
            '~output/occ/labels', LabelArray, queue_size=1)
        self.pub_occ_cls_lbl = self.advertise(
            '~output/occ/cls_label', Image, queue_size=1)
        self.pub_occ_ins_lbl = self.advertise(
            '~output/occ/ins_label', Image, queue_size=1)

        # bbox
        self.pub_rects = self.advertise(
            "~output/rects", RectArray, queue_size=1)

        # class
        self.pub_class = self.advertise(
            "~output/class", ClassificationResult,
            queue_size=1)

        # single
        self.pub_sg_cpi = self.advertise(
            '~output/single/cluster_indices',
            ClusterPointIndices, queue_size=1)
        self.pub_sg_labels = self.advertise(
            '~output/single/labels', LabelArray, queue_size=1)
        self.pub_sg_cls_lbl = self.advertise(
            '~output/single/cls_label', Image, queue_size=1)
        self.pub_sg_ins_lbl = self.advertise(
            '~output/single/ins_label', Image, queue_size=1)

        # dual
        self.pub_dg_cpi = self.advertise(
            '~output/dual/cluster_indices',
            ClusterPointIndices, queue_size=1)
        self.pub_dg_labels = self.advertise(
            '~output/dual/labels', LabelArray, queue_size=1)
        self.pub_dg_cls_lbl = self.advertise(
            '~output/dual/cls_label', Image, queue_size=1)
        self.pub_dg_ins_lbl = self.advertise(
            '~output/dual/ins_label', Image, queue_size=1)

        # output
        self.pub_grasp_mask = self.advertise(
            '~output/grasp_mask', Image, queue_size=1)
        self.pub_grasp_class = self.advertise(
            '~output/grasp_class', GraspClassificationResult, queue_size=1)

        self.get_another_service = rospy.Service(
            '~get_another', GetAnother, self._get_another)
        self.reset_service = rospy.Service(
            '~reset', Trigger, self._reset)
        self.dyn_srv = Server(
            DualarmOccludedGraspInstanceSegmentationConfig,
            self._dyn_callback)

    def subscribe(self):
        # larger buff_size is necessary for taking time callback
        # http://stackoverflow.com/questions/26415699/ros-subscriber-not-up-to-date/29160379#29160379  # NOQA
        queue_size = rospy.get_param('~queue_size', 10)
        self.use_mask = rospy.get_param('~use_mask', True)
        if self.use_mask:
            sub = message_filters.Subscriber(
                '~input', Image, queue_size=1, buff_size=2**24)
            sub_mask = message_filters.Subscriber(
                '~input/mask', Image, queue_size=1, buff_size=2**24)
            self.subs = [sub, sub_mask]
            if rospy.get_param('~approximate_sync', False):
                slop = rospy.get_param('~slop', 0.1)
                sync = message_filters.ApproximateTimeSynchronizer(
                    self.subs, queue_size=queue_size, slop=slop)
            else:
                sync = message_filters.TimeSynchronizer(
                    self.subs, queue_size=queue_size)
            sync.registerCallback(self._recognize)
        else:
            sub = rospy.Subscriber(
                '~input', Image, callback=self._recognize,
                queue_size=queue_size, buff_size=2**24)

    def unsubscribe(self):
        for sub in self.subs:
            sub.unregister()

    def _recognize(self, img_msg, mask_msg=None):
        self.model.score_thresh = self.score_thresh
        self.model.nms_thresh = self.nms_thresh

        bridge = cv_bridge.CvBridge()
        rgb = bridge.imgmsg_to_cv2(img_msg, desired_encoding='rgb8')
        if self.use_mask:
            if mask_msg is not None:
                mask = bridge.imgmsg_to_cv2(mask_msg)
                # rgb[mask < 128] = self.model.mean.flatten()
        # H, W, C -> C, H, W
        img = rgb.transpose((2, 0, 1))
        results = self.model.predict([img], return_probs=True)
        ins_labels, ins_probs, labels, bboxes, scores = results[:5]
        sg_labels, sg_probs, dg_labels, dg_probs = results[5:]
        try:
            ins_label, ins_prob, label, bbox, score = \
                ins_labels[0], ins_probs[0], labels[0], bboxes[0], scores[0]
            sg_label, sg_prob, dg_label, dg_prob = \
                sg_labels[0], sg_probs[0], dg_labels[0], dg_probs[0]
        except IndexError:
            rospy.logerr('no predicts returned')
            return

        # matplot
        fig, axes = plt.subplots(
            1, 5, sharey=True, figsize=(100, 20), dpi=120)
        vis_occluded_grasp_instance_segmentation(
            img, ins_label, label, bbox, score,
            sg_label, dg_label, self.label_names,
            rotate_angle=self.rotate_angle, axes=axes,
            linewidth=5.0, fontsize=30)
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        vis_output_img = np.fromstring(
            fig.canvas.tostring_rgb(), dtype=np.uint8)
        fig.clf()
        vis_output_img.shape = (h, w, 3)
        plt.close()
        vis_output_msg = bridge.cv2_to_imgmsg(
            vis_output_img, encoding='rgb8')
        vis_output_msg.header = img_msg.header

        # msg
        # input
        net_input_msg = bridge.cv2_to_imgmsg(
            rgb.astype(np.uint8), encoding='rgb8')
        net_input_msg.header = img_msg.header

        # vis lbls
        vis_cpi_msg = ClusterPointIndices(header=img_msg.header)
        vis_labels_msg = LabelArray(header=img_msg.header)
        vis_cls_lbl = - np.ones(img.shape[1:], dtype=np.int32)
        vis_ins_lbl = - np.ones(img.shape[1:], dtype=np.int32)
        vis_ins_n_pixel = []

        # occ lbls
        occ_cpi_msg = ClusterPointIndices(header=img_msg.header)
        occ_labels_msg = LabelArray(header=img_msg.header)
        occ_cls_lbl = - np.ones(img.shape[1:], dtype=np.int32)
        occ_ins_lbl = - np.ones(img.shape[1:], dtype=np.int32)
        occ_ins_n_pixel = []

        for ins_id, (cls_id, ins_lbl) in enumerate(zip(label, ins_label)):
            # vis region
            if self.use_mask:
                vis_msk = np.logical_and(ins_lbl == 1, mask > 128)
            else:
                vis_msk = ins_lbl == 1
            vis_ins_n_pixel.append(vis_msk.sum())

            # occ region
            if self.use_mask:
                occ_msk = np.logical_and(ins_lbl == 2, mask > 128)
            else:
                occ_msk = ins_lbl == 2
            occ_ins_n_pixel.append(occ_msk.sum())

            if cls_id not in self.candidates:
                continue

            # vis region
            class_name = self.label_names[cls_id]
            vis_indices = np.where(vis_msk.flatten())[0]
            vis_indices_msg = PointIndices(
                header=img_msg.header, indices=vis_indices)
            vis_cpi_msg.cluster_indices.append(vis_indices_msg)
            vis_labels_msg.labels.append(
                Label(id=cls_id, name=class_name))
            vis_cls_lbl[vis_msk] = cls_id
            vis_ins_lbl[vis_msk] = ins_id

            # occ region
            occ_indices = np.where(occ_msk.flatten())[0]
            occ_indices_msg = PointIndices(
                header=img_msg.header, indices=occ_indices)
            occ_cpi_msg.cluster_indices.append(occ_indices_msg)
            occ_labels_msg.labels.append(
                Label(id=cls_id, name=class_name))
            occ_cls_lbl[occ_msk] = cls_id
            occ_ins_lbl[occ_msk] = ins_id

        vis_cls_lbl_msg = bridge.cv2_to_imgmsg(vis_cls_lbl)
        vis_cls_lbl_msg.header = img_msg.header
        vis_ins_lbl_msg = bridge.cv2_to_imgmsg(vis_ins_lbl)
        vis_ins_lbl_msg.header = img_msg.header
        vis_ins_n_pixel = np.array(vis_ins_n_pixel, dtype=np.int32)

        occ_cls_lbl_msg = bridge.cv2_to_imgmsg(occ_cls_lbl)
        occ_cls_lbl_msg.header = img_msg.header
        occ_ins_lbl_msg = bridge.cv2_to_imgmsg(occ_ins_lbl)
        occ_ins_lbl_msg.header = img_msg.header
        occ_ins_n_pixel = np.array(occ_ins_n_pixel, dtype=np.int32)

        # bbox
        rects_msg = RectArray(header=img_msg.header)
        for bb in bbox:
            rect = Rect(x=bb[1], y=bb[0],
                        width=bb[3] - bb[1],
                        height=bb[2] - bb[0])
            rects_msg.rects.append(rect)

        # classification
        cls_msg = ClassificationResult(
            header=img_msg.header,
            classifier='OccludedGraspMaskRCNNResNet101',
            target_names=self.label_names,
            labels=label,
            label_names=[self.label_names[lbl] for lbl in label],
            label_proba=score,
        )

        # sg, dg
        sg_cpi_msg = ClusterPointIndices(header=img_msg.header)
        sg_labels_msg = LabelArray(header=img_msg.header)
        sg_cls_lbl = - np.ones(img.shape[1:], dtype=np.int32)
        sg_ins_lbl = - np.ones(img.shape[1:], dtype=np.int32)
        dg_cpi_msg = ClusterPointIndices(header=img_msg.header)
        dg_labels_msg = LabelArray(header=img_msg.header)
        dg_cls_lbl = - np.ones(img.shape[1:], dtype=np.int32)
        dg_ins_lbl = - np.ones(img.shape[1:], dtype=np.int32)

        for ins_id, (cls_id, sg_lbl, dg_lbl) in enumerate(
                zip(label, sg_label, dg_label)):
            if cls_id not in self.candidates:
                continue
            class_name = self.label_names[cls_id]

            # sg
            if self.use_mask:
                sg_msk = np.logical_and(sg_lbl > 0, mask > 128)
            else:
                sg_msk = sg_lbl > 0
            sg_indices = np.where(sg_msk.flatten())[0]
            sg_indices_msg = PointIndices(
                header=img_msg.header, indices=sg_indices)
            sg_cpi_msg.cluster_indices.append(sg_indices_msg)
            sg_labels_msg.labels.append(
                Label(id=cls_id, name=class_name))
            sg_cls_lbl[sg_msk] = cls_id
            sg_ins_lbl[sg_msk] = ins_id

            # dg
            if self.use_mask:
                dg_msk = np.logical_and(dg_lbl > 0, mask > 128)
            else:
                dg_msk = dg_lbl > 0
            dg_indices = np.where(dg_msk.flatten())[0]
            dg_indices_msg = PointIndices(
                header=img_msg.header, indices=dg_indices)
            dg_cpi_msg.cluster_indices.append(dg_indices_msg)
            dg_labels_msg.labels.append(
                Label(id=cls_id, name=class_name))
            dg_cls_lbl[dg_msk] = cls_id
            dg_ins_lbl[dg_msk] = ins_id

        sg_cls_lbl_msg = bridge.cv2_to_imgmsg(sg_cls_lbl)
        sg_cls_lbl_msg.header = img_msg.header
        sg_ins_lbl_msg = bridge.cv2_to_imgmsg(sg_ins_lbl)
        sg_ins_lbl_msg.header = img_msg.header
        dg_cls_lbl_msg = bridge.cv2_to_imgmsg(dg_cls_lbl)
        dg_cls_lbl_msg.header = img_msg.header
        dg_ins_lbl_msg = bridge.cv2_to_imgmsg(dg_ins_lbl)
        dg_ins_lbl_msg.header = img_msg.header

        sg_ins_prob_img = ins_prob[:, 1, :, :] * \
            np.sum(sg_prob[:, 1:, :, :], axis=1)
        if self.use_mask:
            sg_ins_prob_mask = np.repeat(
                (mask <= 128)[None], len(sg_ins_prob_img), axis=0)
            sg_ins_prob_img[sg_ins_prob_mask] = 0
        sg_ins_prob = np.max(sg_ins_prob_img, axis=(1, 2))
        assert len(sg_ins_prob) == len(sg_prob)
        assert sg_ins_prob.ndim == 1

        dg_ins_prob_img = ins_prob[:, 1, :, :] * \
            np.sum(dg_prob[:, 1:, :, :], axis=1)
        if self.use_mask:
            dg_ins_prob_mask = np.repeat(
                (mask <= 128)[None], len(dg_ins_prob_img), axis=0)
            dg_ins_prob_img[dg_ins_prob_mask] = 0
        dg_ins_prob = np.max(dg_ins_prob_img, axis=(1, 2))
        assert len(dg_ins_prob) == len(dg_prob)
        assert dg_ins_prob.ndim == 1

        # grasp mask and style
        if self.sampling:
            if len(self.candidates) != 2:
                rospy.logerr('Invalid tote contents num: {}'.format(
                             self.candidates))
            grasp_style = self.grasping_way
            if self.candidates[1] in label:
                grasp_cls_ids = [self.candidates[1]]
                if grasp_style == 'single':
                    grasp_probs = sg_ins_prob[label == grasp_cls_ids[0]][0:1]
                    grasp_mask = self._random_sample_sg_mask(
                        sg_cls_lbl == grasp_cls_ids[0],
                        vis_cls_lbl == grasp_cls_ids[0])
                else:
                    grasp_probs = dg_ins_prob[label == grasp_cls_ids[0]][0:1]
                    grasp_mask = self._random_sample_dg_mask(
                        dg_cls_lbl == grasp_cls_ids[0],
                        vis_cls_lbl == grasp_cls_ids[0])
            else:
                grasp_cls_ids = []
                grasp_probs = []
                grasp_mask = np.zeros(img.shape[1:], dtype=np.uint8)
            is_target = True
        else:
            sg_ins_prob[sg_ins_prob <= self.grasp_thresh] = 0
            dg_ins_prob[dg_ins_prob <= self.grasp_thresh] = 0
            vis_ratio = vis_ins_n_pixel / (vis_ins_n_pixel + occ_ins_n_pixel)
            vis_ratio[np.isnan(vis_ratio)] = 0
            target_ins_ids = []
            if self.target_grasp:
                for ins_id in range(len(vis_ratio)):
                    if (label[ins_id] in self.target_ids and
                            (sg_ins_prob[ins_id] > 0 or
                                dg_ins_prob[ins_id] > 0)):
                        target_ins_ids.append(ins_id)
            target_ins_ids = np.array(target_ins_ids, dtype=np.int32)

            if self.target_grasp and len(target_ins_ids) > 0:
                target_sg_ins_prob = sg_ins_prob[target_ins_ids]
                target_dg_ins_prob = dg_ins_prob[target_ins_ids]

                sg_ins_prob_max = target_sg_ins_prob.max()
                dg_ins_prob_max = target_dg_ins_prob.max()
                sg_ins_id = target_ins_ids[target_sg_ins_prob.argmax()]
                dg_ins_id = target_ins_ids[target_dg_ins_prob.argmax()]

                # either of sg, dg are graspable
                if vis_ratio[sg_ins_id] > self.vis_thresh \
                        or vis_ratio[dg_ins_id] > self.vis_thresh:
                    # both of sg, dg are graspable
                    if vis_ratio[sg_ins_id] > self.vis_thresh \
                            and vis_ratio[dg_ins_id] > self.vis_thresh:
                        if sg_ins_prob_max > dg_ins_prob_max:
                            grasp_style = 'single'
                        else:
                            grasp_style = 'dual'
                    # sg is grapable, but dg is not
                    elif vis_ratio[sg_ins_id] > self.vis_thresh:
                        grasp_style = 'single'
                    # dg is grapable, but sg is not
                    else:
                        grasp_style = 'dual'

                    if grasp_style == 'single':
                        grasp_ins_ids = [sg_ins_id]
                        grasp_probs = [sg_ins_prob_max]
                        grasp_mask = \
                            sg_ins_prob_img[sg_ins_id] > self.grasp_thresh
                    else:
                        grasp_ins_ids = [dg_ins_id]
                        grasp_probs = [dg_ins_prob_max]
                        grasp_mask = \
                            dg_ins_prob_img[dg_ins_id] > self.grasp_thresh
                # none of sg, dg are graspable
                else:
                    if sg_ins_prob_max > dg_ins_prob_max:
                        occ_ins_id = self._find_occ_top(
                            sg_ins_id, ins_label, label)
                    else:
                        occ_ins_id = self._find_occ_top(
                            dg_ins_id, ins_label, label)

                    if sg_ins_prob[occ_ins_id] > dg_ins_prob[occ_ins_id]:
                        grasp_style = 'single'
                        grasp_ins_ids = [occ_ins_id]
                        grasp_probs = [sg_ins_prob[occ_ins_id]]
                        grasp_mask = \
                            sg_ins_prob_img[occ_ins_id] > self.grasp_thresh
                    else:
                        grasp_style = 'dual'
                        grasp_ins_ids = [occ_ins_id]
                        grasp_probs = [dg_ins_prob[occ_ins_id]]
                        grasp_mask = \
                            dg_ins_prob_img[occ_ins_id] > self.grasp_thresh
            else:
                sg_ins_prob[vis_ratio <= self.vis_thresh] = 0
                dg_ins_prob[vis_ratio <= self.vis_thresh] = 0
                is_candidates = np.array(
                    [lbl in self.candidates for lbl in label], dtype=np.bool)
                sg_ins_prob[~is_candidates] = 0
                dg_ins_prob[~is_candidates] = 0
                sg_ins_prob_max = sg_ins_prob.max()
                dg_ins_prob_max = dg_ins_prob.max()

                if sg_ins_prob_max > dg_ins_prob_max:
                    grasp_style = 'single'
                    grasp_ins_ids = [np.argmax(sg_ins_prob)]
                    grasp_probs = [sg_ins_prob_max]
                    if label[grasp_ins_ids[0]] in self.candidates:
                        grasp_mask = sg_ins_prob_img[
                            grasp_ins_ids[0]] > self.grasp_thresh
                    else:
                        grasp_mask = np.zeros(img.shape[1:], dtype=np.uint8)
                else:
                    grasp_style = 'dual'
                    grasp_ins_ids = [np.argmax(dg_ins_prob)]
                    grasp_probs = [dg_ins_prob_max]
                    if label[grasp_ins_ids[0]] in self.candidates:
                        grasp_mask = dg_ins_prob_img[
                            grasp_ins_ids[0]] > self.grasp_thresh
                    else:
                        grasp_mask = np.zeros(img.shape[1:], dtype=np.uint8)

            grasp_mask = grasp_mask.astype(np.uint8) * 255
            grasp_cls_ids = [
                label[grasp_ins_id] for grasp_ins_id in grasp_ins_ids]
            if self.target_grasp:
                is_target = grasp_cls_ids[0] in self.target_ids
            else:
                is_target = True

        grasp_mask_msg = bridge.cv2_to_imgmsg(grasp_mask, encoding='mono8')
        grasp_mask_msg.header = img_msg.header

        grasp_label_names = []
        for grasp_cls_id in grasp_cls_ids:
            if grasp_cls_id >= 0:
                grasp_label_names.append(self.label_names[grasp_cls_id])
            else:
                grasp_label_names.append('__background__')

        grasp_cls_msg = GraspClassificationResult(
            header=img_msg.header,
            style=grasp_style,
            is_target=is_target,
            classification=ClassificationResult(
                header=img_msg.header,
                labels=grasp_cls_ids,
                label_names=grasp_label_names,
                label_proba=grasp_probs,
                classifier='OccludedGraspMaskRCNNResNet101',
                target_names=self.label_names))

        # publish
        # input
        self.pub_net_input.publish(net_input_msg)
        self.pub_vis_output.publish(vis_output_msg)

        # vis
        self.pub_vis_cpi.publish(vis_cpi_msg)
        self.pub_vis_labels.publish(vis_labels_msg)
        self.pub_vis_cls_lbl.publish(vis_cls_lbl_msg)
        self.pub_vis_ins_lbl.publish(vis_ins_lbl_msg)

        # occ
        self.pub_occ_cpi.publish(occ_cpi_msg)
        self.pub_occ_labels.publish(occ_labels_msg)
        self.pub_occ_cls_lbl.publish(occ_cls_lbl_msg)
        self.pub_occ_ins_lbl.publish(occ_ins_lbl_msg)

        # bbox
        self.pub_rects.publish(rects_msg)

        # class
        self.pub_class.publish(cls_msg)

        # sg
        self.pub_sg_cpi.publish(sg_cpi_msg)
        self.pub_sg_labels.publish(sg_labels_msg)
        self.pub_sg_cls_lbl.publish(sg_cls_lbl_msg)
        self.pub_sg_ins_lbl.publish(sg_ins_lbl_msg)

        # dg
        self.pub_dg_cpi.publish(dg_cpi_msg)
        self.pub_dg_labels.publish(dg_labels_msg)
        self.pub_dg_cls_lbl.publish(dg_cls_lbl_msg)
        self.pub_dg_ins_lbl.publish(dg_ins_lbl_msg)

        self.pub_grasp_mask.publish(grasp_mask_msg)
        self.pub_grasp_class.publish(grasp_cls_msg)
        # time.sleep(1.0)

    def _random_sample_sg_mask(self, sg_prb, label_msk):
        weight = sg_prb
        weight[~label_msk] = 0.0
        if self.sampling_weighted and np.sum(weight) > 0:
            weight = weight.ravel() / np.sum(weight)
        elif np.sum(label_msk) > 0:
            label_msk = label_msk.astype(np.float32)
            weight = label_msk.ravel() / np.sum(label_msk)
        else:
            weight = None
        sample_grasp = np.zeros(sg_prb.shape)
        sampled_i = np.random.choice(sg_prb.size, p=weight)
        sampled_index = (
            sampled_i // sg_prb.shape[1],
            sampled_i % sg_prb.shape[1],
        )
        sample_grasp[sampled_index] = 255
        sample_grasp = scipy.ndimage.filters.gaussian_filter(
            sample_grasp, sigma=20)
        sample_grasp = sample_grasp / sample_grasp.max()
        sample_mask = sample_grasp > self.sampling_thresh
        sample_mask = sample_mask.astype(np.uint8) * 255
        return sample_mask

    def _random_sample_dg_mask(self, dg_prb, label_msk):
        indices = np.column_stack(np.where(label_msk))

        c_masks = []
        if indices.size > 1:
            kmeans = KMeans(n_clusters=2)
            try:
                kmeans.fit(indices)
                centers = kmeans.cluster_centers_
                labels = kmeans.labels_
                for label, center in enumerate(centers):
                    center = np.round(center).astype(np.int32)
                    c_mask = np.zeros(label_msk.shape).astype(bool)
                    masked_indices = indices[labels == label]
                    masked_indices = (
                        masked_indices[:, 0], masked_indices[:, 1])
                    c_mask[masked_indices] = True
                    weight = dg_prb.copy()
                    weight[~c_mask] = 0.0
                    if self.sampling_weighted and np.sum(weight) > 0:
                        weight = weight.ravel() / np.sum(weight)
                    elif np.sum(label_msk) > 0:
                        label_msk = label_msk.astype(np.float32)
                        weight = label_msk.ravel() / np.sum(label_msk)
                    else:
                        weight = None
                    dual_giveup = False
                    trial_num = 10
                    for i in range(0, trial_num):
                        sampled_i = np.random.choice(dg_prb.size, p=weight)
                        sampled_index = (
                            sampled_i // dg_prb.shape[1],
                            sampled_i % dg_prb.shape[1],
                        )
                        c_grasp = np.zeros(dg_prb.shape)
                        c_grasp[sampled_index] = 255
                        c_grasp = scipy.ndimage.filters.gaussian_filter(
                            c_grasp, sigma=20)
                        c_grasp = c_grasp / c_grasp.max()
                        c_mask = c_grasp > self.sampling_thresh
                        if len(c_masks) > 0:
                            if not np.any(np.logical_and(c_mask, c_masks[0])):
                                c_masks.append(c_mask)
                                break
                        else:
                            c_masks.append(c_mask)
                            break
                        if i == trial_num - 1:
                            dual_giveup = True
            except Exception:
                dual_giveup = True
        if len(c_masks) == 2 and dual_giveup is False:
            sample_mask = np.logical_or(c_masks[0], c_masks[1])
            sample_mask = sample_mask.astype(np.uint8) * 255
        else:
            sample_mask = np.zeros(dg_prb.shape, dtype=np.uint8)

        return sample_mask

    def _find_occ_top(self, target_ins_id, ins_label, label):
        checked_ids = set()
        checked_ids.add(target_ins_id)
        vis_msk = ins_label[target_ins_id] == 1
        vis_rto = vis_msk.sum() / (ins_label[target_ins_id] > 0).sum()
        if vis_rto > self.vis_thresh:
            rospy.loginfo(
                '{}_{} is not occluded but graspable!'
                .format(self.label_names[label[target_ins_id]], target_ins_id))
            return target_ins_id
        ret_ins_id = None
        for ins_id, ins_lbl in enumerate(ins_label):
            if ins_id in checked_ids:
                continue
            else:
                occ_msk = ins_label[target_ins_id] == 2
                occ_msk_by_this = np.logical_and(ins_lbl == 1, occ_msk)
                occ_rto = occ_msk_by_this.sum() / occ_msk.sum()
                if occ_rto > 0.1:
                    rospy.loginfo(
                        '{}_{} is occluded by {}_{}'
                        .format(self.label_names[label[target_ins_id]],
                                target_ins_id,
                                self.label_names[label[ins_id]], ins_id))
                    ret_ins_id = self._find_occ_top_step(
                        ins_id, ins_label, label, checked_ids)
                if ret_ins_id is not None:
                    break
        if ret_ins_id is None:
            return target_ins_id
        else:
            return ret_ins_id

    def _find_occ_top_step(self, target_ins_id, ins_label, label, checked_ids):
        checked_ids.add(target_ins_id)
        vis_msk = ins_label[target_ins_id] == 1
        vis_rto = vis_msk.sum() / (ins_label[target_ins_id] > 0).sum()
        if vis_rto > self.vis_thresh:
            rospy.loginfo(
                '{}_{} is not occluded but graspable!'
                .format(self.label_names[label[target_ins_id]], target_ins_id))
            return target_ins_id
        ret_ins_id = None
        for ins_id, ins_lbl in enumerate(ins_label):
            if ins_id in checked_ids:
                continue
            else:
                occ_msk = ins_label[target_ins_id] == 2
                occ_msk_by_this = np.logical_and(ins_lbl == 1, occ_msk)
                occ_rto = occ_msk_by_this.sum() / occ_msk.sum()
                if occ_rto > 0.1:
                    rospy.loginfo(
                        '{}_{} is occluded by {}_{}'
                        .format(self.label_names[label[target_ins_id]],
                                target_ins_id,
                                self.label_names[label[ins_id]], ins_id))
                    ret_ins_id = self._find_occ_top_step(
                        ins_id, ins_label, label, checked_ids)
                if ret_ins_id is not None:
                    break
        return ret_ins_id

    def _get_another(self, req):
        grasp_style = req.style
        label = req.label
        self.giveup_ins_ids[grasp_style].append(label)
        response = GetAnotherResponse()
        response.success = True
        return response

    def _reset(self, req):
        self.giveup_ins_ids = {
            'single': [],
            'dual': [],
        }
        response = TriggerResponse()
        response.success = True
        return response

    def _dyn_callback(self, config, level):
        self.score_thresh = config.score_thresh
        self.grasp_thresh = config.grasp_thresh
        self.nms_thresh = config.nms_thresh
        self.vis_thresh = config.vis_thresh
        self.sampling_thresh = config.sampling_thresh
        self.grasping_way = config.grasping_way
        return config


if __name__ == '__main__':
    rospy.init_node('dualarm_occluded_grasp_instance_segmentation')
    node = DualarmOccludedGraspInstanceSegmentation()
    rospy.spin()
