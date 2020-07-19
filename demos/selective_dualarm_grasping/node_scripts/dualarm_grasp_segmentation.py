#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import chainer
import chainer.serializers as S
from chainer import Variable
import cupy
import cv2
import numpy as np
import os.path as osp
import scipy
from sklearn.cluster import KMeans
import time
import yaml

import cv_bridge
from dynamic_reconfigure.server import Server
from jsk_recognition_msgs.msg import ClassificationResult
from jsk_topic_tools import ConnectionBasedTransport
from jsk_topic_tools.log_utils import logerr_throttle
import message_filters
import rospy
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger
from std_srvs.srv import TriggerResponse

from dualarm_grasping.cfg import DualarmGraspSegmentationConfig
from dualarm_grasping.models import DualarmGraspFCN32s
from dualarm_grasping.msg import GraspClassificationResult
from dualarm_grasping.srv import GetAnother
from dualarm_grasping.srv import GetAnotherResponse


filepath = osp.dirname(osp.realpath(__file__))


class DualarmGraspSegmentation(ConnectionBasedTransport):

    def __init__(self):
        super(self.__class__, self).__init__()
        self.gpu = rospy.get_param('~gpu', -1)
        model_file = rospy.get_param('~model_file')
        self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])

        self.label_names = rospy.get_param('~label_names')
        self.bg_index = rospy.get_param('~bg_index', 0)
        self.sampling = rospy.get_param('~sampling', False)
        self.sampling_weighted = rospy.get_param('~sampling_weighted', False)
        cfgpath = rospy.get_param(
            '~config_yaml', osp.join(filepath, '../yaml/config.yaml'))
        with open(cfgpath, 'r') as f:
            config = yaml.load(f)

        tote_contents = rospy.get_param('~tote_contents', None)
        if tote_contents is None:
            self.candidates = range(len(self.label_names))
        else:
            self.candidates = [0] + [self.label_names.index(x)
                                     for x in tote_contents]

        self.giveup_labels = {
            'single': [self.bg_index],
            'dual': [self.bg_index],
        }

        self.model = DualarmGraspFCN32s(
            n_class=len(self.label_names))
        S.load_npz(model_file, self.model)

        if 'alpha' in config:
            alpha = config['alpha']
            if isinstance(alpha, dict):
                self.model.alpha_single = alpha['single']
                self.model.alpha_dual = alpha['dual']
            else:
                self.model.alpha_single = alpha
                self.model.alpha_dual = 1.0
        else:
            self.model.alpha_single = 1.0
            self.model.alpha_dual = 1.0

        if 'frequency_balancing' in config:
            frq_balancing = config['frequency_balancing']
            self.model.frq_balancing = frq_balancing
        else:
            self.model.frq_balancing = False

        chainer.global_config.train = False
        chainer.global_config.enable_backprop = False

        if self.gpu != -1:
            chainer.cuda.get_device(self.gpu).use()
            self.model.to_gpu(self.gpu)
        self.pub_input = self.advertise(
            '~debug/net_input', Image, queue_size=1)
        self.pub_seg = self.advertise(
            '~output/seg_prob', Image, queue_size=1)
        self.pub_label = self.advertise(
            '~output/label', Image, queue_size=1)

        self.pub_sg_grasp = self.advertise(
            '~output/single/grasp', Image, queue_size=1)
        self.pub_sg_mask = self.advertise(
            '~output/single/mask', Image, queue_size=1)
        self.pub_sg_label = self.advertise(
            '~output/single/class', ClassificationResult, queue_size=1)

        self.pub_dg_grasp = self.advertise(
            '~output/dual/grasp', Image, queue_size=1)
        self.pub_dg_mask = self.advertise(
            '~output/dual/mask', Image, queue_size=1)
        self.pub_dg_label = self.advertise(
            '~output/dual/class', ClassificationResult, queue_size=1)

        self.pub_mask = self.advertise(
            '~output/grasp_mask', Image, queue_size=1)
        self.pub_grasp = self.advertise(
            '~output/grasp_class', GraspClassificationResult, queue_size=1)

        self.get_another_service = rospy.Service(
            '~get_another', GetAnother, self._get_another)
        self.reset_service = rospy.Service(
            '~reset', Trigger, self._reset)
        self.dyn_srv = Server(
            DualarmGraspSegmentationConfig, self._dyn_callback)

    def subscribe(self):
        # larger buff_size is necessary for taking time callback
        # http://stackoverflow.com/questions/26415699/ros-subscriber-not-up-to-date/29160379#29160379  # NOQA
        queue_size = rospy.get_param('~queue_size', 10)
        use_mask = rospy.get_param('~use_mask', True)
        if use_mask:
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

    def _recognize(self, imgmsg, mask_msg=None):
        bridge = cv_bridge.CvBridge()
        bgr = bridge.imgmsg_to_cv2(imgmsg, desired_encoding='bgr8')
        bgr = cv2.resize(bgr, (640, 480))
        if mask_msg is not None:
            mask = bridge.imgmsg_to_cv2(mask_msg)
            mask = cv2.resize(mask, (640, 480))
            if mask.shape != bgr.shape[:2]:
                logerr_throttle(10,
                                'Size of input image and mask is different')
                return
            elif mask.size == 0:
                logerr_throttle(10, 'Size of input mask is 0')
                return
            bgr[mask < 128] = self.mean_bgr

        blob = (bgr - self.mean_bgr).transpose((2, 0, 1))
        x_data = np.array([blob], dtype=np.float32)
        if self.gpu != -1:
            x_data = chainer.cuda.to_gpu(x_data, device=self.gpu)
        x = Variable(x_data)
        self.model(x)

        score = self.model.score.array[0]
        sg_score = self.model.single_grasp_score.array[0]
        dg_score = self.model.dual_grasp_score.array[0]

        # segmentation
        seg_prob = chainer.functions.softmax(score, axis=0)
        seg_prob = seg_prob.array
        n_labels = len(self.label_names)
        for lbl in range(n_labels):
            if lbl not in self.candidates:
                seg_prob[lbl, :, :] = 0.0
        seg_prob = seg_prob / cupy.atleast_3d(seg_prob.sum(axis=0)[None, :, :])
        seg_prob[seg_prob < self.score_thresh] = 0.0
        label_pred = seg_prob.argmax(axis=0)

        # singlearm grasp
        sg_probs = chainer.functions.softmax(sg_score, axis=0)
        sg_probs = sg_probs.array
        sg_grasp = sg_probs[1, :, :]
        sg_grasp = seg_prob * sg_grasp[None]
        sg_label_prob = sg_grasp.max(axis=(1, 2))
        sg_labels = sg_label_prob.argsort()[::-1]
        sg_label = None
        for lbl in sg_labels:
            if lbl not in self.giveup_labels['single']:
                sg_label = lbl
                break
        if sg_label == 0 or sg_label is None:
            sg_mask = cupy.zeros(sg_grasp.shape[1:], dtype=np.int32)
        else:
            sg_mask = sg_probs[1, :, :]
            sg_mask[seg_prob[sg_label] < self.score_thresh] = 0.0
            # if self.grasp_thresh > sg_mask.max():
            sg_thresh = sg_mask.max() - 0.05
            # else:
            #     sg_thresh = self.grasp_thresh
            sg_mask = (sg_mask > sg_thresh).astype(np.int32)
            sg_mask = sg_mask * 255

        # dualarm grasp
        dg_probs = chainer.functions.softmax(dg_score, axis=0)
        dg_probs = dg_probs.array
        dg_grasp = dg_probs[1, :, :]
        dg_grasp = seg_prob * dg_grasp[None]
        dg_label_prob = dg_grasp.max(axis=(1, 2))
        dg_labels = dg_label_prob.argsort()[::-1]
        dg_label = None
        for lbl in dg_labels:
            if lbl not in self.giveup_labels['dual']:
                dg_label = lbl
                break
        if dg_label == 0 or dg_label is None:
            dg_mask = cupy.zeros(dg_grasp.shape[1:], dtype=np.int32)
        else:
            dg_mask = dg_probs[1, :, :]
            dg_mask[seg_prob[dg_label] < self.score_thresh] = 0.0
            # if self.grasp_thresh > dg_mask.max():
            dg_thresh = dg_mask.max() - 0.05
            # else:
            #     dg_thresh = self.grasp_thresh
            dg_mask = (dg_mask > dg_thresh).astype(np.int32)
            dg_mask = dg_mask * 255

        # GPU -> CPU
        seg_prob = chainer.cuda.to_cpu(seg_prob)
        label_pred = chainer.cuda.to_cpu(label_pred)
        sg_probs = chainer.cuda.to_cpu(sg_probs)
        sg_grasp = chainer.cuda.to_cpu(sg_grasp)
        sg_mask = chainer.cuda.to_cpu(sg_mask)
        sg_label = np.asscalar(chainer.cuda.to_cpu(sg_label))
        sg_label_prob = chainer.cuda.to_cpu(sg_label_prob)
        dg_probs = chainer.cuda.to_cpu(dg_probs)
        dg_grasp = chainer.cuda.to_cpu(dg_grasp)
        dg_mask = chainer.cuda.to_cpu(dg_mask)
        dg_label = np.asscalar(chainer.cuda.to_cpu(dg_label))
        dg_label_prob = chainer.cuda.to_cpu(dg_label_prob)

        # msg
        input_msg = bridge.cv2_to_imgmsg(
            bgr.astype(np.uint8), encoding='bgr8')
        input_msg.header = imgmsg.header
        seg_msg = bridge.cv2_to_imgmsg(seg_prob.astype(np.float32))
        seg_msg.header = imgmsg.header
        label_msg = bridge.cv2_to_imgmsg(
            label_pred.astype(np.int32), '32SC1')
        label_msg.header = imgmsg.header

        sg_grasp_msg = bridge.cv2_to_imgmsg(sg_grasp.astype(np.float32))
        sg_grasp_msg.header = imgmsg.header
        sg_mask_msg = bridge.cv2_to_imgmsg(
            sg_mask.astype(np.uint8), 'mono8')
        sg_mask_msg.header = imgmsg.header
        sg_label_msg = ClassificationResult(
            header=imgmsg.header,
            labels=[sg_label],
            label_names=[self.label_names[sg_label]],
            label_proba=[sg_label_prob[sg_label]],
            probabilities=sg_label_prob,
            classifier='DualarmGraspFCN32s',
            target_names=self.label_names)

        dg_grasp_msg = bridge.cv2_to_imgmsg(dg_grasp.astype(np.float32))
        dg_grasp_msg.header = imgmsg.header
        dg_mask_msg = bridge.cv2_to_imgmsg(
            dg_mask.astype(np.uint8), 'mono8')
        dg_mask_msg.header = imgmsg.header
        dg_label_msg = ClassificationResult(
            header=imgmsg.header,
            labels=[dg_label],
            label_names=[self.label_names[dg_label]],
            label_proba=[dg_label_prob[dg_label]],
            probabilities=dg_label_prob,
            classifier='DualarmGraspFCN32s',
            target_names=self.label_names)

        if self.sampling:
            if len(self.candidates) != 2:
                rospy.logerr('Invalid tote contents num: {}'.format(
                             self.candidates))
            style = self.grasping_way
            label_mask = label_pred == self.candidates[1]
            if style == 'single':
                class_msg = sg_label_msg
                sample_mask = self._random_sample_single(
                    sg_probs[1, :, :], label_mask)
            elif style == 'dual':
                class_msg = dg_label_msg
                sample_mask = self._random_sample_dual(
                    dg_probs[1, :, :], label_mask)
            else:
                rospy.logerr('Invalid sampling style: {}'.format(style))
            mask_msg = bridge.cv2_to_imgmsg(
                sample_mask.astype(np.uint8), 'mono8')
            mask_msg.header = imgmsg.header
        else:
            # select grasp style
            if sg_label_prob[sg_label] > dg_label_prob[dg_label]:
                style = 'single'
                class_msg = sg_label_msg
                mask_msg = sg_mask_msg
            else:
                style = 'dual'
                class_msg = dg_label_msg
                mask_msg = dg_mask_msg
        grasp_class = GraspClassificationResult()
        grasp_class.header = imgmsg.header
        grasp_class.style = style
        grasp_class.is_target = True
        grasp_class.classification = class_msg

        # publish
        self.pub_input.publish(input_msg)
        self.pub_seg.publish(seg_msg)
        self.pub_label.publish(label_msg)

        self.pub_sg_grasp.publish(sg_grasp_msg)
        self.pub_sg_mask.publish(sg_mask_msg)
        self.pub_sg_label.publish(sg_label_msg)

        self.pub_dg_grasp.publish(dg_grasp_msg)
        self.pub_dg_mask.publish(dg_mask_msg)
        self.pub_dg_label.publish(dg_label_msg)

        self.pub_mask.publish(mask_msg)
        self.pub_grasp.publish(grasp_class)
        time.sleep(1.0)

    def _get_another(self, req):
        grasp_style = req.style
        label = req.label
        self.giveup_labels[grasp_style].append(label)
        response = GetAnotherResponse()
        response.success = True
        return response

    def _reset(self, req):
        self.giveup_labels = {
            'single': [self.bg_index],
            'dual': [self.bg_index],
        }
        response = TriggerResponse()
        response.success = True
        return response

    def _dyn_callback(self, config, level):
        self.score_thresh = config.score_thresh
        self.grasp_thresh = config.grasp_thresh
        self.sampling_thresh = config.sampling_thresh
        self.grasping_way = config.grasping_way
        return config

    def _random_sample_single(self, sg_prob, label_mask):
        weight = sg_prob
        weight[~label_mask] = 0.0
        if self.sampling_weighted and np.sum(weight) > 0:
            weight = weight.ravel() / np.sum(weight)
        elif np.sum(label_mask) > 0:
            label_mask = label_mask.astype(np.float32)
            weight = label_mask.ravel() / np.sum(label_mask)
        else:
            weight = None
        sample_grasp = np.zeros(sg_prob.shape)
        sampled_i = np.random.choice(sg_prob.size, p=weight)
        sampled_index = (
            sampled_i // sg_prob.shape[1],
            sampled_i % sg_prob.shape[1],
        )
        sample_grasp[sampled_index] = 255
        sample_grasp = scipy.ndimage.filters.gaussian_filter(
            sample_grasp, sigma=20)
        sample_grasp = sample_grasp / sample_grasp.max()
        sample_mask = sample_grasp > self.sampling_thresh
        sample_mask = sample_mask.astype(np.uint8) * 255
        return sample_mask

    def _random_sample_dual(self, dg_prob, label_mask):
        indices = np.column_stack(np.where(label_mask))

        c_masks = []
        if indices.size > 1:
            kmeans = KMeans(n_clusters=2)
            try:
                kmeans.fit(indices)
                centers = kmeans.cluster_centers_
                labels = kmeans.labels_
                for label, center in enumerate(centers):
                    center = np.round(center).astype(np.int32)
                    c_mask = np.zeros(label_mask.shape).astype(bool)
                    masked_indices = indices[labels == label]
                    masked_indices = (
                        masked_indices[:, 0], masked_indices[:, 1])
                    c_mask[masked_indices] = True
                    weight = dg_prob.copy()
                    weight[~c_mask] = 0.0
                    if self.sampling_weighted and np.sum(weight) > 0:
                        weight = weight.ravel() / np.sum(weight)
                    elif np.sum(label_mask) > 0:
                        label_mask = label_mask.astype(np.float32)
                        weight = label_mask.ravel() / np.sum(label_mask)
                    else:
                        weight = None
                    dual_giveup = False
                    trial_num = 10
                    for i in range(0, trial_num):
                        sampled_i = np.random.choice(dg_prob.size, p=weight)
                        sampled_index = (
                            sampled_i // dg_prob.shape[1],
                            sampled_i % dg_prob.shape[1],
                        )
                        c_grasp = np.zeros(dg_prob.shape)
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
            sample_mask = np.zeros(dg_prob.shape, dtype=np.uint8)

        return sample_mask


if __name__ == '__main__':
    rospy.init_node('dualarm_grasp_segmentation')
    node = DualarmGraspSegmentation()
    rospy.spin()
