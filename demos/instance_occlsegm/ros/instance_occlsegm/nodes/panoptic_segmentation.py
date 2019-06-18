#!/usr/bin/env python

from __future__ import print_function
import sys

import chainer
import numpy as np

import chainer_mask_rcnn as cmr
import yaml

import cv_bridge
from dynamic_reconfigure.server import Server
import message_filters
import rospy
from sensor_msgs.msg import Image

from synthetic2d.cfg import MaskRCNNRelookConfig
from synthetic2d.srv import IsTarget
from synthetic2d.srv import IsTargetResponse

import instance_occlsegm_lib
from instance_occlsegm_lib.contrib import instance_occlsegm


def get_visible_ratios(bboxes, masks):
    visible_ratios = []
    for bbox, mask in zip(bboxes, masks):
        y1, x1, y2, x2 = bbox
        mask_whole = np.isin(mask, [1, 2])
        mask_visible = mask == 1
        try:
            visible_ratio = 1. * mask_visible.sum() / mask_whole.sum()
        except ZeroDivisionError:
            visible_ratio = 0
        visible_ratios.append(visible_ratio)
    visible_ratios = np.asarray(visible_ratios)
    return visible_ratios


def suppress_by_visible_ratio(bboxes, masks, labels, scores):
    if len(bboxes) == 0:
        return bboxes, masks, labels, scores
    visible_ratios = get_visible_ratios(bboxes, masks)
    keep = visible_ratios > 0.1
    bboxes = bboxes[keep]
    masks = masks[keep]
    labels = labels[keep]
    scores = scores[keep]
    return bboxes, masks, labels, scores


def nms_masks(bboxes, masks, labels, scores, nms_thresh=0.3):
    if len(bboxes) == 0:
        return bboxes, masks, labels, scores
    keep = [True] * len(bboxes)
    for i, mask_i in enumerate(masks):
        for j, mask_j in enumerate(masks):
            if i == j:
                continue
            iou = cmr.utils.get_mask_overlap(
                mask_i == 1, mask_j == 1
            )
            if iou > 0.3:
                if scores[i] < scores[j]:
                    keep[i] = False
                else:
                    keep[j] = False
    bboxes = bboxes[keep]
    masks = masks[keep]
    labels = labels[keep]
    scores = scores[keep]
    return bboxes, masks, labels, scores


def apply_mask(bboxes, masks, labels, scores, mask_fg):
    if len(bboxes) == 0:
        return bboxes, masks, labels, scores
    assert mask_fg.dtype == bool
    keep = [True] * len(bboxes)
    for i, mask in enumerate(masks):
        ratio = (
            1. * np.bitwise_and(mask == 1, ~mask_fg).sum() / (mask == 1).sum()
        )
        if ratio > 0.9:
            keep[i] = False
    keep = np.asarray(keep)
    bboxes = bboxes[keep]
    masks = masks[keep]
    labels = labels[keep]
    scores = scores[keep]
    return bboxes, masks, labels, scores


def get_files(baseline):
    if baseline:
        # no joint learning
        yaml_file = instance_occlsegm_lib.data.download(
            url='https://drive.google.com/uc?id=1Ph52kKbi8WOS70G1H54kzsxBISx7MXAo'  # NOQA
        )
        pretrained_model = instance_occlsegm_lib.data.download(
            url='https://drive.google.com/uc?id=1uaKP7yil-Z-c_Cf7WALgxHOy8gWi3fkG'  # NOQA
        )
    else:
        # joint learning
        yaml_file = instance_occlsegm_lib.data.download(
            url='https://drive.google.com/uc?id=1UIKArWE1rxY8OKPcwKGxr3OFfDotR8cT'  # NOQA
        )
        pretrained_model = instance_occlsegm_lib.data.download(
            url='https://drive.google.com/uc?id=1QZu2z1vJWDnEBGGhbYgSnEOrK823iFDR'  # NOQA
        )
    return yaml_file, pretrained_model


class PanopticSegmentation(object):

    def __init__(self):
        baseline = rospy.get_param('~baseline', False)

        yaml_file, pretrained_model = get_files(baseline)

        with open(yaml_file) as f:
            params = yaml.load(f)
        assert params['pooling_func'] == 'align'
        pooling_func = cmr.functions.roi_align_2d

        self._fg_class_names = \
            instance_occlsegm.datasets.PanopticOcclusionSegmentationDataset(
                'train'
            ).class_names
        self._model = instance_occlsegm.models.MaskRCNNPanopticResNet(
            n_layers=50,
            n_fg_class=40,
            pretrained_model=pretrained_model,
            pooling_func=pooling_func,
            anchor_scales=params['anchor_scales'],
            min_size=params['min_size'],
            max_size=params['max_size'],
            rpn_dim=params['rpn_dim'],
        )

        chainer.cuda.get_device_from_id(0).use()
        self._model.to_gpu()

        self._mask = None
        self._label = None
        self._context = []
        self._server_is_target = rospy.Service(
            '~is_target', IsTarget, self._is_target_cb
        )
        self._pub_target = rospy.Publisher(
            '~output/target_mask', Image, queue_size=1
        )
        self._pub_place = rospy.Publisher(
            '~output/place_mask', Image, queue_size=1
        )
        self._pub_viz = rospy.Publisher('~output/viz', Image, queue_size=1)
        self._sub = message_filters.Subscriber(
            '~input', Image, queue_size=1, buff_size=2 ** 24
        )
        self._sub_mask = message_filters.Subscriber(
            '~input/mask', Image, queue_size=1, buff_size=2 ** 24
        )
        sync = message_filters.ApproximateTimeSynchronizer(
            [self._sub, self._sub_mask], queue_size=50, slop=0.1
        )
        sync.registerCallback(self._callback)

        self._dynparam_server = Server(
            MaskRCNNRelookConfig, self._dynparam_callback
        )

        self._stamp_to_target = []

    def _callback(self, imgmsg, maskmsg):
        bridge = cv_bridge.CvBridge()
        img = bridge.imgmsg_to_cv2(imgmsg, desired_encoding='rgb8')
        mask_fg = bridge.imgmsg_to_cv2(maskmsg, desired_encoding='mono8')
        mask_fg = mask_fg > 127

        img_chw = img.transpose(2, 0, 1)
        bboxes, masks, labels, scores = self._model.predict([img_chw])[:4]
        bboxes, masks, labels, scores = \
            bboxes[0], masks[0], labels[0], scores[0]
        bboxes, masks, labels, scores = suppress_by_visible_ratio(
            bboxes, masks, labels, scores
        )
        bboxes, masks, labels, scores = nms_masks(
            bboxes, masks, labels, scores
        )
        bboxes, masks, labels, scores = apply_mask(
            bboxes, masks, labels, scores, mask_fg
        )

        if self._context:
            keep = np.isin(labels, self._context)
            bboxes = bboxes[keep]
            masks = masks[keep]
            labels = labels[keep]
            scores = scores[keep]

        print(self._context, bboxes)

        sort = np.argsort(scores)[::-1]
        bboxes = bboxes[sort]
        masks = masks[sort]
        labels = labels[sort]
        scores = scores[sort]

        # img[~mask_fg] = 0

        captions = [
            '{label}:{name}: {score:.0%}'
            .format(label=l, name=self._fg_class_names[l], score=s)
            for l, s in zip(labels, scores)
        ]
        for caption in captions:
            print(caption, file=sys.stderr)
        vizs = [img]
        # visible
        viz = cmr.utils.draw_instance_bboxes(
            img, bboxes, labels, n_class=40, bg_class=-1,
            masks=masks == 1, captions=captions)
        vizs.append(viz)
        # occluded
        viz = cmr.utils.draw_instance_bboxes(
            img, bboxes, labels, n_class=40, bg_class=-1,
            masks=masks == 2, captions=captions)
        vizs.append(viz)

        pick = None
        if len(bboxes) > 0 and self._target >= 0 and self._target in labels:
            max_level = 5
            target_id = np.where(labels == self._target)[0][0]
            obstacles_all = set()
            for _ in range(max_level):
                obstacles = instance_occlsegm.planning.pick.find_obstacles(
                    target_id, bboxes, labels, masks, self._fg_class_names
                )
                print(
                    'Obstacles: {}, Obstacle labels: {}'
                    .format(obstacles, labels[obstacles]),
                    file=sys.stderr,
                )
                obstacles_all.union(obstacles)
                if obstacles:
                    target_id = obstacles[0]
                else:
                    break
            if obstacles:
                pick = obstacles[0]
            else:
                pick = target_id

        place_mask = None
        if pick is None:
            if len(bboxes) == 0:
                mask = None
                label = None
            elif self._mask is None:
                assert self._label is None
                visible_ratios = get_visible_ratios(bboxes, masks)
                pick = np.argmax(visible_ratios >= 0.9)
                ins_id = pick
                mask = masks[ins_id] == 1
                mask = mask.astype(np.uint8) * 255
                label = labels[ins_id]
            else:
                mask = self._mask
                label = self._label
        else:
            ins_id = pick
            mask = masks[ins_id] == 1
            mask = mask.astype(np.uint8) * 255
            label = labels[ins_id]
            # if label != self._target:
            #     place_mask = instance_occlsegm.planning.place.find_space(
            #         img,
            #         bboxes,
            #         labels,
            #         masks,
            #         obstacles=list(obstacles_all),
            #         target_id=target_id,
            #         pick=pick,
            #         n_times=5,
            #         mask_fg=mask_fg,
            #     )

        if place_mask is None:
            place_mask = np.zeros(img.shape[:2], dtype=np.uint8)
        else:
            place_mask = place_mask.astype(np.uint8) * 255
        out_msg = bridge.cv2_to_imgmsg(place_mask, encoding='mono8')
        out_msg.header = imgmsg.header
        self._pub_place.publish(out_msg)

        if mask is None:
            vizs.append(np.zeros_like(img))
            # vizs.append(np.zeros_like(img))
        else:
            assert label is not None

            # self._mask = mask
            # self._label = label
            out_msg = bridge.cv2_to_imgmsg(mask, encoding='mono8')
            out_msg.header = imgmsg.header
            self._pub_target.publish(out_msg)

            self._stamp_to_target.append(
                (out_msg.header.stamp.to_nsec(), label)
            )
            stamp_to_target = []
            for nsec, target in self._stamp_to_target:
                if rospy.Duration(nsecs=rospy.Time.now().to_nsec() - nsec) > \
                        rospy.Duration(secs=60):
                    continue
                stamp_to_target.append((nsec, target))
            self._stamp_to_target = stamp_to_target

            viz = img.copy()
            viz[mask < 127] = 0
            vizs.append(viz)

            # viz = img.copy()
            # viz[place_mask < 127] = 0
            # vizs.append(viz)

        print('<' * 79, file=sys.stderr)

        viz = np.hstack(vizs)
        out_msg = bridge.cv2_to_imgmsg(viz, encoding='rgb8')
        out_msg.header = imgmsg.header
        self._pub_viz.publish(out_msg)

    def _is_target_cb(self, req):
        nsec = req.stamp.to_nsec()
        stamp_prev = None
        label_prev = None
        target_label = None
        for stamp, target in self._stamp_to_target:
            if stamp_prev is None:
                stamp_prev = stamp
                label_prev = target
                continue
            if stamp_prev == nsec:
                target_label = label_prev
                break
            elif stamp_prev < nsec < stamp:
                target_label = label_prev
                break
            elif nsec == stamp:
                target_label = target
                break
            stamp_prev = stamp
            label_prev = target
        print('target_label: ', target_label, 'self._target:', self._target)
        return IsTargetResponse(is_target=target_label == self._target)

    def _dynparam_callback(self, config, level):
        self._model.score_thresh = config.score_thresh
        self._model.nms_thresh = config.nms_thresh
        self._target = config.target
        context = []
        for l in config.context.split(','):
            try:
                context.append(int(l))
            except ValueError:
                pass
        self._context = context
        return config


if __name__ == '__main__':
    rospy.init_node('panoptic_segmentation')
    app = PanopticSegmentation()
    rospy.spin()
