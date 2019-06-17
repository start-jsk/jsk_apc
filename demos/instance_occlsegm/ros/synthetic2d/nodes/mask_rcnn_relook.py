#!/usr/bin/env python

import chainer
import numpy as np

import chainer_mask_rcnn as mrcnn

import instance_occlsegm_lib
from instance_occlsegm_lib.contrib import synthetic2d

import cv_bridge
import rospy
from sensor_msgs.msg import Image

from synthetic2d.cfg import MaskRCNNRelookConfig

from place_planning import get_place_mask


class MaskRCNNRelook(object):

    def __init__(self):
        min_size = 600
        max_size = 1000
        anchor_scales = [4, 8, 16, 32]
        proposal_creator_params = dict(
            n_train_pre_nms=12000,
            n_train_post_nms=2000,
            n_test_pre_nms=6000,
            n_test_post_nms=1000,
            min_size=0,
        )
        pooling_func = mrcnn.functions.roi_align_2d

        self._fg_class_names = \
            instance_occlsegm_lib.datasets.apc.\
            ARC2017SemanticSegmentationDataset.class_names[1:]
        self._model = synthetic2d.models.MaskRCNNResNet(
            # n_layers=50,
            # pretrained_model='/home/wkentaro/instance_occlsegm_lib/examples/synthetic2d/logs/train_mrcnn_lbl/20180220_201514/snapshot_model.npz',  # NOQA
            n_layers=101,
            pretrained_model='/home/wkentaro/instance_occlsegm_lib/examples/synthetic2d/logs/train_mrcnn_lbl/20180225_141226/snapshot_model.npz',  # NOQA
            n_fg_class=40,
            pooling_func=pooling_func,
            anchor_scales=anchor_scales,
            proposal_creator_params=proposal_creator_params,
            min_size=min_size,
            max_size=max_size,
            mask_loss='softmax_relook_softmax+',
        )
        self._model.use_preset('visualize')
        # self._model.nms_thresh = 0.3
        # self._model.score_thresh = 0.6

        chainer.cuda.get_device_from_id(0).use()
        self._model.to_gpu()

        from dynamic_reconfigure.server import Server
        self._dynparam_server = Server(
            MaskRCNNRelookConfig, self._dynparam_callback)

        self._mask = None
        self._label = None
        from synthetic2d.srv import IsTarget
        self._server_is_target = rospy.Service(
            '~is_target', IsTarget, self._is_target_cb)
        self._pub_target = rospy.Publisher(
            '~output/target_mask', Image, queue_size=1)
        self._pub_place = rospy.Publisher(
            '~output/place_mask', Image, queue_size=1)
        self._pub_viz = rospy.Publisher('~output/viz', Image, queue_size=1)
        import message_filters
        self._sub = message_filters.Subscriber(
            '~input', Image, queue_size=1, buff_size=2 ** 24)
        self._sub_mask = message_filters.Subscriber(
            '~input/mask', Image, queue_size=1, buff_size=2 ** 24)
        sync = message_filters.ApproximateTimeSynchronizer(
            [self._sub, self._sub_mask],
            queue_size=10, slop=0.1
        )
        sync.registerCallback(self._callback)

        self._stamp_to_target = []

    def _is_target_cb(self, req):
        from synthetic2d.srv import IsTargetResponse
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
        self._model.context = context
        return config

    def _callback(self, imgmsg, maskmsg):
        bridge = cv_bridge.CvBridge()
        img = bridge.imgmsg_to_cv2(imgmsg, desired_encoding='rgb8')
        mask_fg = bridge.imgmsg_to_cv2(maskmsg, desired_encoding='mono8')
        mask_fg = mask_fg > 127
        img_chw = img.transpose(2, 0, 1)
        bboxes, masks, labels, scores = self._model.predict_masks([img_chw])
        bboxes, masks, labels, scores = \
            bboxes[0], masks[0], labels[0], scores[0]

        # suppress visualization according to the visible ratio
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
        keep = visible_ratios > 0.1
        bboxes = bboxes[keep]
        masks = masks[keep]
        labels = labels[keep]
        scores = scores[keep]

        keep = [True] * len(bboxes)
        for i, mask_i in enumerate(masks):
            for j, mask_j in enumerate(masks):
                if i != j:
                    iou = mrcnn.utils.get_mask_overlap(
                        mask_i == 1, mask_j == 1)
                    if iou > 0.5:
                        if scores[i] < scores[j]:
                            keep[i] = False
                        else:
                            keep[j] = False
        bboxes = bboxes[keep]
        masks = masks[keep]
        labels = labels[keep]
        scores = scores[keep]

        captions = ['{label}:{name}: {score:.0%}, {ratio:.0%}'
                    .format(label=l, name=self._fg_class_names[l],
                            score=s, ratio=v)
                    for l, s, v in zip(labels, scores, visible_ratios)]
        # rospy.logerr('>' * 79)
        # for caption in captions:
        #     rospy.logerr(caption)
        # rospy.logerr('<' * 79)

        vizs = [img]
        # visible
        viz = mrcnn.utils.draw_instance_boxes(
            img, bboxes, labels, n_class=40, bg_class=-1,
            masks=masks == 1, captions=captions)
        vizs.append(viz)
        # occluded
        viz = mrcnn.utils.draw_instance_boxes(
            img, bboxes, labels, n_class=40, bg_class=-1,
            masks=masks == 2, captions=captions)
        vizs.append(viz)

        # rospy.logerr(self._target)
        if self._target < 0:
            return

        # rospy.logerr('>' * 79)
        # rospy.logerr('Target: {}, Found: {}'.format(self._target, labels))
        # rospy.logerr('<' * 79)

        def find_obstacles(target_id, bboxes, labels, masks):
            obstacles = []
            mask_occluded = masks[target_id] == 2
            mask_whole = np.isin(masks[target_id], [1, 2])
            occluded_ratio = 1. * mask_occluded.sum() / mask_whole.sum()
            rospy.logerr('occluded_ratio: {}:{}: {:%}'.format(
                labels[target_id], self._fg_class_names[labels[target_id]],
                occluded_ratio))
            if occluded_ratio > 0.1:
                for ins_id, (label, mask) in enumerate(zip(labels, masks)):
                    if ins_id == target_id:
                        continue
                    mask_occluded_by_this = np.bitwise_and(
                        mask == 1, mask_occluded)
                    try:
                        ratio_occluded_by_this = (
                            1. * mask_occluded_by_this.sum() /
                            mask_occluded.sum())
                    except ZeroDivisionError:
                        ratio_occluded_by_this = 0
                    if ratio_occluded_by_this > 0.1:
                        rospy.logerr(
                            'Target: {}:{} is occluded by {}:{}: {:%}'.format(
                                labels[target_id],
                                self._fg_class_names[labels[target_id]],
                                label,
                                self._fg_class_names[label],
                                ratio_occluded_by_this,
                            )
                        )
                        obstacles.append(ins_id)
            return obstacles

        pick = None
        if self._target in labels:
            target_id = np.where(labels == self._target)[0][0]
            max_level = 3
            obstacles_all = set()
            for _ in range(max_level):
                obstacles = find_obstacles(target_id, bboxes, labels, masks)
                rospy.logerr(obstacles)
                rospy.logerr(labels[obstacles])
                obstacles_all.union(obstacles)
                if obstacles:
                    target_id = obstacles[0]
                else:
                    break
            if obstacles:
                pick = obstacles[0]
            else:
                pick = target_id

        if pick is not None:
            ins_id = pick
            mask = masks[ins_id] == 1
            mask = mask.astype(np.uint8) * 255
            label = labels[ins_id]
        else:
            mask = self._mask
            label = self._label

        if pick is None:
            place_mask = None
        else:
            place_mask = get_place_mask(
                img, bboxes, labels, masks,
                list(obstacles_all), target_id, pick, n_times=5,
                mask_fg=mask_fg)

        if place_mask is None:
            place_mask = np.zeros(img.shape[:2], dtype=np.uint8)
        else:
            place_mask = place_mask.astype(np.uint8) * 255
        out_msg = bridge.cv2_to_imgmsg(place_mask, encoding='mono8')
        out_msg.header = imgmsg.header
        self._pub_place.publish(out_msg)

        if mask is not None:
            assert label is not None

            self._mask = mask
            self._label = label
            out_msg = bridge.cv2_to_imgmsg(mask, encoding='mono8')
            out_msg.header = imgmsg.header
            self._pub_target.publish(out_msg)

            self._stamp_to_target.append(
                (out_msg.header.stamp.to_nsec(), label))
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

            viz = img.copy()
            viz[place_mask < 127] = 0
            vizs.append(viz)

        viz = np.hstack(vizs)
        out_msg = bridge.cv2_to_imgmsg(viz, encoding='rgb8')
        out_msg.header = imgmsg.header
        self._pub_viz.publish(out_msg)


if __name__ == '__main__':
    rospy.init_node('mask_rcnn_relook')
    app = MaskRCNNRelook()
    rospy.spin()
