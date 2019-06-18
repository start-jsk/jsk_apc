#!/usr/bin/env python

from chainer import cuda
import numpy as np

import grasp_fusion_lib
from grasp_fusion_lib.contrib import grasp_fusion

import cv_bridge
from jsk_topic_tools.transport import ConnectionBasedTransport
import message_filters
import rospy
from sensor_msgs.msg import Image

from affordance_segmentation import get_pretrained_model


class SoleAffordanceSegmentation(ConnectionBasedTransport):

    def __init__(self):
        super(SoleAffordanceSegmentation, self).__init__()

        affordance = rospy.get_param('~affordance')

        pretrained_model, modal, out_channels = get_pretrained_model(
            affordance
        )
        self.out_channels = out_channels

        gpu = rospy.get_param('~gpu', 0)

        model = grasp_fusion.models.FCN8sVGG16Sigmoid(
            out_channels=out_channels,
            pretrained_model=pretrained_model,
            modal=modal,
        )
        if gpu >= 0:
            cuda.get_device_from_id(gpu).use()
            model.to_gpu()

        self.model = model

        self.pub_label = self.advertise('~output/label', Image, queue_size=1)
        self.pub_prob = self.advertise('~output/prob', Image, queue_size=1)
        self.pub_viz = self.advertise('~output/viz', Image, queue_size=1)

    def subscribe(self):
        sub_rgb = message_filters.Subscriber(
            '~input/rgb', Image, queue_size=1, buff_size=2**24,
        )
        sub_depth = message_filters.Subscriber(
            '~input/depth', Image, queue_size=1, buff_size=2**24,
        )
        sub_ins = message_filters.Subscriber(
            '~input/label_ins', Image, queue_size=1, buff_size=2**24,
        )
        self.subs = [sub_rgb, sub_depth, sub_ins]
        sync = message_filters.TimeSynchronizer(
            fs=self.subs, queue_size=100
        )
        sync.registerCallback(self.callback)

    def unsubscribe(self):
        for sub in self.subs:
            sub.unregister()

    def callback(self, imgmsg, depthmsg, insmsg):
        bridge = cv_bridge.CvBridge()
        img = bridge.imgmsg_to_cv2(imgmsg, desired_encoding='rgb8')
        depth = bridge.imgmsg_to_cv2(depthmsg, desired_encoding='32FC1')
        ins = bridge.imgmsg_to_cv2(insmsg, desired_encoding='32SC1')

        # viz_depth = grasp_fusion_lib.image.colorize_depth(
        #     depth, 0, self.model.depth_max_value
        # )
        # viz_ins = grasp_fusion_lib.image.label2rgb(ins + 1, img)
        # viz = grasp_fusion_lib.image.tile([img, viz_depth, viz_ins])
        # vizmsg = bridge.cv2_to_imgmsg(viz, encoding='rgb8')
        # vizmsg.header = imgmsg.header
        # self.pub_viz.publish(vizmsg)

        instance_ids = np.unique(ins)
        instance_ids = instance_ids[instance_ids != -1]
        assert (instance_ids >= 0).all()

        sole_imgs = []
        sole_depths = []
        for ins_id in instance_ids:
            mask = np.isin(ins, (-1, ins_id))
            img_i = img.copy()
            depth_i = depth.copy()
            img_i[~mask] = 0
            depth_i[~mask] = 0
            sole_imgs.append(img_i.transpose(2, 0, 1))
            sole_depths.append(depth_i)

        probs = self.model.predict_proba(sole_imgs, sole_depths)
        lbls = self.model.proba_to_lbls(probs)
        del sole_imgs, sole_depths

        H, W = img.shape[:2]
        prob_all = np.zeros((H, W, self.out_channels), dtype=np.float32)
        lbl_all = np.zeros((H, W, self.out_channels), dtype=np.int32)
        for ins_id, prob, lbl in zip(instance_ids, probs, lbls):
            mask = ins == ins_id
            prob = prob.transpose(1, 2, 0)
            lbl = lbl.transpose(1, 2, 0)
            prob_all[mask] = prob[mask]
            lbl_all[mask] = lbl[mask]
        del probs, lbls

        prob = prob_all
        lbl = lbl_all
        del prob_all, lbl_all

        lblmsg = bridge.cv2_to_imgmsg(lbl)
        lblmsg.header = imgmsg.header
        self.pub_label.publish(lblmsg)

        vizs = []
        for c in range(lbl.shape[2]):
            lbl_c = lbl[:, :, c]
            viz = grasp_fusion_lib.image.label2rgb(lbl_c, img, alpha=0.7)
            vizs.append(viz)
        viz_lbl = grasp_fusion_lib.image.tile(vizs, boundary=True)

        probmsg = bridge.cv2_to_imgmsg(prob)
        probmsg.header = imgmsg.header
        self.pub_prob.publish(probmsg)

        vizs = []
        for c in range(prob.shape[2]):
            prob_c = prob[:, :, c]
            viz = grasp_fusion_lib.image.colorize_heatmap(prob_c)
            viz = grasp_fusion_lib.image.overlay_color_on_mono(
                img_color=viz, img_mono=img, alpha=0.7
            )
            vizs.append(viz)
        viz_prob = grasp_fusion_lib.image.tile(vizs, boundary=True)

        img = grasp_fusion_lib.image.resize(img, height=viz_lbl.shape[0])
        depth = grasp_fusion_lib.image.colorize_depth(
            depth, min_value=0, max_value=self.model.depth_max_value
        )
        depth = grasp_fusion_lib.image.resize(depth, height=viz_lbl.shape[0])
        viz = np.hstack((img, depth, viz_prob, viz_lbl))

        vizmsg = bridge.cv2_to_imgmsg(viz, encoding='rgb8')
        vizmsg.header = imgmsg.header
        self.pub_viz.publish(vizmsg)


if __name__ == '__main__':
    rospy.init_node('sole_affordance_segmentation')
    node = SoleAffordanceSegmentation()
    rospy.spin()
