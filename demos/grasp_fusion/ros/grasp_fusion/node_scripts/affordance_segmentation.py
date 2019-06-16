#!/usr/bin/env python

from threading import Lock

from chainer import cuda
import numpy as np

import grasp_fusion_lib
from grasp_fusion_lib.contrib import grasp_fusion

import cv_bridge
from jsk_topic_tools.transport import ConnectionBasedTransport
import message_filters
import rospy
from sensor_msgs.msg import Image
from std_srvs.srv import Empty
from std_srvs.srv import EmptyResponse


def get_pretrained_model(affordance):
    if affordance == 'suction':
        url = 'https://drive.google.com/uc?id=1wTrWCPP2IuPzk06XQzn9oLJKk9iS1H7O'  # NOQA
        out_channels = 1
        modal = 'rgb+depth'
    else:
        assert affordance == 'pinch'
        url = 'https://drive.google.com/uc?id=16NNJHVGja4NEW1LYRDYedJ0baqR6JXyZ'  # NOQA
        modal = 'rgb+depth'
        out_channels = 12
    pretrained_model = grasp_fusion_lib.data.download(url)
    return pretrained_model, modal, out_channels


class AffordanceSegmentation(ConnectionBasedTransport):

    def __init__(self):
        super(AffordanceSegmentation, self).__init__()

        affordance = rospy.get_param('~affordance')

        pretrained_model, modal, out_channels = get_pretrained_model(
            affordance
        )

        gpu = rospy.get_param('~gpu', 0)

        self.lock = Lock()

        # rospy.set_param('~always_subscribe', True)

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

        self._prob = None
        self.srv_reset = rospy.Service('~reset', Empty, self.reset_callback)

    def subscribe(self):
        sub_rgb = message_filters.Subscriber(
            '~input/rgb', Image, queue_size=1, buff_size=2**24,
        )
        sub_depth = message_filters.Subscriber(
            '~input/depth', Image, queue_size=1, buff_size=2**24,
        )
        self.subs = [sub_rgb, sub_depth]
        sync = message_filters.TimeSynchronizer(
            fs=self.subs, queue_size=100
        )
        sync.registerCallback(self.callback)

    def unsubscribe(self):
        for sub in self.subs:
            sub.unregister()

    def reset_callback(self, req):
        with self.lock:
            self._prob = None
        return EmptyResponse()

    def callback(self, imgmsg, depthmsg):
        bridge = cv_bridge.CvBridge()
        img = bridge.imgmsg_to_cv2(imgmsg, desired_encoding='rgb8')
        depth = bridge.imgmsg_to_cv2(depthmsg, desired_encoding='32FC1')

        prob = self.model.predict_proba([img.transpose(2, 0, 1)], [depth])[0]

        prob = prob.transpose(1, 2, 0)
        self.lock.acquire()
        if self._prob is not None:
            score1 = np.log(prob / (1 - prob))
            score2 = np.log(self._prob / (1 - self._prob))
            score = score1 + score2
            prob = 1. / (1 + np.exp(-score))
        self._prob = prob
        self.lock.release()

        lbl = self.model.proba_to_lbls([prob.transpose(2, 0, 1)])[0]
        lbl = lbl.transpose(1, 2, 0)

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
    rospy.init_node('suction_affordance_segmentation')
    node = AffordanceSegmentation()
    rospy.spin()
