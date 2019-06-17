#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import chainer
import chainer.functions as F
import chainer.links as L
import chainer.serializers as S
from chainer import Variable
import numpy as np

import cv_bridge
from jsk_recognition_msgs.msg import ClassificationResult
from jsk_topic_tools import ConnectionBasedTransport
from jsk_topic_tools.log_utils import logerr_throttle
import message_filters
import rospy
from sensor_msgs.msg import Image


class DualarmAlex(chainer.Chain):
    def __init__(self, n_failure, n_class, threshold=0.5, pt_func=None):
        self.threshold = threshold
        self.pt_func = pt_func
        self.n_failure = n_failure
        super(DualarmAlex, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(3, 96, 11, stride=4, pad=4)
            self.bn1 = L.BatchNormalization(96)
            self.conv2 = L.Convolution2D(96, 256, 5, stride=1, pad=1)
            self.bn2 = L.BatchNormalization(256)
            self.conv3 = L.Convolution2D(256, 384, 3, stride=1, pad=1)
            self.conv4 = L.Convolution2D(384, 384, 3, stride=1, pad=1)
            self.conv5 = L.Convolution2D(384, 256, 3, stride=1, pad=1)
            self.bn5 = L.BatchNormalization(256)
            self.fc6_failure = L.Linear(33280, 4096)
            self.fc7_failure = L.Linear(4096, 4096)
            self.fc8_failure = L.Linear(4096, 2*n_failure)

            self.fc6_cls = L.Linear(33280, 4096)
            self.fc7_cls = L.Linear(4096, 4096)
            self.fc8_cls = L.Linear(4096, n_class)

    def __call__(self, x, t=None, t_cls=None):
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.max_pooling_2d(h, 3, stride=2)
        h = F.relu(self.bn2(self.conv2(h)))
        h = F.max_pooling_2d(h, 3, stride=2)
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.relu(self.bn5(self.conv5(h)))
        h = F.max_pooling_2d(h, 3, stride=3)
        conv4 = h

        # failure prediction
        h = F.dropout(F.relu(self.fc6_failure(conv4)), ratio=0.5)
        h = F.dropout(F.relu(self.fc7_failure(h)), ratio=0.5)
        h = self.fc8_failure(h)
        h = h.reshape((-1, 2, self.n_failure))
        fc8_failure = h

        fail_prob = F.softmax(fc8_failure, axis=1)[:, 1, :]
        self.fail_prob = fail_prob

        # classification prediction
        h = F.dropout(F.relu(self.fc6_cls(conv4)), ratio=0.5)
        h = F.dropout(F.relu(self.fc7_cls(h)), ratio=0.5)
        h = self.fc8_cls(h)
        cls_score = h
        self.cls_score = cls_score


class DualarmAlexProbaEstimation(ConnectionBasedTransport):

    def __init__(self):
        super(self.__class__, self).__init__()
        chainer.config.train = False
        chainer.config.enable_backprop = False

        self.gpu = rospy.get_param('~gpu', -1)
        model_h5 = rospy.get_param('~model_h5')
        self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.target_names = rospy.get_param('~target_names')
        self.model = DualarmAlex(n_class=10, n_failure=len(self.target_names))
        S.load_hdf5(model_h5, self.model)
        if self.gpu != -1:
            self.model.to_gpu(self.gpu)
        self.pub = self.advertise('~output', ClassificationResult,
                                  queue_size=1)
        self.pub_input = self.advertise(
            '~debug/net_input', Image, queue_size=1)

    def subscribe(self):
        # larger buff_size is necessary for taking time callback
        # http://stackoverflow.com/questions/26415699/ros-subscriber-not-up-to-date/29160379#29160379  # NOQA
        sub = message_filters.Subscriber(
            '~input', Image, queue_size=1, buff_size=2**24)
        sub_mask = message_filters.Subscriber(
            '~input/mask', Image, queue_size=1, buff_size=2**24)
        self.subs = [sub, sub_mask]
        queue_size = rospy.get_param('~queue_size', 10)
        if rospy.get_param('~approximate_sync', False):
            slop = rospy.get_param('~slop', 0.1)
            sync = message_filters.ApproximateTimeSynchronizer(
                self.subs, queue_size=queue_size, slop=slop)
        else:
            sync = message_filters.TimeSynchronizer(
                self.subs, queue_size=queue_size)
        sync.registerCallback(self._recognize)

    def unsubscribe(self):
        for sub in self.subs:
            sub.unregister()

    def _recognize(self, imgmsg, mask_msg=None):
        bridge = cv_bridge.CvBridge()
        bgr = bridge.imgmsg_to_cv2(imgmsg, desired_encoding='bgr8')
        if mask_msg is not None:
            mask = bridge.imgmsg_to_cv2(mask_msg)
            if mask.shape != bgr.shape[:2]:
                logerr_throttle(10,
                                'Size of input image and mask is different')
                return
            elif mask.size == 0:
                logerr_throttle(10, 'Size of input mask is 0')
                return
            bgr[mask < 128] = self.mean_bgr
        input_msg = bridge.cv2_to_imgmsg(bgr.astype(np.uint8), encoding='bgr8')
        input_msg.header = imgmsg.header
        self.pub_input.publish(input_msg)

        blob = (bgr - self.mean_bgr).transpose((2, 0, 1))
        x_data = np.array([blob], dtype=np.float32)
        if self.gpu != -1:
            x_data = chainer.cuda.to_gpu(x_data, device=self.gpu)
        x = Variable(x_data)

        self.model(x)

        proba = chainer.cuda.to_cpu(self.model.fail_prob.data)[0]
        cls_msg = ClassificationResult(
            header=imgmsg.header,
            labels=None,
            label_names=None,
            label_proba=None,
            probabilities=proba,
            target_names=self.target_names)
        self.pub.publish(cls_msg)


if __name__ == '__main__':
    rospy.init_node('dualarm_alex_proba_estimation')
    app = DualarmAlexProbaEstimation()
    rospy.spin()
