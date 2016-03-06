#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Extract BoF histogram in realtime.
"""
import glob
import caffe
import cv_bridge
import numpy as np
import cv2
import skimage

from jsk_recognition_msgs.msg import ClassificationResult
from jsk_topic_tools import ConnectionBasedTransport
import jsk_recognition_utils
import message_filters
from sensor_msgs.msg import Image
import rospy

import jsk_apc2015_common

class CnnObjectMatcher(ConnectionBasedTransport):
    def __init__(self):
        super(CnnObjectMatcher, self).__init__()
        self._init_classifier()
        self.queue_size = rospy.get_param('~queue_size', 10)
        self._pub = self.advertise('~output', ClassificationResult, queue_size=1)

    def _init_classifier(self):
        input_scale,raw_scale,mean,channel_swap = None,None,None,None
        self.center_only = None
        gpu = False
        if gpu:
            caffe.set_mode_gpu()
        else:
            caffe.set_mode_cpu()

        model_def = rospy.get_param('~model_def')
        pretrained_model = rospy.get_param('~pretrained_model')
        self.image_dims =rospy.get_param('~size',(32,32))
        self.matcher = caffe.Classifier(model_def, pretrained_model,
                                           image_dims=self.image_dims, mean=mean,
                                           input_scale=input_scale, raw_scale=raw_scale,
                                           channel_swap=channel_swap)


    def subscribe(self):
        self._sub_img = message_filters.Subscriber('~input', Image)
        self._sub_label = message_filters.Subscriber('~input_label', Image)
        use_async = rospy.get_param('~approximate_sync', False)
        if use_async:
            sync = message_filters.ApproximateTimeSynchronizer(
                [self._sub_img, self._sub_label],
                queue_size=self.queue_size, slop=0.1)
        else:
            sync = message_filters.TimeSynchronizer(
                [self._sub_img, self._sub_label],
                queue_size=self.queue_size)
        sync.registerCallback(self._apply)

    def unsubscribe(self):
        self._sub_img.unregister()
        self._sub_label.unregister()

    def _apply(self, img_msg, label_msg):
        bridge = cv_bridge.CvBridge()
        input_label = bridge.imgmsg_to_cv2(label_msg)
        input_image = bridge.imgmsg_to_cv2(img_msg,'bgr8')##Do not mix it with rgb
        input_image = input_image * 255
        region_imgs = []
        for l in np.unique(input_label):
            if l == 0:
                continue
            mask = (input_label == l)
            region = jsk_recognition_utils.bounding_rect_of_mask(
                input_image, mask)
            region = skimage.img_as_float(region).astype(np.float32)
            region_imgs.append(region)

        y_proba = self.matcher.predict(region_imgs, not self.center_only)
        target_names = np.array(jsk_apc2015_common.get_object_list())
        y_pred = np.argmax(y_proba, axis=-1)
        label_proba = [p[i] for p, i in zip(y_proba, y_pred)]
        res = ClassificationResult()
        res.header = img_msg.header
        res.labels = y_pred
        res.label_names = target_names[y_pred]
        res.label_proba = label_proba
        res.probabilities = y_proba.reshape(-1)
        res.classifier = '<jsk_2016_01_baxter_apc.CnnClassifier>'
        res.target_names = target_names

        self._pub.publish(res)


if __name__ == '__main__':
    rospy.init_node('cnn_object_matcher')
    object_matcher = CnnObjectMatcher()
    rospy.spin()
