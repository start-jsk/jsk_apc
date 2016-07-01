#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from chainer import cuda
import chainer.serializers as S
from chainer import Variable
from fcn.models import FCN32s
import numpy as np

import jsk_apc2016_common
from jsk_topic_tools import ConnectionBasedTransport
from jsk_topic_tools.log_utils import logwarn_throttle
from jsk_topic_tools.log_utils import jsk_logwarn
import message_filters
import rospy
from sensor_msgs.msg import Image


class FCNMaskForLabelNames(ConnectionBasedTransport):

    mean_bgr = np.array((104.00698793, 116.66876762, 122.67891434))

    def __init__(self):
        super(self.__class__, self).__init__()

        # set target_names
        self.target_names = ['background'] + \
            [datum['name']
             for datum in jsk_apc2016_common.get_object_data()]
        n_class = len(self.target_names)
        assert n_class == 40

        # load model
        self.gpu = rospy.get_param('~gpu', 0)
        chainermodel = rospy.get_param('~chainermodel')
        self.model = FCN32s(n_class=n_class)
        S.load_hdf5(chainermodel, self.model)
        if self.gpu != -1:
            self.model.to_gpu(self.gpu)
        jsk_logwarn('>> Model is loaded <<')

        while True:
            self.tote_contents = rospy.get_param('~tote_contents', None)
            if self.tote_contents is not None:
                break
            logwarn_throttle(10, 'param ~tote_contents is not set. Waiting..')
            rospy.sleep(0.1)
        self.label_names = rospy.get_param('~label_names')
        self.negative = rospy.get_param('~negative', False)
        jsk_logwarn('>> Param is set <<')

        self.pub = self.advertise('~output', Image, queue_size=1)
        self.pub_debug = self.advertise('~debug', Image, queue_size=1)

    def subscribe(self):
        self.sub_img = message_filters.Subscriber('~input', Image)
        self.sub_mask = message_filters.Subscriber('~input/mask', Image)
        sync = message_filters.ApproximateTimeSynchronizer(
            [self.sub_img, self.sub_mask], queue_size=100, slop=0.1)
        sync.registerCallback(self._callback)

    def unsubscribe(self):
        self.sub_img.unregister()
        self.sub_mask.unregister()

    def _callback(self, img_msg, mask_msg):
        jsk_logwarn('>> Start Processing <<')
        import cv_bridge
        bridge = cv_bridge.CvBridge()
        bgr_img = bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
        mask_img = bridge.imgmsg_to_cv2(mask_msg, desired_encoding='mono8')
        if mask_img.ndim == 3 and mask_img.shape[2] == 1:
            mask_img = mask_img.reshape(mask_img.shape[:2])
        if mask_img.shape != bgr_img.shape[:2]:
            from skimage.transform import resize
            jsk_logwarn('Size of mask and color image is different.'
                        'Resizing.. mask {0} to {1}'
                        .format(mask_img.shape, bgr_img.shape[:2]))
            mask_img = resize(mask_img, bgr_img,
                              preserve_range=True).astype(np.uint8)

        blob = bgr_img - self.mean_bgr
        blob = blob.transpose((2, 0, 1))

        import numpy as np
        x_data = np.array([blob], dtype=np.float32)
        if self.gpu != -1:
            x_data = cuda.to_gpu(x_data, device=self.gpu)
        x = Variable(x_data, volatile=True)
        self.model(x)
        pred_datum = cuda.to_cpu(self.model.score.data[0])

        candidate_labels = [self.target_names.index(name)
                            for name in self.tote_contents]
        label_pred_in_candidates = pred_datum[candidate_labels].argmax(axis=0)
        label_pred = np.zeros_like(label_pred_in_candidates)
        for idx, label_val in enumerate(candidate_labels):
            label_pred[label_pred_in_candidates == idx] = label_val
        label_pred[mask_img == 0] = 0  # set bg_label

        from skimage.color import label2rgb
        label_viz = label2rgb(label_pred, bgr_img, bg_label=0)
        label_viz = (label_viz * 255).astype(np.uint8)
        debug_msg = bridge.cv2_to_imgmsg(label_viz, encoding='rgb8')
        debug_msg.header = img_msg.header
        self.pub_debug.publish(debug_msg)

        output_mask = np.zeros(mask_img.shape, dtype=bool)
        for label_val, label_name in enumerate(self.target_names):
            if label_name in self.label_names:
                output_mask[label_pred == label_val] = True
        if self.negative:
            output_mask = ~output_mask
        output_mask = output_mask.astype(np.uint8)
        output_mask[output_mask == 1] = 255
        output_mask_msg = bridge.cv2_to_imgmsg(output_mask, encoding='mono8')
        output_mask_msg.header = img_msg.header
        self.pub.publish(output_mask_msg)
        jsk_logwarn('>> Finshed processing <<')


if __name__ == '__main__':
    rospy.init_node('fcn_mask_for_label_names')
    FCNMaskForLabelNames()
    rospy.spin()
