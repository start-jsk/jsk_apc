#!/usr/bin/env python

import chainer
import fcn

import cv_bridge
from jsk_topic_tools import ConnectionBasedTransport
import rospy
from sensor_msgs.msg import Image

from mvtk.contrib.grasp_prediction_arc2017 import models


class FCNObjectSegmentation(ConnectionBasedTransport):

    def __init__(self):
        super(FCNObjectSegmentation, self).__init__()

        model_name = rospy.get_param('~model_name')
        n_class = rospy.get_param('~n_class')
        class_agnostic = rospy.get_param('~class_agnostic', False)
        if class_agnostic and model_name != 'fcn32s':
            rospy.logwarn(
                '~class_agnostic=True is only supported for fcn32s,'
                ' so ignoring it.'
            )
        if model_name == 'fcn32s':
            self.model = models.fcn32s.FCN32s(
                n_class=n_class, class_agnostic=class_agnostic
            )
        elif model_name == 'fcn8s_at_once':
            self.model = models.fcn8s.FCN8sAtOnce(n_class=n_class)
        else:
            raise ValueError('Unsupported ~model_name: {}'.format(model_name))

        chainer.global_config.enable_backprop = False
        chainer.global_config.train = False
        self.gpu = rospy.get_param('~gpu')

        # model_file = osp.expanduser('~/data/mvtk/grasp_prediction_arc2017/logs/fcn32s_CFG-000_VCS-2400e9e_TIME-20170827-233211/models/FCN32s_iter00044000.npz')  # NOQA
        # model_file = osp.expanduser('~/data/iros2018/system_inputs/ForItemDataBooks3/FCN32s_180215_210043_iter00050000.npz')  # NOQA
        model_file = rospy.get_param('~model_file')
        chainer.serializers.load_npz(model_file, self.model)

        if self.gpu >= 0:
            chainer.cuda.get_device_from_id(self.gpu).use()
            self.model.to_gpu()

        self.thresh_cls_prob = rospy.get_param('~thresh_class_prob', 0)
        self.thresh_suc_prob = rospy.get_param('~thresh_suction_prob', 0)
        self.pub_cls = self.advertise(
            '~output/label_class', Image, queue_size=1)
        self.pub_cls_prob = self.advertise(
            '~output/prob_class', Image, queue_size=1)
        self.pub_suc = self.advertise(
            '~output/label_suction', Image, queue_size=1)
        self.pub_suc_prob = self.advertise(
            '~output/prob_suction', Image, queue_size=1)

    def subscribe(self):
        self.sub = rospy.Subscriber('~input', Image, self._input_cb)

    def unsubscribe(self):
        self.sub.unregister()

    def _input_cb(self, imgmsg):
        bridge = cv_bridge.CvBridge()
        img = bridge.imgmsg_to_cv2(imgmsg, desired_encoding='rgb8')

        x_data, = fcn.datasets.transform_lsvrc2012_vgg16((img,))
        x_data = x_data[None, :, :, :]
        if self.gpu >= 0:
            x_data = chainer.cuda.to_gpu(x_data)
        x = chainer.Variable(x_data)

        self.model(x)

        prob_cls = chainer.functions.softmax(self.model.score_cls)
        lbl_cls = chainer.functions.argmax(prob_cls, axis=1)
        if self.thresh_cls_prob:
            lbl_cls.data[chainer.functions.max(prob_cls, axis=1).data <
                         self.thresh_cls_prob] = 0
        lbl_cls = chainer.cuda.to_cpu(lbl_cls.data[0])
        cls_msg = bridge.cv2_to_imgmsg(lbl_cls)
        cls_msg.header = imgmsg.header
        self.pub_cls.publish(cls_msg)
        prob_cls = chainer.cuda.to_cpu(prob_cls.data[0, :].transpose(1, 2, 0))
        cls_prob_msg = bridge.cv2_to_imgmsg(prob_cls)
        cls_prob_msg.header = imgmsg.header
        self.pub_cls_prob.publish(cls_prob_msg)

        prob_suc = chainer.functions.softmax(self.model.score_suc)
        lbl_suc = chainer.functions.argmax(prob_suc, axis=1)
        if self.thresh_suc_prob:
            lbl_suc.data[chainer.functions.max(prob_suc, axis=1).data <
                         self.thresh_suc_prob] = 0
        lbl_suc = chainer.cuda.to_cpu(lbl_suc.data[0])
        suc_msg = bridge.cv2_to_imgmsg(lbl_suc)
        suc_msg.header = imgmsg.header
        self.pub_suc.publish(suc_msg)
        prob_suc = chainer.cuda.to_cpu(prob_suc.data[0, 1])
        suc_prob_msg = bridge.cv2_to_imgmsg(prob_suc)
        suc_prob_msg.header = imgmsg.header
        self.pub_suc_prob.publish(suc_prob_msg)


if __name__ == '__main__':
    rospy.init_node('fcn_object_segmentation')
    app = FCNObjectSegmentation()
    rospy.spin()
