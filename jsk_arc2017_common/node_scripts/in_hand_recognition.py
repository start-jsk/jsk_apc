#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from jsk_recognition_msgs.msg import ClassificationResult

from cv_bridge import CvBridge
import matplotlib.pyplot as plt
import numpy as np

import chainer
from chainer.datasets import TupleDataset

from jsk_arc2017_common.in_hand_recognition.resnet_pca import ResNet50PCA
from jsk_arc2017_common.in_hand_recognition.retrieval_predictor import \
    RetrievalPredictor
from jsk_topic_tools import ConnectionBasedTransport


class InHandRecognitionNode(ConnectionBasedTransport):

    """In hand recognition using an image retrieval algorithm.

    Two rosparams are paths to database images and labels.

    * ~imgs_path: Path to a numpy.ndarray file. It contains batch of images,
        and its shape is (B, 3, H, W).
    * ~labels_path: Path to a numpy.ndarray file. It contains batch of labels,
        and its shape is (B,).

    The ClassificationResult message contains class labels that corresponds to
    the class labels in the database.

    """

    def __init__(self):
        super(InHandRecognitionNode, self).__init__()
        gpu = rospy.get_param('~gpu', -1)
        imgs_path = rospy.get_param('~imgs_path')
        labels_path = rospy.get_param('~labels_path')

        base_model = ResNet50PCA(pretrained_model='imagenet_pca')
        self.model = RetrievalPredictor(base_model, k=3)
        if gpu >= 0:
            self.model.to_gpu(gpu)
            chainer.cuda.get_device(gpu).use()
        imgs = np.load(imgs_path)[:1]
        labels = np.load(labels_path)[:1]
        it = chainer.iterators.SerialIterator(
            TupleDataset(imgs, labels), batch_size=1,
            shuffle=False, repeat=False)
        self.model.load_db(it)

        self.pub =self.advertise(
            '~classification', ClassificationResult, queue_size=10)
        self.bridge = CvBridge()

    def subscribe(self):
        self.sub = rospy.Subscriber('~input', Image, self.callback)

    def unsubscribe(self):
        self.sub.unregister()

    def callback(self, img_msg):
        # Subscribe results
        img = self.bridge.imgmsg_to_cv2(img_msg)
        # TODO: Make sure that this image is RGB
        img = img.transpose(2, 0, 1)

        # Predictions
        pred_indices = self.model.predict([img])[0]  # (K,)
        pred_labels = self.model.db_labels[pred_indices]  # (K,)
        probs = np.linspace(1, 0, len(pred_labels) + 1)[:-1]

        # Publish results
        msg = ClassificationResult(
            labels=pred_labels,
            label_proba=probs,
            classifier='in_hand_recognition'
        )
        msg.header.stamp = rospy.Time.now()
        self.pub.publish(msg)


if __name__ == '__main__':
    rospy.init_node('in_hand_recognition')
    node = InHandRecognitionNode()
    rospy.spin()
