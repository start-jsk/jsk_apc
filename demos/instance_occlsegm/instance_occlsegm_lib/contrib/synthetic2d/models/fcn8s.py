import os.path as osp

import chainer
import chainer.functions as F
from fcn import data
from fcn.models import FCN8s
import numpy as np


class FCN8sAtOnce(FCN8s):

    pretrained_model = osp.expanduser(
        '~/data/models/chainer/fcn8s-atonce_from_caffe.npz')

    def __call__(self, x):
        # conv1
        h = F.relu(self.conv1_1(x))
        conv1_1 = h
        h = F.relu(self.conv1_2(conv1_1))
        conv1_2 = h
        h = F.max_pooling_2d(conv1_2, 2, stride=2, pad=0)
        pool1 = h  # 1/2

        # conv2
        h = F.relu(self.conv2_1(pool1))
        conv2_1 = h
        h = F.relu(self.conv2_2(conv2_1))
        conv2_2 = h
        h = F.max_pooling_2d(conv2_2, 2, stride=2, pad=0)
        pool2 = h  # 1/4

        # conv3
        h = F.relu(self.conv3_1(pool2))
        conv3_1 = h
        h = F.relu(self.conv3_2(conv3_1))
        conv3_2 = h
        h = F.relu(self.conv3_3(conv3_2))
        conv3_3 = h
        h = F.max_pooling_2d(conv3_3, 2, stride=2, pad=0)
        pool3 = h  # 1/8

        # conv4
        h = F.relu(self.conv4_1(pool3))
        h = F.relu(self.conv4_2(h))
        h = F.relu(self.conv4_3(h))
        h = F.max_pooling_2d(h, 2, stride=2, pad=0)
        pool4 = h  # 1/16

        # conv5
        h = F.relu(self.conv5_1(pool4))
        h = F.relu(self.conv5_2(h))
        h = F.relu(self.conv5_3(h))
        h = F.max_pooling_2d(h, 2, stride=2, pad=0)
        pool5 = h  # 1/32

        # fc6
        h = F.relu(self.fc6(pool5))
        h = F.dropout(h, ratio=.5)
        fc6 = h  # 1/32

        # fc7
        h = F.relu(self.fc7(fc6))
        h = F.dropout(h, ratio=.5)
        fc7 = h  # 1/32

        # score_fr
        h = self.score_fr(fc7)
        score_fr = h  # 1/32

        # score_pool3
        scale_pool3 = 0.0001 * pool3  # XXX: scale to train at once
        h = self.score_pool3(scale_pool3)
        score_pool3 = h  # 1/8

        # score_pool4
        scale_pool4 = 0.01 * pool4  # XXX: scale to train at once
        h = self.score_pool4(scale_pool4)
        score_pool4 = h  # 1/16

        # upscore2
        h = self.upscore2(score_fr)
        upscore2 = h  # 1/16

        # score_pool4c
        h = score_pool4[:, :,
                        5:5 + upscore2.data.shape[2],
                        5:5 + upscore2.data.shape[3]]
        score_pool4c = h  # 1/16

        # fuse_pool4
        h = upscore2 + score_pool4c
        fuse_pool4 = h  # 1/16

        # upscore_pool4
        h = self.upscore_pool4(fuse_pool4)
        upscore_pool4 = h  # 1/8

        # score_pool4c
        h = score_pool3[:, :,
                        9:9 + upscore_pool4.data.shape[2],
                        9:9 + upscore_pool4.data.shape[3]]
        score_pool3c = h  # 1/8

        # fuse_pool3
        h = upscore_pool4 + score_pool3c
        fuse_pool3 = h  # 1/8

        # upscore8
        h = self.upscore8(fuse_pool3)
        upscore8 = h  # 1/1

        # score
        h = upscore8[:, :, 31:31 + x.shape[2], 31:31 + x.shape[3]]
        score = h  # 1/1

        return score

    def init_from_vgg16(self, vgg16):
        for l in self.children():
            if l.name.startswith('conv'):
                l1 = getattr(vgg16, l.name)
                l2 = getattr(self, l.name)
                assert l1.W.shape == l2.W.shape
                assert l1.b.shape == l2.b.shape
                l2.W.data[...] = l1.W.data[...]
                l2.b.data[...] = l1.b.data[...]
            elif l.name in ['fc6', 'fc7']:
                l1 = getattr(vgg16, l.name)
                l2 = getattr(self, l.name)
                assert l1.W.size == l2.W.size
                assert l1.b.size == l2.b.size
                l2.W.data[...] = l1.W.data.reshape(l2.W.shape)[...]
                l2.b.data[...] = l1.b.data.reshape(l2.b.shape)[...]

    @classmethod
    def download(cls):
        return data.cached_download(
            url='https://drive.google.com/uc?id=0B9P1L--7Wd2vZ1RJdXotZkNhSEk',
            path=cls.pretrained_model,
            md5='5f3ffdc7fae1066606e1ef45cfda548f',
        )

    def predict(self, imgs):
        lbls = list()
        for img in imgs:
            with chainer.using_config('train', False), \
                    chainer.function.no_backprop_mode():
                x = chainer.Variable(self.xp.asarray(img[np.newaxis]))
                score = self.__call__(x)[0].data
            score = chainer.cuda.to_cpu(score)
            lbl = np.argmax(score, axis=0).astype(np.int32)
            lbls.append(lbl)
        return lbls
