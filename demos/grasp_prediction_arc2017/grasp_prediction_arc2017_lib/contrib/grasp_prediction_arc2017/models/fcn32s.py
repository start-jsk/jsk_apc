import chainer
import chainer.functions as F
import chainer.links as L

import fcn
import numpy as np


class FCN32s(chainer.Chain):

    def __init__(self, n_class=21, class_agnostic=False):
        self.n_class = n_class
        self.class_agnostic = class_agnostic
        kwargs = {
            'initialW': chainer.initializers.Zero(),
            'initial_bias': chainer.initializers.Zero(),
        }
        super(FCN32s, self).__init__()
        with self.init_scope():
            self.conv1_1 = L.Convolution2D(3, 64, 3, 1, 100, **kwargs)
            self.conv1_2 = L.Convolution2D(64, 64, 3, 1, 1, **kwargs)

            self.conv2_1 = L.Convolution2D(64, 128, 3, 1, 1, **kwargs)
            self.conv2_2 = L.Convolution2D(128, 128, 3, 1, 1, **kwargs)

            self.conv3_1 = L.Convolution2D(128, 256, 3, 1, 1, **kwargs)
            self.conv3_2 = L.Convolution2D(256, 256, 3, 1, 1, **kwargs)
            self.conv3_3 = L.Convolution2D(256, 256, 3, 1, 1, **kwargs)

            self.conv4_1 = L.Convolution2D(256, 512, 3, 1, 1, **kwargs)
            self.conv4_2 = L.Convolution2D(512, 512, 3, 1, 1, **kwargs)
            self.conv4_3 = L.Convolution2D(512, 512, 3, 1, 1, **kwargs)

            self.conv5_1 = L.Convolution2D(512, 512, 3, 1, 1, **kwargs)
            self.conv5_2 = L.Convolution2D(512, 512, 3, 1, 1, **kwargs)
            self.conv5_3 = L.Convolution2D(512, 512, 3, 1, 1, **kwargs)

            self.fc6 = L.Convolution2D(512, 4096, 7, 1, 0, **kwargs)
            self.fc7 = L.Convolution2D(4096, 4096, 1, 1, 0, **kwargs)

            suc_channels = 2 if class_agnostic else n_class * 2

            self.score_fr = L.Convolution2D(4096, n_class, 1, 1, 0, **kwargs)
            self.score_fr_suc = L.Convolution2D(
                4096, suc_channels, 1, 1, 0, **kwargs)

            self.upscore_cls = L.Deconvolution2D(
                n_class, n_class, 64, 32, 0, nobias=True,
                initialW=fcn.initializers.UpsamplingDeconvWeight())
            self.upscore_suc = L.Deconvolution2D(
                suc_channels, suc_channels, 64, 32, 0, nobias=True,
                initialW=fcn.initializers.UpsamplingDeconvWeight())

    def __call__(self, x, t_cls=None, t_suc=None):
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

        # XXX:
        # fc7 -> <branch>.--> score_cls
        #                |
        #                `--> score_suc

        # score_cls
        h = self.score_fr(fc7)  # 1/32
        h = self.upscore_cls(h)
        h = h[:, :, 19:19 + x.data.shape[2], 19:19 + x.data.shape[3]]
        upscore_cls = h  # 1/1
        self.score_cls = upscore_cls

        # score_suc
        h = self.score_fr_suc(fc7)  # 1/32
        h = self.upscore_suc(h)
        h = h[:, :, 19:19 + x.data.shape[2], 19:19 + x.data.shape[3]]
        upscore_suc = h  # 1/1

        if not self.class_agnostic:
            # N, n_class * 2, H, W -> N, n_class, 2, H, W
            N, n_class_x2, H, W = upscore_suc.shape
            n_class = n_class_x2 // 2
            assert n_class == self.n_class
            upscore_suc = F.reshape(upscore_suc, (N, n_class, 2, H, W))
            # N, n_class, 2, H, W -> n_class, 2, N, H, W
            upscore_suc = F.transpose(upscore_suc, (1, 2, 0, 3, 4))
            # n_class, 2, N, H, W -> n_class, 2, N * H * W
            upscore_suc = F.reshape(upscore_suc, (n_class, 2, -1))

            if not chainer.config.train:  # used at val + test.
                # N, n_class, H, W -> N, H, W
                label_cls = F.argmax(upscore_cls, axis=1)  # [0, n_class - 1]
                # n_class, 2, N * H * W -> 2, N * H * W
                upscore_suc = upscore_suc[
                    label_cls.reshape(-1).array, :, np.arange(N * H * W)]
                # N * H * W, 2 -> N, H, W, 2 -> N, 2, H, W
                upscore_suc = F.reshape(upscore_suc, (N, H, W, 2))
                upscore_suc = F.transpose(upscore_suc, (0, 3, 1, 2))
                del label_cls

        self.score_suc = upscore_suc

        if t_cls is None or t_suc is None:
            assert not chainer.config.train
            return

        if not self.class_agnostic and chainer.config.train:  # used at train.
            # n_class, 2, N * H * W -> 2, N * H * W
            upscore_suc = upscore_suc[
                t_cls.array.reshape(-1), :, np.arange(N * H * W)]
            # N * H * W, 2 -> N, H, W, 2 -> N, 2, H, W
            upscore_suc = F.reshape(upscore_suc, (N, H, W, 2))
            upscore_suc = F.transpose(upscore_suc, (0, 3, 1, 2))
            self.score_suc = upscore_suc

        loss_cls = F.softmax_cross_entropy(
            upscore_cls, t_cls, normalize=False)
        loss_suc = F.softmax_cross_entropy(
            upscore_suc, t_suc, normalize=False)
        if np.isnan(float(loss_cls.data)) or np.isnan(float(loss_suc.data)):
            raise ValueError('Loss is nan!')
        return loss_cls, loss_suc

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
