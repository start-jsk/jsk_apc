import os.path as osp

import chainer
import chainer.functions as F
import chainer.links as L
import fcn
import numpy as np

import grasp_fusion_lib


class FCN8sVGG16(chainer.Chain):

    """fcn.models.FCN8sAtOnce."""

    # LSVRC2012 used by VGG16
    MEAN_BGR = np.array([104.00698793, 116.66876762, 122.67891434])

    def __init__(
        self,
        out_channels,
        pretrained_model='auto',
        modal='rgb',
        depth_max_value=0.3,
    ):
        if modal in ['rgb', 'depth']:
            in_channels = 3
        else:
            assert modal == 'rgb+depth'
            in_channels = 6
        self.modal = modal

        assert 0 <= depth_max_value and isinstance(depth_max_value, float)
        self.depth_max_value = depth_max_value

        kwargs = {
            'initialW': chainer.initializers.Zero(),
            'initial_bias': chainer.initializers.Zero(),
        }
        super(FCN8sVGG16, self).__init__()
        with self.init_scope():
            self.conv1_1 = L.Convolution2D(
                in_channels, 64, 3, 1, 100, **kwargs
            )
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

            self.score_fr = L.Convolution2D(
                4096, out_channels, 1, 1, 0, **kwargs
            )

            self.upscore2 = L.Deconvolution2D(
                out_channels, out_channels, 4, 2, 0, nobias=True,
                initialW=fcn.initializers.UpsamplingDeconvWeight())
            self.upscore8 = L.Deconvolution2D(
                out_channels, out_channels, 16, 8, 0, nobias=True,
                initialW=fcn.initializers.UpsamplingDeconvWeight())

            self.score_pool3 = L.Convolution2D(
                256, out_channels, 1, 1, 0, **kwargs
            )
            self.score_pool4 = L.Convolution2D(
                512, out_channels, 1, 1, 0, **kwargs
            )
            self.upscore_pool4 = L.Deconvolution2D(
                out_channels, out_channels, 4, 2, 0, nobias=True,
                initialW=fcn.initializers.UpsamplingDeconvWeight())

        if pretrained_model == 'auto':
            self.init_from_vgg16()
        else:
            chainer.serializers.load_npz(pretrained_model, self)

    def init_from_vgg16(self):
        vgg16 = fcn.models.VGG16()
        if not osp.exists(vgg16.pretrained_model):
            vgg16.download()
        chainer.serializers.load_npz(vgg16.pretrained_model, vgg16)

        for l in self.children():
            if l.name == 'conv1_1':
                l1 = getattr(vgg16, l.name)
                l2 = getattr(self, l.name)
                if l1.W.shape == l2.W.shape:
                    assert l1.b.shape == l2.b.shape
                    l2.W.data[...] = l1.W.data[...]
                    l2.b.data[...] = l1.b.data[...]
                else:
                    assert l1.W.shape[1] == 3
                    assert l2.W.shape[1] == 6
                    l2.W.data[:, :3] = l1.W.data[...]
                    l2.W.data[:, 3:] = l1.W.data[...]
                    l2.b.data[...] = l1.b.data[...]
            elif l.name.startswith('conv'):
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

    def __call__(self, imgs, depths):
        if self.modal == 'rgb':
            x = imgs
        elif self.modal == 'depth':
            x = depths
        else:
            x = self.xp.concatenate((imgs, depths), axis=1)
        del imgs, depths

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
        scores = h  # 1/1

        return scores

    def prepare(self, imgs, depths):
        prepared_imgs = []
        prepared_depths = []
        for img, depth in zip(imgs, depths):
            assert img.shape[0] == 3
            img = img.copy()
            img = img.transpose(1, 2, 0)  # CHW -> HWC
            img = img[:, :, ::-1]  # RGB -> BGR
            img = img.astype(np.float32)
            img -= self.MEAN_BGR
            img = img.transpose(2, 0, 1)  # HWC -> CHW
            prepared_imgs.append(img)

            assert depth.shape == img.shape[1:]
            depth = grasp_fusion_lib.image.colorize_depth(
                depth,
                min_value=0,
                max_value=self.depth_max_value,
                dtype=np.float32,
            )
            assert 0 <= depth.min() and depth.max() <= 1
            depth = (depth - 0.5) * 255
            depth = depth.transpose(2, 0, 1)  # HWC -> CHW
            prepared_depths.append(depth)
        return prepared_imgs, prepared_depths


class FCN8sVGG16Sigmoid(FCN8sVGG16):

    def predict(self, imgs, depths):
        proba = self.predict_proba(imgs, depths)
        return self.proba_to_lbls(proba)

    @staticmethod
    def proba_to_lbls(proba, threshold=0.5):
        lbls = []
        for prob in proba:
            lbls.append((prob > threshold).astype(np.int32))
        return lbls

    def predict_proba(self, imgs, depths):
        imgs, depths = self.prepare(imgs, depths)

        proba = []
        for img, depth in zip(imgs, depths):
            with chainer.no_backprop_mode(), \
                    chainer.using_config('train', False):
                x1 = self.xp.asarray(img[None])
                x2 = self.xp.asarray(depth[None])
                prob = F.sigmoid(self(x1, x2))[0]
            prob = chainer.cuda.to_cpu(prob.array)
            proba.append(prob)
        return proba
