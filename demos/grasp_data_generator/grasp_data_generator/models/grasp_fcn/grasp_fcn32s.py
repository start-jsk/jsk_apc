# modified by Shingo Kitagawa

# Original work by Kentaro Wada
# https://github.com/wkentaro/fcn


import chainer
import chainer.functions as F
import chainer.links as L
import fcn


class GraspFCN32s(chainer.Chain):

    def __init__(self, n_class=21):
        self.n_class = n_class
        kwargs = {
            'initialW': chainer.initializers.Zero(),
            'initial_bias': chainer.initializers.Zero(),
        }
        super(GraspFCN32s, self).__init__()
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

            self.score_fr = L.Convolution2D(4096, n_class, 1, 1, 0, **kwargs)
            self.grasp_score_fr = L.Convolution2D(
                4096, 2, 1, 1, 0, **kwargs)

            self.upscore = L.Deconvolution2D(
                n_class, n_class, 64, 32, 0, nobias=True,
                initialW=fcn.initializers.UpsamplingDeconvWeight())

            self.grasp_upscore = L.Deconvolution2D(
                2, 2, 64, 32, 0, nobias=True,
                initialW=fcn.initializers.UpsamplingDeconvWeight())

    def __call__(self, x, label=None, grasp=None):
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

        # grasp_fr
        h = self.grasp_score_fr(fc7)
        grasp_score_fr = h

        # upscore
        h = self.upscore(score_fr)
        h = h[:, :, 19:19 + x.data.shape[2], 19:19 + x.data.shape[3]]
        score = h  # 1/1
        self.score = score

        # grasp upscore
        h = self.grasp_upscore(grasp_score_fr)
        h = h[:, :, 19:19 + x.data.shape[2], 19:19 + x.data.shape[3]]
        grasp_score = h
        self.grasp_score = grasp_score

        if label is None:
            assert not chainer.config.train
            return

        seg_loss = F.softmax_cross_entropy(
            self.score, label, normalize=False)

        with chainer.cuda.get_device_from_array(seg_loss.data):
            if self.xp.isnan(seg_loss.data):
                raise ValueError('Loss is nan.')
        self.seg_loss = seg_loss

        grasp_loss = F.softmax_cross_entropy(
            self.grasp_score, grasp, normalize=False)

        with chainer.cuda.get_device_from_array(grasp_loss.data):
            if self.xp.isnan(grasp_loss.data):
                raise ValueError('Loss is nan.')
        self.grasp_loss = grasp_loss

        loss = seg_loss + self.alpha * grasp_loss
        return loss

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
