# modified by Shingo Kitagawa

# Original work by Kentaro Wada
# https://github.com/wkentaro/fcn


import chainer
import chainer.functions as F
import chainer.links as L
import fcn


class DualarmGraspFCN32s(chainer.Chain):

    def __init__(self, n_class=21):
        self.n_class = n_class
        kwargs = {
            'initialW': chainer.initializers.Zero(),
            'initial_bias': chainer.initializers.Zero(),
        }
        super(DualarmGraspFCN32s, self).__init__()
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
            self.single_grasp_score_fr = L.Convolution2D(
                4096, 2, 1, 1, 0, **kwargs)
            self.dual_grasp_score_fr = L.Convolution2D(
                4096, 2, 1, 1, 0, **kwargs)

            self.upscore = L.Deconvolution2D(
                n_class, n_class, 64, 32, 0, nobias=True,
                initialW=fcn.initializers.UpsamplingDeconvWeight())

            self.single_grasp_upscore = L.Deconvolution2D(
                2, 2, 64, 32, 0, nobias=True,
                initialW=fcn.initializers.UpsamplingDeconvWeight())
            self.dual_grasp_upscore = L.Deconvolution2D(
                2, 2, 64, 32, 0, nobias=True,
                initialW=fcn.initializers.UpsamplingDeconvWeight())

    def __call__(self, x, label=None, single_grasp=None, dual_grasp=None):
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
        if chainer.config.train and not self.use_seg_loss:
            fc7.unchain_backward()
        h = self.score_fr(fc7)
        score_fr = h  # 1/32

        # single_grasp_fr
        h = self.single_grasp_score_fr(fc7)
        single_grasp_score_fr = h

        # dual_grasp_fr
        h = self.dual_grasp_score_fr(fc7)
        dual_grasp_score_fr = h

        # upscore
        h = self.upscore(score_fr)
        h = h[:, :, 19:19 + x.data.shape[2], 19:19 + x.data.shape[3]]
        score = h  # 1/1
        self.score = score

        # single_grasp upscore
        h = self.single_grasp_upscore(single_grasp_score_fr)
        h = h[:, :, 19:19 + x.data.shape[2], 19:19 + x.data.shape[3]]
        single_grasp_score = h
        self.single_grasp_score = single_grasp_score

        # dual_grasp upscore
        h = self.dual_grasp_upscore(dual_grasp_score_fr)
        h = h[:, :, 19:19 + x.data.shape[2], 19:19 + x.data.shape[3]]
        dual_grasp_score = h
        self.dual_grasp_score = dual_grasp_score

        if label is None:
            assert not chainer.config.train
            return

        # seg loss
        seg_loss = F.softmax_cross_entropy(
            self.score, label, normalize=False)

        with chainer.cuda.get_device_from_array(seg_loss.data):
            if self.xp.isnan(seg_loss.data):
                raise ValueError('Loss is nan.')
        self.seg_loss = seg_loss

        if self.train_as_fcn:
            return seg_loss

        with chainer.cuda.get_device_from_array(label.data):
            fg_mask = label.data[0] > 0
            fg_sum = fg_mask.astype(self.xp.int32).sum()

        # single grasp loss
        # median frequency balancing
        with chainer.cuda.get_device_from_array(single_grasp.data):
            if self.frq_balancing and fg_mask.any():
                sg = single_grasp.data[0]
                sg = sg[fg_mask]
                sg_frq = self.xp.bincount(sg.flatten(), minlength=2)
                sg_frq = sg_frq.astype(self.xp.float32)
                # sg_avr = sg_frq.sum() / 2.0
                if sg_frq[1] == 0:
                    sg_weight = self.xp.array([1.0, 0.0], self.xp.float32)
                elif sg_frq[0] == 0:
                    sg_weight = self.xp.array([0.0, 1.0], self.xp.float32)
                else:
                    # sg_weight = sg_avr / sg_frq
                    sg_frq[1] = sg_frq[1] * self.alpha_graspable
                    sg_weight = fg_sum.astype(self.xp.float32) / sg_frq
            else:
                sg_weight = None

        single_grasp_loss = F.softmax_cross_entropy(
            self.single_grasp_score, single_grasp,
            class_weight=sg_weight, normalize=False)

        with chainer.cuda.get_device_from_array(single_grasp_loss.data):
            if self.xp.isnan(single_grasp_loss.data):
                raise ValueError('Loss is nan.')
        self.single_grasp_loss = single_grasp_loss

        # dual grasp loss
        # median frequency balancing
        with chainer.cuda.get_device_from_array(dual_grasp.data):
            if self.frq_balancing and fg_mask.any():
                dg = dual_grasp.data[0]
                dg = dg[fg_mask]
                dg_frq = self.xp.bincount(dg.flatten(), minlength=2)
                dg_frq = dg_frq.astype(self.xp.float32)
                # dg_avr = dg_frq.sum() / 2.0
                if dg_frq[1] == 0:
                    dg_weight = self.xp.array([1.0, 0.0], self.xp.float32)
                elif dg_frq[0] == 0:
                    dg_weight = self.xp.array([0.0, 1.0], self.xp.float32)
                else:
                    # dg_weight = dg_avr / dg_frq
                    dg_frq[1] = dg_frq[1] * self.alpha_graspable
                    dg_weight = fg_sum.astype(self.xp.float32) / dg_frq
            else:
                dg_weight = None

        dual_grasp_loss = F.softmax_cross_entropy(
            self.dual_grasp_score, dual_grasp,
            class_weight=dg_weight, normalize=False)

        with chainer.cuda.get_device_from_array(dual_grasp_loss.data):
            if self.xp.isnan(dual_grasp_loss.data):
                raise ValueError('Loss is nan.')
        self.dual_grasp_loss = dual_grasp_loss

        if self.use_seg_loss is True:
            loss = seg_loss
            loss = loss + self.alpha_single * single_grasp_loss
            loss = loss + self.alpha_dual * dual_grasp_loss
        else:
            loss = self.alpha_single * single_grasp_loss
            loss = loss + self.alpha_dual * dual_grasp_loss
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
