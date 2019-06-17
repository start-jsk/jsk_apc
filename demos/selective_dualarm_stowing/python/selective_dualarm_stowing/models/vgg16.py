import chainer
import chainer.functions as F
import chainer.links as L


class VGG16(chainer.Chain):

    def __init__(self, n_class=1000, threshold=0.5, pt_func=None):
        self.threshold = threshold
        self.pt_func = pt_func
        self.n_class = n_class
        super(VGG16, self).__init__(
            conv1_1=L.Convolution2D(3, 64, 3, stride=1, pad=1),
            bn1_1=L.BatchNormalization(64),
            conv1_2=L.Convolution2D(64, 64, 3, stride=1, pad=1),
            bn1_2=L.BatchNormalization(64),

            conv2_1=L.Convolution2D(64, 128, 3, stride=1, pad=1),
            bn2_1=L.BatchNormalization(128),
            conv2_2=L.Convolution2D(128, 128, 3, stride=1, pad=1),
            bn2_2=L.BatchNormalization(128),

            conv3_1=L.Convolution2D(128, 256, 3, stride=1, pad=1),
            bn3_1=L.BatchNormalization(256),
            conv3_2=L.Convolution2D(256, 256, 3, stride=1, pad=1),
            bn3_2=L.BatchNormalization(256),
            conv3_3=L.Convolution2D(256, 256, 3, stride=1, pad=1),
            bn3_3=L.BatchNormalization(256),

            conv4_1=L.Convolution2D(256, 512, 3, stride=1, pad=1),
            bn4_1=L.BatchNormalization(512),
            conv4_2=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            bn4_2=L.BatchNormalization(512),
            conv4_3=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            bn4_3=L.BatchNormalization(512),

            conv5_1=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            bn5_1=L.BatchNormalization(512),
            conv5_2=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            bn5_2=L.BatchNormalization(512),
            conv5_3=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            bn5_3=L.BatchNormalization(512),

            fc6=L.Linear(40960, 4096),
            fc7=L.Linear(4096, 4096),
            fc8=L.Linear(4096, 2*n_class)
            # fc6=L.Convolution2D(512, 4096, (8, 10)),
            # fc7=L.Convolution2D(4096, 4096, 1),
            # fc8=L.Convolution2D(4096, n_class),
        )

    def __call__(self, x, t=None):
        n_batch = len(x)
        assert n_batch == len(t)

        h = F.relu(self.bn1_1(self.conv1_1(x)))
        h = F.relu(self.bn1_2(self.conv1_2(h)))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.bn2_1(self.conv2_1(h)))
        h = F.relu(self.bn2_2(self.conv2_2(h)))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.bn3_1(self.conv3_1(h)))
        h = F.relu(self.bn3_2(self.conv3_2(h)))
        h = F.relu(self.bn3_3(self.conv3_3(h)))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.bn4_1(self.conv4_1(h)))
        h = F.relu(self.bn4_2(self.conv4_2(h)))
        h = F.relu(self.bn4_3(self.conv4_3(h)))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.bn5_1(self.conv5_1(h)))
        h = F.relu(self.bn5_2(self.conv5_2(h)))
        h = F.relu(self.bn5_3(self.conv5_3(h)))
        h = F.max_pooling_2d(h, 2, stride=2)

        if not self.train_conv:
            h.unchain_backward()

        h = F.dropout(F.relu(self.fc6(h)), ratio=0.5)
        h = F.dropout(F.relu(self.fc7(h)), ratio=0.5)
        h = self.fc8(h)
        h = h.reshape((-1, 2, self.n_class))

        h_prob = F.softmax(h, axis=1)[:, 1, :]
        self.h_prob = h_prob

        if t is None:
            assert not chainer.config.train
            return

        half_n = self.n_class / 2
        is_singlearm_mask = t[:, half_n] == -1
        # loss for single arm
        h_single = h[is_singlearm_mask][:, :, :half_n]
        t_single = t[is_singlearm_mask][:, :half_n]
        # Requires: https://github.com/chainer/chainer/pull/3310
        if h_single.data.shape[0] > 0:
            loss_single = F.softmax_cross_entropy(
                h_single, t_single, normalize=False)
        else:
            loss_single = None

        # loss for dual arm
        h_dual = h[~is_singlearm_mask][:, :, half_n:]
        t_dual = t[~is_singlearm_mask][:, half_n:]
        # Requires: https://github.com/chainer/chainer/pull/3310
        if h_dual.data.shape[0] > 0:
            loss_dual = F.softmax_cross_entropy(
                h_dual, t_dual, normalize=False)
        else:
            loss_dual = None

        if loss_single is None:
            self.loss = loss_dual
        elif loss_dual is None:
            self.loss = loss_single
        else:
            self.loss = loss_single + loss_dual

        # calculate acc on CPU
        h_prob_single = h_prob[is_singlearm_mask][:, :half_n]
        h_prob_single = chainer.cuda.to_cpu(h_prob_single.data)
        t_single = chainer.cuda.to_cpu(t_single)
        h_prob_dual = h_prob[~is_singlearm_mask][:, half_n:]
        h_prob_dual = chainer.cuda.to_cpu(h_prob_dual.data)
        t_dual = chainer.cuda.to_cpu(t_dual)

        label_single = (h_prob_single > self.threshold).astype(self.xp.int32)
        label_dual = (h_prob_dual > self.threshold).astype(self.xp.int32)
        acc_single = (t_single == label_single).all(axis=1)
        acc_single = acc_single.astype(self.xp.int32).flatten()
        acc_dual = (t_dual == label_dual).all(axis=1)
        acc_dual = acc_dual.astype(self.xp.int32).flatten()

        self.acc = self.xp.sum(acc_single) + self.xp.sum(acc_dual)
        self.acc = self.acc / float(len(acc_single) + len(acc_dual))

        chainer.reporter.report({
            'loss': self.loss,
            'acc': self.acc,
            }, self)

        if chainer.config.train:
            return self.loss
