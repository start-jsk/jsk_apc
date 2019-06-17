import chainer
import chainer.functions as F
import chainer.links as L


class DualarmVGG16(chainer.Chain):

    def __init__(self, n_failure, n_class, threshold=0.5, pt_func=None):
        self.threshold = threshold
        self.pt_func = pt_func
        self.n_failure = n_failure
        super(DualarmVGG16, self).__init__()
        with self.init_scope():
            self.conv1_1 = L.Convolution2D(3, 64, 3, stride=1, pad=1)
            self.bn1_1 = L.BatchNormalization(64)
            self.conv1_2 = L.Convolution2D(64, 64, 3, stride=1, pad=1)
            self.bn1_2 = L.BatchNormalization(64)

            self.conv2_1 = L.Convolution2D(64, 128, 3, stride=1, pad=1)
            self.bn2_1 = L.BatchNormalization(128)
            self.conv2_2 = L.Convolution2D(128, 128, 3, stride=1, pad=1)
            self.bn2_2 = L.BatchNormalization(128)

            self.conv3_1 = L.Convolution2D(128, 256, 3, stride=1, pad=1)
            self.bn3_1 = L.BatchNormalization(256)
            self.conv3_2 = L.Convolution2D(256, 256, 3, stride=1, pad=1)
            self.bn3_2 = L.BatchNormalization(256)
            self.conv3_3 = L.Convolution2D(256, 256, 3, stride=1, pad=1)
            self.bn3_3 = L.BatchNormalization(256)

            self.conv4_1 = L.Convolution2D(256, 512, 3, stride=1, pad=1)
            self.bn4_1 = L.BatchNormalization(512)
            self.conv4_2 = L.Convolution2D(512, 512, 3, stride=1, pad=1)
            self.bn4_2 = L.BatchNormalization(512)
            self.conv4_3 = L.Convolution2D(512, 512, 3, stride=1, pad=1)
            self.bn4_3 = L.BatchNormalization(512)

            self.conv5_1 = L.Convolution2D(512, 512, 3, stride=1, pad=1)
            self.bn5_1 = L.BatchNormalization(512)
            self.conv5_2 = L.Convolution2D(512, 512, 3, stride=1, pad=1)
            self.bn5_2 = L.BatchNormalization(512)
            self.conv5_3 = L.Convolution2D(512, 512, 3, stride=1, pad=1)
            self.bn5_3 = L.BatchNormalization(512)

            self.fc6_failure = L.Linear(40960, 4096)
            self.fc7_failure = L.Linear(4096, 4096)
            self.fc8_failure = L.Linear(4096, 2*n_failure)

            self.fc6_cls = L.Linear(40960, 4096)
            self.fc7_cls = L.Linear(4096, 4096)
            self.fc8_cls = L.Linear(4096, 2*n_failure)

    def __call__(self, x, t=None, t_cls=None):
        n_batch = len(x)
        assert n_batch == len(t) == len(t_cls)

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
        conv5 = h

        if not self.train_conv:
            h.unchain_backward()

        # failure prediction
        h = F.dropout(F.relu(self.fc6_failure(conv5)), ratio=0.5)
        h = F.dropout(F.relu(self.fc7_failure(h)), ratio=0.5)
        h = self.fc8_failure(h)
        h = h.reshape((-1, 2, self.n_failure))
        fc8_failure = h

        fail_prob = F.softmax(fc8_failure, axis=1)[:, 1, :]
        self.fail_prob = fail_prob

        # classification prediction
        h = F.dropout(F.relu(self.fc6_cls(conv5)), ratio=0.5)
        h = F.dropout(F.relu(self.fc7_cls(h)), ratio=0.5)
        h = self.fc8_cls(h)
        cls_score = h
        self.cls_score = cls_score

        if t is None:
            assert not chainer.config.train
            return

        half_n = self.n_failure / 2
        is_singlearm_mask = t[:, half_n] == -1

        # loss for single arm
        h_single = fc8_failure[is_singlearm_mask][:, :, :half_n]
        t_single = t[is_singlearm_mask][:, :half_n]
        # Requires: https://github.com/chainer/chainer/pull/3310
        if h_single.data.shape[0] > 0:
            loss_single = F.softmax_cross_entropy(
                h_single, t_single, normalize=False)
        else:
            loss_single = None

        # loss for dual arm
        h_dual = fc8_failure[~is_singlearm_mask][:, :, half_n:]
        t_dual = t[~is_singlearm_mask][:, half_n:]
        # Requires: https://github.com/chainer/chainer/pull/3310
        if h_dual.data.shape[0] > 0:
            loss_dual = F.softmax_cross_entropy(
                h_dual, t_dual, normalize=False)
        else:
            loss_dual = None

        # classification loss
        cls_loss = F.softmax_cross_entropy(cls_score, t_cls)
        self.cls_loss = cls_loss

        if loss_single is None:
            self.fail_loss = loss_dual
        elif loss_dual is None:
            self.fail_loss = loss_single
        else:
            self.fail_loss = loss_single + loss_dual

        self.loss = self.fail_loss + self.cls_loss

        # calculate acc on CPU
        fail_prob_single = fail_prob[is_singlearm_mask][:, :half_n]
        fail_prob_single = chainer.cuda.to_cpu(fail_prob_single.data)
        t_single = chainer.cuda.to_cpu(t_single)
        fail_prob_dual = fail_prob[~is_singlearm_mask][:, half_n:]
        fail_prob_dual = chainer.cuda.to_cpu(fail_prob_dual.data)
        t_dual = chainer.cuda.to_cpu(t_dual)

        fail_label_single = fail_prob_single > self.threshold
        fail_label_single = fail_label_single.astype(self.xp.int32)
        fail_label_dual = fail_prob_dual > self.threshold
        fail_label_dual = fail_label_dual.astype(self.xp.int32)
        fail_acc_single = (t_single == fail_label_single).all(axis=1)
        fail_acc_single = fail_acc_single.astype(self.xp.int32).flatten()
        fail_acc_dual = (t_dual == fail_label_dual).all(axis=1)
        fail_acc_dual = fail_acc_dual.astype(self.xp.int32).flatten()

        self.fail_acc = self.xp.sum(fail_acc_single)
        self.fail_acc += self.xp.sum(fail_acc_dual)
        self.fail_acc /= float(len(fail_acc_single) + len(fail_acc_dual))

        cls_pred = F.argmax(cls_score, axis=1)
        cls_pred = chainer.cuda.to_cpu(cls_pred.data)
        t_cls = chainer.cuda.to_cpu(t_cls)
        self.cls_acc = self.xp.sum(t_cls == cls_pred)
        self.cls_acc /= float(len(t_cls))

        chainer.reporter.report({
            'loss': self.loss,
            'cls/loss': self.cls_loss,
            'cls/acc': self.cls_acc,
            'fail/loss': self.fail_loss,
            'fail/acc': self.fail_acc,
            }, self)

        if chainer.config.train:
            return self.loss
