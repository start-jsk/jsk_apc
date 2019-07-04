import chainer
import chainer.functions as F
import chainer.links as L


class DualarmAlex(chainer.Chain):
    def __init__(self, n_failure, n_class, threshold=0.5, pt_func=None):
        self.threshold = threshold
        self.pt_func = pt_func
        self.n_failure = n_failure
        super(DualarmAlex, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(3, 96, 11, stride=4, pad=4)
            self.bn1 = L.BatchNormalization(96)
            self.conv2 = L.Convolution2D(96, 256, 5, stride=1, pad=1)
            self.bn2 = L.BatchNormalization(256)
            self.conv3 = L.Convolution2D(256, 384, 3, stride=1, pad=1)
            self.conv4 = L.Convolution2D(384, 384, 3, stride=1, pad=1)
            self.conv5 = L.Convolution2D(384, 256, 3, stride=1, pad=1)
            self.bn5 = L.BatchNormalization(256)
            self.fc6_failure = L.Linear(33280, 4096)
            self.fc7_failure = L.Linear(4096, 4096)
            self.fc8_failure = L.Linear(4096, 2*n_failure)

            self.fc6_cls = L.Linear(33280, 4096)
            self.fc7_cls = L.Linear(4096, 4096)
            self.fc8_cls = L.Linear(4096, n_class)

    def __call__(self, x, t=None, t_cls=None):
        n_batch = len(x)

        h = F.relu(self.bn1(self.conv1(x)))
        h = F.max_pooling_2d(h, 3, stride=2)
        h = F.relu(self.bn2(self.conv2(h)))
        h = F.max_pooling_2d(h, 3, stride=2)
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.relu(self.bn5(self.conv5(h)))
        h = F.max_pooling_2d(h, 3, stride=3)
        conv4 = h

        if not self.train_conv:
            h.unchain_backward()

        # failure prediction
        h = F.dropout(F.relu(self.fc6_failure(conv4)), ratio=0.5)
        h = F.dropout(F.relu(self.fc7_failure(h)), ratio=0.5)
        h = self.fc8_failure(h)
        h = h.reshape((-1, 2, self.n_failure))
        fc8_failure = h

        fail_prob = F.softmax(fc8_failure, axis=1)[:, 1, :]
        self.fail_prob = fail_prob

        # classification prediction
        h = F.dropout(F.relu(self.fc6_cls(conv4)), ratio=0.5)
        h = F.dropout(F.relu(self.fc7_cls(h)), ratio=0.5)
        h = self.fc8_cls(h)
        cls_score = h
        self.cls_score = cls_score

        if t is None:
            assert not chainer.config.train
            return

        # failure loss
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
