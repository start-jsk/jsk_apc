import chainer
import chainer.functions as F


class FCNSigmoidTrainChain(chainer.Chain):

    def __init__(self, predictor):
        super(FCNSigmoidTrainChain, self).__init__()
        with self.init_scope():
            self.predictor = predictor

    def __call__(self, imgs, depths, lbls):
        scores = self.predictor(imgs, depths)
        loss = F.sigmoid_cross_entropy(scores, lbls)
        chainer.report({'loss': loss}, self)
        return loss
