import numpy as np

import chainer
from chainer.initializers import constant
from chainer.initializers import normal
import chainer.functions as F
import chainer.links as L
from jsk_arc2017_common.in_hand_recognition.resnet.building_block import BuildingBlock

from chainercv.utils import download_model


class ResNet50(chainer.Chain):

    _models = {
        'imagenet': {
            'n_class': 1000,
            'url': 'https://github.com/yuyu2172/chainer-tools/releases/'
            'download/v0.0.1/resnet50_06_19.npz'
        }
    }

    def __init__(self, pretrained_model=None):
        super(ResNet50, self).__init__()
        kwargs = {}
        with self.init_scope():
            self.conv1 = L.Convolution2D(3, 64, 7, 2, 3, **kwargs)
            self.bn1 = L.BatchNormalization(64)
            self.res2 = BuildingBlock(3, 64, 64, 256, 1, **kwargs)
            self.res3 = BuildingBlock(4, 256, 128, 512, 2, **kwargs)
            self.res4 = BuildingBlock(6, 512, 256, 1024, 2, **kwargs)
            self.res5 = BuildingBlock(3, 1024, 512, 2048, 2, **kwargs)

        if pretrained_model in self._models:
            path = download_model(self._models[pretrained_model]['url'])
            chainer.serializers.load_npz(path, self)
        elif pretrained_model:
            chainer.serializers.load_npz(pretrained_model, self)

    def __call__(self, x):
        h = self.conv1(x)
        h = self.bn1(h)
        h = F.relu(h)
        h = F.max_pooling_2d(h, ksize=3, stride=2)
        h = self.res2(h)
        h = self.res3(h)
        h = self.res4(h)
        h = self.res5(h)
        h = _global_average_pooling_2d(h)
        return h
        

def _global_average_pooling_2d(x):
    n, channel, rows, cols = x.data.shape
    h = F.average_pooling_2d(x, (rows, cols), stride=1)
    h = h.reshape(n, channel)
    return h
