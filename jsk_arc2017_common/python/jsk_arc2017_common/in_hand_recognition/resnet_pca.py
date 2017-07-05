import chainer
import chainer.functions as F

from chainercv.utils import download_model

from jsk_arc2017_common.in_hand_recognition.resnet.resnet_50 import ResNet50


class ResNet50PCA(chainer.Chain):

    _models = {
        'imagenet_pca': {
            'url': 'https://github.com/yuyu2172/chainer-tools/releases/'
            'download/v0.0.1/resnet50_pca_06_19.npz'
        }
    }

    def __init__(self, pretrained_model=None):
        base_pretrained_model = None
        if pretrained_model == 'imagenet':
            base_pretrained_model = 'imagenet'
            pretrained_model = None
        super(ResNet50PCA, self).__init__()

        with self.init_scope():
            self.resnet = ResNet50(
                pretrained_model=base_pretrained_model)

        self.add_persistent('pc', None)
        self.add_persistent('pc_mean', None)

        if pretrained_model in self._models:
            path = download_model(self._models[pretrained_model]['url'])
            chainer.serializers.load_npz(path, self)
        elif pretrained_model:
            chainer.serializers.load_npz(pretrained_model, self)

    def __call__(self, x):
        h = self.resnet(x)
        if self.pc_mean is not None:
            h = h - self.pc_mean[None]
        if self.pc is not None:
            h = F.matmul(h, self.pc.T)
        return h


if __name__ == '__main__':
    model = ResNet50PCA(pretrained_model='imagenet')
