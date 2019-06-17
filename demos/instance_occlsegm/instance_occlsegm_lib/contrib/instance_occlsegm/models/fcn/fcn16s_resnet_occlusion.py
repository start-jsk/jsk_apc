import chainer
import chainer.functions as F
import chainer.links as L

from chainer_mask_rcnn.models.mask_rcnn_resnet import _copy_persistent_chain
from chainer_mask_rcnn.models.resnet_extractor import _convert_bn_to_affine

from ..resnet import BuildingBlock
from ..resnet import ResNet101Extractor
from ..resnet import ResNet50Extractor


class FCN16sResNetOcclusion(chainer.Chain):

    def __init__(self, n_class=21):
        self.n_class = n_class
        kwargs = {
            'initialW': chainer.initializers.Normal(0.01),
        }
        super(FCN16sResNetOcclusion, self).__init__()
        with self.init_scope():
            self.extractor = ResNet50Extractor(remove_layers=['res5', 'fc6'])
            self.res5 = BuildingBlock(
                3, 1024, 512, 2048, stride=1, dilate=2,
                initialW=chainer.initializers.Zero(),
            )

            # head
            self.conv6 = L.Convolution2D(2048, 1024, 1, 1, 0, **kwargs)
            self.score_fr = L.Convolution2D(1024, n_class, 1, 1, 0, **kwargs)

            n_fg_class = n_class - 1
            self.score_oc = L.Convolution2D(
                1024, n_fg_class, 1, 1, 0, **kwargs
            )

        _convert_bn_to_affine(self.res5)
        self._copy_imagenet_pretrained_resnet(n_layers=50)

    def _copy_imagenet_pretrained_resnet(self, n_layers):
        if n_layers == 50:
            pretrained_model = ResNet50Extractor(pretrained_model='auto')
        elif n_layers == 101:
            pretrained_model = ResNet101Extractor(pretrained_model='auto')
        else:
            raise ValueError
        self.res5.copyparams(pretrained_model.res5)
        _copy_persistent_chain(self.res5, pretrained_model.res5)

    def __call__(self, x):
        assert x.shape[2] % 16 == 0
        assert x.shape[3] % 16 == 0

        # conv1 -> bn1 -> res2 -> res3 -> res4
        h = self.extractor(x)  # 1/16

        # res5
        h = self.res5(h)  # 1/16

        assert h.shape[2] == (x.shape[2] / 16)
        assert h.shape[3] == (x.shape[3] / 16)

        h = self.conv6(h)  # 1/16
        conv6 = h

        # score
        h = self.score_fr(conv6)  # 1/16
        h = F.resize_images(h, x.shape[2:4])  # 1/1
        score = h

        # score_oc
        h = self.score_oc(conv6)  # 1/16
        h = F.resize_images(h, x.shape[2:4])  # 1/1
        score_oc = h

        return score, score_oc

    def predict(self, imgs):
        lbls = []
        masks_oc = []
        for img in imgs:
            with chainer.no_backprop_mode(), \
                    chainer.using_config('train', False):
                x = self.xp.asarray(img[None])
                score, score_oc = self.__call__(x)
                lbl = chainer.functions.argmax(score, axis=1)
                prob_oc = chainer.functions.sigmoid(score_oc)
            lbl = chainer.cuda.to_cpu(lbl.array[0])
            mask_oc = chainer.cuda.to_cpu(prob_oc.array[0] > 0.5)
            lbls.append(lbl)
            masks_oc.append(mask_oc)
        return lbls, masks_oc


class OcclusionSegmentationTrainChain(chainer.Chain):

    def __init__(self, predictor, train_occlusion=True):
        super(OcclusionSegmentationTrainChain, self).__init__()
        with self.init_scope():
            self.predictor = predictor

        self._train_occlusion = train_occlusion

    def __call__(self, x, lbl_vis, lbl_occ):
        score_vis, score_occ = self.predictor(x)
        loss_vis = F.softmax_cross_entropy(score_vis, lbl_vis)

        if self._train_occlusion:
            loss_occ = F.sigmoid_cross_entropy(score_occ, lbl_occ)
        else:
            loss_occ = chainer.Variable(
                self.xp.zeros((), dtype=self.xp.float32)
            )

        loss = loss_vis + loss_occ

        chainer.report({
            'loss': loss,
            'loss_vis': loss_vis,
            'loss_occ': loss_occ,
        }, self)
        return loss
