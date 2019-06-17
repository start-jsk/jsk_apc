import chainer
from chainer import cuda
from chainer.dataset import concat_examples
import chainer.functions as F
import chainer.links as L
from chainer_mask_rcnn import functions
from chainer_mask_rcnn.models.region_proposal_network \
    import RegionProposalNetwork
import cv2
import numpy as np

from ..resnet import BuildingBlock
from ..resnet import ResNet101Extractor
from ..resnet import ResNet50Extractor
from .mask_rcnn import MaskRCNN
from .mask_rcnn_resnet import _convert_bn_to_affine
from .mask_rcnn_resnet import _copy_persistent_chain


class MaskRCNNPanoptic(MaskRCNN):

    def __init__(
            self, extractor, rpn, head, head_pix,
            mean,
            min_size=600,
            max_size=1000,
            loc_normalize_mean=(0., 0., 0., 0.),
            loc_normalize_std=(0.1, 0.1, 0.2, 0.2),
            detections_per_im=100):
        super(MaskRCNN, self).__init__()
        with self.init_scope():
            self.extractor = extractor
            self.rpn = rpn
            self.head = head
            self.head_pix = head_pix

        self.mean = mean
        self.min_size = min_size
        self.max_size = max_size
        self.loc_normalize_mean = loc_normalize_mean
        self.loc_normalize_std = loc_normalize_std

        self.nms_thresh = 0.5
        self.score_thresh = 0.05

        self._detections_per_im = detections_per_im

    def __call__(self, x, scales):
        img_size = x.shape[2:]

        h = self.extractor(x)
        rpn_locs, rpn_scores, rois, roi_indices, anchor =\
            self.rpn(h, img_size, scales)
        roi_cls_locs, roi_scores, roi_masks = self.head(
            h, rois, roi_indices)
        score_vis, score_occ = self.head_pix(h, img_size)
        return (
            roi_cls_locs,
            roi_scores,
            rois,
            roi_indices,
            roi_masks,
            score_vis,
            score_occ,
        )

    def prepare(self, imgs):
        prepared_imgs = []
        sizes = []
        scales = []
        for img in imgs:
            _, H, W = img.shape

            scale = 1.

            if self.min_size:
                scale = self.min_size / min(H, W)

            if self.max_size and scale * max(H, W) > self.max_size:
                scale = self.max_size / max(H, W)

            img = img.transpose(1, 2, 0)  # CHW -> HWC
            img = cv2.resize(img, None, fx=scale, fy=scale)
            img = pad_multiple_of(img, mode='constant', constant_values=0)
            img = img.transpose(2, 0, 1)  # HWC -> CHW

            img = (img - self.mean).astype(np.float32, copy=False)

            prepared_imgs.append(img)
            sizes.append((H, W))
            scales.append(scale)
        return prepared_imgs, sizes, scales

    def _to_pixel_labels(self, score_vis, score_occ, scales, sizes):
        score_vis = cuda.to_cpu(score_vis.array)
        score_occ = cuda.to_cpu(score_occ.array)
        lbls_vis = []
        lbls_occ = []
        for score_vis_i, score_occ_i, scale, size in \
                zip(score_vis, score_occ, scales, sizes):
            score_vis_i = score_vis_i.transpose(1, 2, 0)
            score_vis_i = cv2.resize(
                score_vis_i, None, None, fx=1. / scale, fy=1. / scale
            )
            score_vis_i = score_vis_i[:size[0], :size[1]]
            lbl_vis = np.argmax(score_vis_i, axis=2)
            lbls_vis.append(lbl_vis)

            score_occ_i = score_occ_i.transpose(1, 2, 0)
            score_occ_i = cv2.resize(
                score_occ_i, None, None, fx=1. / scale, fy=1. / scale
            )
            score_occ_i = score_occ_i[:size[0], :size[1]]
            score_occ_i = score_occ_i.transpose(2, 0, 1)
            lbl_occ = score_occ_i > 0
            lbls_occ.append(lbl_occ)
        return lbls_vis, lbls_occ

    def predict(self, imgs):
        imgs, sizes, scales = self.prepare(imgs)

        batch = list(zip(imgs, scales))
        x, scales = concat_examples(batch, padding=0)
        x = self.xp.asarray(x)

        with chainer.using_config('train', False), chainer.no_backprop_mode():
            h = self.extractor(x)
            rpn_locs, rpn_scores, rois, roi_indices, anchor = self.rpn(
                h, x.shape[2:], scales,
            )
            roi_cls_locs, roi_scores, _ = self.head(
                h, rois, roi_indices, pred_mask=False,
            )
            score_vis, score_occ = self.head_pix(h, x.shape[2:])

        bboxes, labels, scores, assigns = self._to_bboxes(
            roi_cls_locs, roi_scores, rois, roi_indices, sizes, scales,
        )

        roi_indices = cuda.to_cpu(roi_indices)
        roi_indices = np.concatenate([
            roi_indices[roi_indices == i][assign]
            for i, assign in enumerate(assigns)
        ], axis=0)

        roi_masks = self._to_roi_masks(h, bboxes, roi_indices, scales)
        masks = self._to_masks(bboxes, labels, scores, roi_masks, sizes)

        lbls_vis, lbls_occ = self._to_pixel_labels(
            score_vis, score_occ, scales, sizes
        )

        return bboxes, masks, labels, scores, lbls_vis, lbls_occ


def pad_multiple_of(image, num=16, **kwargs):
    H, W = image.shape[:2]

    H_dst = int(np.ceil(1. * H / num) * num)
    W_dst = int(np.ceil(1. * W / num) * num)

    pad_width = [(0, H_dst - H), (0, W_dst - W)]
    if image.ndim == 3:
        pad_width.append((0, 0))

    image = np.pad(image, pad_width, **kwargs)
    return image


class MaskRCNNPanopticResNet(MaskRCNNPanoptic):

    feat_stride = 16

    def __init__(self,
                 n_layers,
                 n_fg_class,
                 pretrained_model=None,
                 min_size=600,
                 max_size=1000,
                 ratios=(0.5, 1, 2),
                 anchor_scales=(4, 8, 16, 32),
                 mean=(123.152, 115.903, 103.063),
                 res_initialW=None,
                 rpn_initialW=None,
                 loc_initialW=None,
                 score_initialW=None,
                 mask_initialW=None,
                 proposal_creator_params=dict(
                     min_size=0,
                     n_test_pre_nms=6000,
                     n_test_post_nms=1000,
                 ),
                 pooling_func=functions.roi_align_2d,
                 rpn_dim=1024,
                 roi_size=7,
                 ):
        if loc_initialW is None:
            loc_initialW = chainer.initializers.Normal(0.001)
        if score_initialW is None:
            score_initialW = chainer.initializers.Normal(0.01)
        if mask_initialW is None:
            mask_initialW = chainer.initializers.Normal(0.01)
        if rpn_initialW is None:
            rpn_initialW = chainer.initializers.Normal(0.01)
        if res_initialW is None and pretrained_model:
            res_initialW = chainer.initializers.constant.Zero()

        if n_layers == 50:
            extractor = ResNet50Extractor(
                pretrained_model=None if pretrained_model else 'auto',
                remove_layers=['res5', 'fc6'],
            )
        elif n_layers == 101:
            extractor = ResNet101Extractor(
                pretrained_model=None if pretrained_model else 'auto',
                remove_layers=['res5', 'fc6'],
            )
        else:
            raise ValueError

        rpn = RegionProposalNetwork(
            1024, rpn_dim,
            ratios=ratios,
            anchor_scales=anchor_scales,
            feat_stride=self.feat_stride,
            initialW=rpn_initialW,
            proposal_creator_params=proposal_creator_params,
        )
        head = ResNetRoIHead(
            n_layers=n_layers,
            n_class=n_fg_class + 1,
            roi_size=roi_size,
            spatial_scale=1. / self.feat_stride,
            pretrained_model=None if pretrained_model else 'auto',
            res_initialW=res_initialW,
            loc_initialW=loc_initialW,
            score_initialW=score_initialW,
            mask_initialW=mask_initialW,
            pooling_func=pooling_func,
        )
        head_pix = ResNetPixelHead(
            n_layers=n_layers,
            n_class=n_fg_class + 1,
        )

        if len(mean) != 3:
            raise ValueError('The mean must be tuple of RGB values.')
        mean = np.asarray(mean, dtype=np.float32)[:, None, None]

        super(MaskRCNNPanopticResNet, self).__init__(
            extractor=extractor,
            rpn=rpn,
            head=head,
            head_pix=head_pix,
            mean=mean,
        )

        if pretrained_model:
            chainer.serializers.load_npz(pretrained_model, self)


class ResNetPixelHead(chainer.Chain):

    def __init__(self, n_layers, n_class):
        self.n_class = n_class
        kwargs = {
            'initialW': chainer.initializers.Normal(0.01),
        }
        super(ResNetPixelHead, self).__init__()
        with self.init_scope():
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
        self._copy_imagenet_pretrained_resnet(n_layers=n_layers)

    def _copy_imagenet_pretrained_resnet(self, n_layers):
        if n_layers == 50:
            pretrained_model = ResNet50Extractor(pretrained_model='auto')
        elif n_layers == 101:
            pretrained_model = ResNet101Extractor(pretrained_model='auto')
        else:
            raise ValueError
        self.res5.copyparams(pretrained_model.res5)
        _copy_persistent_chain(self.res5, pretrained_model.res5)

    def __call__(self, x, img_size):
        assert img_size[0] % 16 == 0
        assert img_size[1] % 16 == 0

        # # conv1 -> bn1 -> res2 -> res3 -> res4
        # h = self.extractor(x)  # 1/16
        h = x

        # res5
        h = self.res5(h)  # 1/16

        assert h.shape[2] == (img_size[0] / 16)
        assert h.shape[3] == (img_size[1] / 16)

        h = self.conv6(h)  # 1/16
        conv6 = h

        # score
        h = self.score_fr(conv6)  # 1/16
        h = F.resize_images(h, img_size)  # 1/1
        score = h

        # score_oc
        h = self.score_oc(conv6)  # 1/16
        h = F.resize_images(h, img_size)  # 1/1
        score_oc = h

        return score, score_oc


class ResNetRoIHead(chainer.Chain):

    mask_size = 14  # Size of the predicted mask.

    def __init__(self, n_layers, n_class, roi_size, spatial_scale,
                 pretrained_model='auto',
                 res_initialW=None, loc_initialW=None, score_initialW=None,
                 mask_initialW=None, pooling_func=functions.roi_align_2d,
                 n_mask_class=3):
        # n_class includes the background
        super(ResNetRoIHead, self).__init__()
        with self.init_scope():
            self.res5 = BuildingBlock(
                3, 1024, 512, 2048, stride=roi_size // 7,
                initialW=res_initialW)
            self.cls_loc = L.Linear(2048, n_class * 4, initialW=loc_initialW)
            self.score = L.Linear(2048, n_class, initialW=score_initialW)

            # 7 x 7 x 2048 -> 14 x 14 x 256
            self.deconv6 = L.Deconvolution2D(
                2048, 256, 2, stride=2, initialW=mask_initialW)
            # 14 x 14 x 256 -> 14 x 14 x 20
            n_fg_class = n_class - 1
            self.mask = L.Convolution2D(
                256, n_fg_class * n_mask_class, 1, initialW=mask_initialW)

        self.n_class = n_class
        self.roi_size = roi_size
        self.spatial_scale = spatial_scale
        self.pooling_func = pooling_func

        _convert_bn_to_affine(self)

        if pretrained_model == 'auto':
            self._copy_imagenet_pretrained_resnet(n_layers)
        else:
            assert pretrained_model is None, \
                'Unsupported pretrained_model: {}'.format(pretrained_model)

    def _copy_imagenet_pretrained_resnet(self, n_layers):
        if n_layers == 50:
            pretrained_model = ResNet50Extractor(pretrained_model='auto')
        elif n_layers == 101:
            pretrained_model = ResNet101Extractor(pretrained_model='auto')
        else:
            raise ValueError
        self.res5.copyparams(pretrained_model.res5)
        _copy_persistent_chain(self.res5, pretrained_model.res5)

    def __call__(self, x, rois, roi_indices, pred_bbox=True, pred_mask=True):
        roi_indices = roi_indices.astype(np.float32)
        indices_and_rois = self.xp.concatenate(
            (roi_indices[:, None], rois), axis=1)
        pool = self.pooling_func(
            x,
            indices_and_rois,
            outh=self.roi_size,
            outw=self.roi_size,
            spatial_scale=self.spatial_scale,
            axes='yx',
        )

        res5 = self.res5(pool)

        roi_cls_locs = None
        roi_scores = None
        roi_masks = None

        if pred_bbox:
            pool5 = F.average_pooling_2d(res5, 7, stride=7)
            roi_cls_locs = self.cls_loc(pool5)
            roi_scores = self.score(pool5)

        if pred_mask:
            deconv6 = F.relu(self.deconv6(res5))
            roi_masks = self.mask(deconv6)

        return roi_cls_locs, roi_scores, roi_masks
