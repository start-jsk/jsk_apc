import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L

from chainer.links.model.vision.resnet import BuildingBlock
from chainercv.links.model.faster_rcnn.region_proposal_network \
    import RegionProposalNetwork

from chainer_mask_rcnn import functions
from chainer_mask_rcnn.models.mask_rcnn_resnet import _copy_persistent_chain
from chainer_mask_rcnn.models.resnet_extractor import _convert_bn_to_affine
from chainer_mask_rcnn.models.resnet_extractor import ResNet101Extractor
from chainer_mask_rcnn.models.resnet_extractor import ResNet50Extractor

from .mask_rcnn import MaskRCNN


class MaskRCNNResNet(MaskRCNN):

    feat_stride = 16

    def __init__(self,
                 n_layers,
                 n_fg_class=None,
                 pretrained_model=None,
                 min_size=600,
                 max_size=1000,
                 mask_loss='softmax',
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
            roi_size=roi_size, spatial_scale=1. / self.feat_stride,
            res_initialW=res_initialW,
            loc_initialW=loc_initialW,
            score_initialW=score_initialW,
            mask_initialW=mask_initialW,
            pooling_func=pooling_func,
            mask_loss=mask_loss,
        )

        if len(mean) != 3:
            raise ValueError('The mean must be tuple of RGB values.')
        mean = np.asarray(mean, dtype=np.float32)[:, None, None]

        super(MaskRCNNResNet, self).__init__(
            extractor,
            rpn,
            head,
            mean=mean,
            min_size=min_size,
            max_size=max_size,
            mask_loss=mask_loss,
        )

        if pretrained_model:
            chainer.serializers.load_npz(pretrained_model, self)


class ResNetRoIHead(chainer.Chain):

    def __init__(self, n_layers, n_class, roi_size, spatial_scale,
                 pretrained_model='auto',
                 res_initialW=None, loc_initialW=None, score_initialW=None,
                 mask_initialW=None,
                 pooling_func=functions.roi_align_2d, mask_loss='softmax'):
        # n_class includes the background
        super(ResNetRoIHead, self).__init__()
        with self.init_scope():
            self.res5 = BuildingBlock(
                3, 1024, 512, 2048, stride=roi_size // 7,
                initialW=res_initialW)
            self.cls_loc = L.Linear(2048, n_class * 4, initialW=loc_initialW)
            self.score = L.Linear(2048, n_class, initialW=score_initialW)

            self.mask_loss = mask_loss

            # 7 x 7 x 2048 -> 14 x 14 x 256
            self.deconv6 = L.Deconvolution2D(
                2048, 256, 2, stride=2, initialW=mask_initialW
            )
            # 14 x 14 x 256
            n_fg_class = n_class - 1
            if self.mask_loss in ['softmax', 'softmax_x2']:
                # -> 14 x 14 x (n_fg_class * 3) (bg vs. vis vs. inv)
                self.mask = L.Convolution2D(
                    256, n_fg_class * 3, 1, initialW=mask_initialW
                )
            elif self.mask_loss == 'sigmoid_softmax':
                # -> 14 x 14 x n_fg_class: vis
                self.mask = L.Convolution2D(
                    256, n_fg_class, 1, initialW=mask_initialW
                )
                # -> 14 x 14 x (n_fg_class * 2): bg vs. inv
                self.mask_bginv = L.Convolution2D(
                    256, n_fg_class * 2, 1, initialW=mask_initialW
                )
            elif self.mask_loss in ['sigmoid_sigmoid', 'sigmoid_sigmoid+']:
                # -> 14 x 14 x n_fg_class: vis
                self.mask = L.Convolution2D(
                    256, n_fg_class, 1, initialW=mask_initialW
                )
                # -> 14 x 14 x n_fg_class: vis + inv
                self.mask_visinv = L.Convolution2D(
                    256, n_fg_class, 1, initialW=mask_initialW
                )
            elif self.mask_loss in [
                'softmax_relook_softmax',
                'softmax_relook_softmax+',
                'softmax_relook_softmax+_res',
                'softmax_relook_softmax_cls',
                'softmax_relook_softmax+_cls',
                'softmax_relook_softmax_tt',
                'softmax_relook_softmax+_tt',
                'softmax_relook_softmax+_tt2',
                'softmax_relook_softmax_cls_tt',
                'softmax_relook_softmax+_cls_tt',
                'softmax_relook_softmax_bbox',
                'softmax_relook_softmax+_bbox',
            ]:
                self.mask = L.Convolution2D(
                    in_channels=256,
                    out_channels=n_fg_class * 3,
                    ksize=1,
                    initialW=mask_initialW,
                )
                if '_cls' in self.mask_loss:
                    self.conv5 = L.Convolution2D(
                        in_channels=n_fg_class * 3 + 1024,
                        out_channels=1024,
                        ksize=3,
                        pad=1,
                        initialW=mask_initialW,
                    )
                elif '_res' in self.mask_loss:
                    self.conv5 = L.Convolution2D(
                        in_channels=3,
                        out_channels=1024,
                        ksize=3,
                        pad=1,
                        initialW=mask_initialW,
                    )
                else:
                    self.conv5 = L.Convolution2D(
                        in_channels=3 + 1024,
                        out_channels=1024,
                        ksize=3,
                        pad=1,
                        initialW=mask_initialW,
                    )
                self.mask2 = L.Convolution2D(
                    in_channels=256,
                    out_channels=3,
                    ksize=1,
                    initialW=mask_initialW,
                )
            else:
                raise ValueError

        self.n_class = n_class
        self.roi_size = roi_size
        self.spatial_scale = spatial_scale
        self.pooling_func = pooling_func

        _convert_bn_to_affine(self)
        self._copy_imagenet_pretrained_resnet(n_layers)

    def _copy_imagenet_pretrained_resnet(self, n_layers):
        if n_layers == 50:
            pretrained_model = ResNet50Extractor(pretrained_model='auto')
        elif n_layers == 101:
            pretrained_model = ResNet101Extractor(pretrained_model='auto')
        else:
            raise ValueError
        self.res5.copyparams(pretrained_model.res5)
        _copy_persistent_chain(self.res5, pretrained_model.res5)

    def __call__(self, x, rois, roi_indices, pred_bbox=True, pred_mask=True,
                 pred_bbox2=False, pred_mask2=True, labels=None):
        roi_indices = roi_indices.astype(np.float32)
        indices_and_rois = self.xp.concatenate(
            (roi_indices[:, None], rois), axis=1)
        pool = _roi_pooling_2d_yx(
            x, indices_and_rois, self.roi_size, self.roi_size,
            self.spatial_scale, self.pooling_func)

        with chainer.using_config('train', False):
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
            if self.mask_loss in ['softmax', 'softmax_x2']:
                roi_masks = self.mask(deconv6)
            elif self.mask_loss == 'sigmoid_softmax':
                roi_masks = self.mask(deconv6)
                roi_masks_bginv = self.mask_bginv(deconv6)
                roi_masks = (roi_masks, roi_masks_bginv)
            elif self.mask_loss == 'sigmoid_sigmoid':
                roi_masks = self.mask(deconv6)
                roi_masks_visinv = self.mask_visinv(deconv6)
                roi_masks = (roi_masks, roi_masks_visinv)
            elif self.mask_loss == 'sigmoid_sigmoid+':
                roi_masks = self.mask(deconv6)
                roi_masks_visinv = self.mask_visinv(deconv6)
                roi_masks = (roi_masks, roi_masks + roi_masks_visinv)
            elif self.mask_loss in [
                'softmax_relook_softmax',
                'softmax_relook_softmax+',
                'softmax_relook_softmax+_res',
                'softmax_relook_softmax_cls',
                'softmax_relook_softmax+_cls',
                'softmax_relook_softmax_tt',
                'softmax_relook_softmax+_tt',
                'softmax_relook_softmax+_tt2',
                'softmax_relook_softmax_cls_tt',
                'softmax_relook_softmax+_cls_tt',
                'softmax_relook_softmax_bbox',
                'softmax_relook_softmax+_bbox',
            ]:
                assert labels is not None

                # roi_masks: (n_roi, n_fg_class, 14, 14) -> (n_roi, 14, 14)
                # print('deconv6', deconv6.shape)
                roi_masks = self.mask(deconv6)
                # print('roi_masks', roi_masks.shape)

                n_roi = rois.shape[0]
                # print('labels', labels.shape)

                n_positive = int((labels > 0).sum())
                # print('n_positive', n_positive)

                labels = labels[:n_positive]
                rois = rois[:n_positive]
                # indices_and_rois = indices_and_rois[:n_positive]
                # print('labels', labels.shape)
                # print('rois', rois.shape)

                roi_masks = F.reshape(
                    roi_masks,
                    (n_roi, -1, 3, roi_masks.shape[2], roi_masks.shape[3]))
                # print('roi_masks', roi_masks.shape)
                roi_masks = roi_masks[np.arange(n_positive), labels - 1]
                assert (labels == 0).sum() == 0
                # print('roi_masks', roi_masks.shape)

                if '_cls' in self.mask_loss:
                    whole_masks = roi_mask_to_whole_mask(
                        F.softmax(roi_masks).array,
                        rois, x.shape[2:4], self.spatial_scale,
                        fg_labels=labels - 1, n_fg_class=self.n_class - 1)
                else:
                    whole_masks = roi_mask_to_whole_mask(
                        F.softmax(roi_masks).array,
                        rois, x.shape[2:4], self.spatial_scale)
                # print('whole_masks', whole_masks.shape)
                whole_masks = F.reshape(
                    whole_masks,
                    (1, -1, whole_masks.shape[2], whole_masks.shape[3]))
                # print('whole_masks', whole_masks.shape)

                if '_res' in self.mask_loss:
                    h = self.conv5(whole_masks)
                    h = F.relu(h + x)
                else:
                    h = F.concat([whole_masks, x], axis=1)
                    # print('h', h.shape)

                    h = F.relu(self.conv5(h))  # 1/16, whole
                    # print('h', h.shape)

                h = _roi_pooling_2d_yx(
                    h, indices_and_rois,
                    self.roi_size, self.roi_size,
                    self.spatial_scale, self.pooling_func)
                # print('h', h.shape)  # 1/16, roi

                with chainer.using_config('train', False):
                    res5 = self.res5(h)
                # print('h', h.shape)  # 1/16, roi

                if pred_bbox2:
                    pool5 = F.average_pooling_2d(res5, 7, stride=7)
                    roi_cls_locs2 = self.cls_loc(pool5)
                    roi_scores2 = self.score(pool5)
                    roi_cls_locs = (roi_cls_locs, roi_cls_locs2)
                    roi_scores = (roi_scores, roi_scores2)

                roi_masks2 = None
                if pred_mask2:
                    h = F.relu(self.deconv6(res5))
                    h = h[:n_positive, :, :, :]
                    # print('h', h.shape)  # 1/8, roi

                    roi_masks2 = self.mask2(h)  # 1/8, roi
                    # print('roi_masks2', roi_masks2.shape)

                roi_masks = (roi_masks, roi_masks2)
            else:
                raise ValueError

        return roi_cls_locs, roi_scores, roi_masks


def roi_mask_to_whole_mask(roi_masks, rois, img_shape, spatial_scale,
                           fg_labels=None, n_fg_class=None):
    class_specific = False
    if fg_labels is not None or n_fg_class is not None:
        assert fg_labels is not None
        assert n_fg_class is not None
        class_specific = True

    xp = chainer.cuda.get_array_module(roi_masks)
    rois = (rois * spatial_scale).astype(xp.int32)
    rois[:, 0::2] = xp.clip(rois[:, 0::2], 0, img_shape[0])
    rois[:, 1::2] = xp.clip(rois[:, 1::2], 0, img_shape[1])

    n_roi = roi_masks.shape[0]
    assert rois.shape[0] == n_roi
    if class_specific:
        masks = xp.zeros(
            (1, n_fg_class, roi_masks.shape[1], img_shape[0], img_shape[1]),
            dtype=xp.float32)
    else:
        masks = xp.zeros(
            (1, roi_masks.shape[1], img_shape[0], img_shape[1]),
            dtype=xp.float32)
    for i in range(n_roi):
        roi_mask = roi_masks[i]
        y1, x1, y2, x2 = rois[i]
        roi_H = int(y2 - y1)
        roi_W = int(x2 - x1)
        y1, x1, y2, x2 = map(int, [y1, x1, y2, x2])
        roi_mask = F.resize_images(
            roi_mask[None, :, :, :], (roi_H, roi_W)).array[0, :, :, :]
        if class_specific:
            fg_label = fg_labels[i]
            masks[0, fg_label, :, y1:y2, x1:x2] += roi_mask
        else:
            masks[0, :, y1:y2, x1:x2] += roi_mask
    masks = masks.reshape(1, -1, img_shape[0], img_shape[1])
    return masks


def _roi_pooling_2d_yx(x, indices_and_rois, outh, outw, spatial_scale,
                       pooling_func):
    xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
    pool = pooling_func(x, xy_indices_and_rois, outh, outw, spatial_scale)
    return pool
