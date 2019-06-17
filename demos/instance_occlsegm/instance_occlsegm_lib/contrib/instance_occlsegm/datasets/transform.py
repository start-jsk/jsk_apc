import chainer_mask_rcnn
from chainercv import transforms

from ..models.mask_rcnn.mask_rcnn_panoptic import pad_multiple_of


class MaskRCNNTransform(chainer_mask_rcnn.datasets.MaskRCNNTransform):

    def __call__(self, in_data):
        out_data = super(MaskRCNNTransform, self).__call__(in_data)

        if not self.train:
            return out_data

        img, bbox, label, mask, scale = out_data

        keep = (mask == 1).sum(axis=(1, 2)) > 0
        bbox = bbox[keep]
        label = label[keep]
        mask = mask[keep]

        return img, bbox, label, mask, scale


class MaskRCNNPanopticTransform(object):

    def __init__(self, mask_rcnn, train=True):
        self.mask_rcnn = mask_rcnn
        self.train = train

    def __call__(self, in_data):
        assert len(in_data) == 6
        img, bbox, label, mask, lbl_vis, lbl_occ = in_data

        # H, W, C -> C, H, W
        img = img.transpose(2, 0, 1)
        lbl_occ = lbl_occ.transpose(2, 0, 1)

        if not self.train:
            return img, bbox, label, mask, lbl_vis, lbl_occ

        imgs, sizes, scales = self.mask_rcnn.prepare([img])
        img = imgs[0]
        H, W = sizes[0]
        scale = scales[0]
        # _, o_H, o_W = img.shape

        o_H, o_W = int(round(scale * H)), int(round(scale * W))

        if len(bbox) > 0:
            bbox = transforms.resize_bbox(bbox, (H, W), (o_H, o_W))
        if len(mask) > 0:
            mask = transforms.resize(
                mask, size=(o_H, o_W), interpolation=0)
            mask = mask.transpose(1, 2, 0)
            mask = pad_multiple_of(mask, mode='constant', constant_values=-1)
            mask = mask.transpose(2, 0, 1)
            assert mask.shape[1:] == img.shape[1:]

        lbl_vis = transforms.resize(
            lbl_vis[None], size=(o_H, o_W), interpolation=0
        )[0]
        lbl_occ = transforms.resize(lbl_occ, size=(o_H, o_W), interpolation=0)
        lbl_vis = pad_multiple_of(lbl_vis, mode='constant', constant_values=-1)
        lbl_occ = lbl_occ.transpose(1, 2, 0)
        lbl_occ = pad_multiple_of(lbl_occ, mode='constant', constant_values=-1)
        lbl_occ = lbl_occ.transpose(2, 0, 1)
        assert lbl_vis.shape == img.shape[1:]
        assert lbl_occ.shape[1:] == img.shape[1:]

        # # horizontally flip
        # img, params = transforms.random_flip(
        #     img, x_random=True, return_param=True)
        # bbox = transforms.flip_bbox(
        #     bbox, (o_H, o_W), x_flip=params['x_flip'])
        # if mask.ndim == 2:
        #     mask = transforms.flip(
        #         mask[None, :, :], x_flip=params['x_flip'])[0]
        # else:
        #     mask = transforms.flip(mask, x_flip=params['x_flip'])
        # lbl_vis = transforms.flip(lbl_vis[None], x_flip=params['x_flip'])[0]
        # lbl_occ = transforms.flip(lbl_occ, x_flip=params['x_flip'])

        keep = (mask == 1).sum(axis=(1, 2)) > 0
        bbox = bbox[keep]
        label = label[keep]
        mask = mask[keep]

        return img, bbox, label, mask, scale, lbl_vis, lbl_occ
