import os.path as osp

import chainer
import numpy as np

import grasp_fusion_lib


class RealInstanceSegmentationDataset(chainer.dataset.DatasetMixin):

    def __init__(self):
        self._dataset = chainer.datasets.ConcatenatedDataset(
            grasp_fusion_lib.datasets.apc.ARC2017InstanceSegmentationDataset(
                'train'),
            grasp_fusion_lib.datasets.apc.ARC2017InstanceSegmentationDataset(
                'test'),
        )
        self.class_names = np.array(['object'])
        self.class_names.setflags(write=0)

    def __len__(self):
        return len(self._dataset)

    def get_example(self, i):
        img, _, _, masks = self._dataset.get_example(i)

        masks[masks == 2] = 0
        assert np.isin(masks, (-1, 0, 1)).all()
        bboxes = grasp_fusion_lib.image.masks_to_bboxes(
            masks == 1).astype(np.float32)
        labels = np.full((len(bboxes),), 0, dtype=np.int32)  # object agnostic

        return img, bboxes, labels, masks


class SyntheticInstanceSegmentationDataset(
    grasp_fusion_lib.datasets.apc.
    ARC2017ItemDataSyntheticInstanceSegmentationDataset
):

    def __init__(
        self,
        augmentation=False,
        augmentation_level='all',
        exclude_arc2017=False,
        background='tote',
    ):
        item_data_dir = self._download()
        super(SyntheticInstanceSegmentationDataset, self).__init__(
            item_data_dir,
            do_aug=augmentation,
            aug_level=augmentation_level,
            exclude_arc2017=exclude_arc2017,
            background=background,
            stack_ratio=(0.2, 0.9)
        )
        self.class_names = np.array(['object'])
        self.class_names.setflags(write=0)

    @staticmethod
    def _download():
        path = osp.expanduser('~/data/grasp_fusion_lib/grasp_fusion/ItemDataAll_MaskPred.zip')  # NOQA
        grasp_fusion_lib.data.download(
            url='https://drive.google.com/uc?id=1aBqPnaPETcLqPFfp2ai1Fp32nxSp0Prq',  # NOQA
            path=path,
            md5='9b691e92c7a7f194563768177d22d66f',
            postprocess=grasp_fusion_lib.data.extractall,
        )
        return osp.splitext(path)[0]

    def get_example(self, i):
        img, _, _, masks = \
            super(SyntheticInstanceSegmentationDataset, self).get_example(i)

        masks[masks == 2] = 0
        assert np.isin(masks, (-1, 0, 1)).all()
        bboxes = grasp_fusion_lib.image.masks_to_bboxes(
            masks == 1).astype(np.float32)
        labels = np.full((len(bboxes),), 0, dtype=np.int32)  # object agnostic

        return img, bboxes, labels, masks
