import imgaug.augmenters as iaa
from imgaug.parameters import Deterministic
import numpy as np
import os.path as osp

from grasp_data_generator.datasets.grasp_dataset import GraspDatasetBase


filepath = osp.dirname(osp.realpath(__file__))


class InstanceDualarmGraspDataset(GraspDatasetBase):

    def __init__(
            self, split='train', random_state=1234,
            imgaug=True, threshold=0.3
    ):
        super(InstanceDualarmGraspDataset, self).__init__(
            split, random_state)
        self.threshold = threshold

        st = lambda x: iaa.Sometimes(0.3, x)  # NOQA
        if imgaug:
            self.color_aug = iaa.Sequential(
                [
                    st(iaa.InColorspace(
                        'HSV',
                        children=iaa.WithChannels([1, 2],
                                                  iaa.Multiply([0.5, 2])))),
                    iaa.WithChannels([0, 1], iaa.Multiply([1, 1.5])),
                ],
                random_order=False,
                random_state=random_state)
            self.aug = iaa.Sequential(
                [
                    iaa.Affine(
                        cval=0,
                        order=0,
                        rotate=(-180, 180),
                        mode='constant'),
                    iaa.Fliplr(0.5)
                ],
                random_order=False,
                random_state=random_state)

        else:
            self.aug = False

    def get_example(self, i):
        data_id = self._ids[self.split][i]
        loaded = np.load(osp.join(self.datadir, data_id, 'data.npz'))
        img = loaded['img']
        seg_img = loaded['label']
        ins_img = loaded['ins']
        single_grasp = loaded['single_grasp']
        dual_grasp = loaded['dual_grasp']
        if self.aug:
            img = self.color_aug.augment_image(img)
            aug = self.aug.to_deterministic()
            aug[0].order = Deterministic(1)
            img = aug.augment_image(img)
            single_grasp = aug.augment_image(single_grasp)
            dual_grasp = aug.augment_image(dual_grasp)
            aug[0].order = Deterministic(0)
            seg_img = aug.augment_image(seg_img)
            ins_img = aug.augment_image(ins_img)
        # ins [0, ... N] -> [-1, ... N-1]
        ins_img = ins_img - 1
        single_grasp = single_grasp > (self.threshold * 255.0)
        single_grasp = single_grasp.astype(np.int32)
        dual_grasp = dual_grasp > (self.threshold * 255.0)
        dual_grasp = dual_grasp.astype(np.int32)
        return img, ins_img, single_grasp, dual_grasp


class InstanceDualarmGraspDatasetV1(InstanceDualarmGraspDataset):

    datadir = osp.join(
        filepath, '../../data/training_data/', '20180209_131552')
