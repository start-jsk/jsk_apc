import argparse
import imgaug.augmenters as iaa
from imgaug.parameters import Deterministic
import numpy as np
import os.path as osp
import scipy.misc
import skimage.color

from grasp_data_generator.datasets.grasp_dataset import GraspDatasetBase


filepath = osp.dirname(osp.realpath(__file__))


class DualarmGraspDataset(GraspDatasetBase):

    def __init__(
            self, split='train', random_state=1234,
            imgaug=True, threshold=0.3
    ):
        super(DualarmGraspDataset, self).__init__(split, random_state)
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

    def __len__(self):
        return len(self._ids[self.split])

    def get_example(self, i):
        data_id = self._ids[self.split][i]
        loaded = np.load(osp.join(self.datadir, data_id, 'data.npz'))
        img = loaded['img']
        label = loaded['label']
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
            label = aug.augment_image(label)
        single_grasp = single_grasp > (self.threshold * 255.0)
        single_grasp = single_grasp.astype(np.int32)
        dual_grasp = dual_grasp > (self.threshold * 255.0)
        dual_grasp = dual_grasp.astype(np.int32)
        return img, label, single_grasp, dual_grasp

    def visualize(self, i):
        img, label, single_grasp, dual_grasp = self.get_example(i)

        # label_viz
        label_viz = skimage.color.label2rgb(label, img, bg_label=0)
        label_viz = (label_viz * 255).astype(np.int32)

        # single_grasp_viz
        single_grasp_viz = img.copy()
        single_grasp = (single_grasp * 255).astype(np.int32)
        single_grasp = np.repeat(single_grasp[:, :, np.newaxis], 3, 2)
        single_grasp = single_grasp * np.array([1, 0, 0])
        single_grasp_viz = 0.3 * single_grasp_viz + 0.7 * single_grasp
        single_grasp_viz = single_grasp_viz.astype(np.int32)

        # dual_grasp_viz
        dual_grasp_viz = img.copy()
        dual_grasp = (dual_grasp * 255).astype(np.int32)
        dual_grasp = np.repeat(dual_grasp[:, :, np.newaxis], 3, 2)
        dual_grasp = dual_grasp * np.array([1, 0, 0])
        dual_grasp_viz = 0.3 * dual_grasp_viz + 0.7 * dual_grasp
        dual_grasp_viz = dual_grasp_viz.astype(np.int32)

        viz = np.concatenate(
            (img, label_viz, single_grasp_viz, dual_grasp_viz), axis=1)
        scipy.misc.imshow(viz)


class DualarmGraspDatasetV1(DualarmGraspDataset):
    # human annotated
    datadir = osp.join(
        filepath, '../../data/training_data/', '20170925_194635')


class DualarmGraspDatasetV2(DualarmGraspDataset):
    # human annotated
    datadir = osp.join(
        filepath, '../../data/training_data/', '20170928_175025')


class DualarmGraspDatasetV3(DualarmGraspDataset):
    # human annotated
    datadir = osp.join(
        filepath, '../../data/training_data/', '20180124_201609')


class DualarmGraspDatasetV4(DualarmGraspDataset):
    # no human annotated
    datadir = osp.join(
        filepath, '../../data/training_data/', '20180129_180625')


class DualarmGraspDatasetV5(DualarmGraspDataset):
    # no human annotated
    yamlpath = osp.join(
        filepath, '../../data/training_data/', '20180216_161425',
        'label_names.yaml')
    datadir = osp.join(
        filepath, '../../data/training_data/', '20180216_161425')


class DualarmGraspDatasetV6(DualarmGraspDataset):
    # human annotated
    yamlpath = osp.join(
        filepath, '../../data/training_data/', '20180216_162048',
        'label_names.yaml')
    datadir = osp.join(
        filepath, '../../data/training_data/', '20180216_162048')


class DualarmGraspDatasetV7(DualarmGraspDataset):
    # no human annotated
    yamlpath = osp.join(
        filepath, '../../data/training_data/', '20180219_184420',
        'label_names.yaml')
    datadir = osp.join(
        filepath, '../../data/training_data/', '20180219_184420')


class DualarmGraspDatasetV8(DualarmGraspDataset):
    # human annotated
    yamlpath = osp.join(
        filepath, '../../data/training_data/', '20180219_184445',
        'label_names.yaml')
    datadir = osp.join(
        filepath, '../../data/training_data/', '20180219_184445')


class FinetuningDualarmGraspDatasetV1(DualarmGraspDataset):
    # only sampled by robot
    # 1st sampling
    yamlpath = osp.join(
        filepath, '../../data/finetuning_data/', '20180222_231433',
        'label_names.yaml')
    datadir = osp.join(
        filepath, '../../data/finetuning_data/', '20180222_231433')


class FinetuningDualarmGraspDatasetV2(DualarmGraspDataset):
    # merging automatic annotation and sampled by robot
    # 1st sampling
    yamlpath = osp.join(
        filepath, '../../data/finetuning_data/', '20180223_183516',
        'label_names.yaml')
    datadir = osp.join(
        filepath, '../../data/finetuning_data/', '20180223_183516')


class FinetuningDualarmGraspDatasetV3(DualarmGraspDataset):
    # merging automatic annotation and sampled by robot
    # 1st sampling and 2nd sampling
    yamlpath = osp.join(
        filepath, '../../data/finetuning_data/', '20180224_170858',
        'label_names.yaml')
    datadir = osp.join(
        filepath, '../../data/finetuning_data/', '20180224_170858')


class FinetuningDualarmGraspDatasetV4(DualarmGraspDataset):
    # only sampled by robot
    # 1st sampling and 2nd sampling
    yamlpath = osp.join(
        filepath, '../../data/finetuning_data/', '20180226_145155',
        'label_names.yaml')
    datadir = osp.join(
        filepath, '../../data/finetuning_data/', '20180226_145155')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--aug', '-a', action='store_true')
    args = parser.parse_args()

    imgaug = args.aug
    dataset = DualarmGraspDataset('all', imgaug=imgaug)

    for i in range(0, 100):
        dataset.visualize(i)
