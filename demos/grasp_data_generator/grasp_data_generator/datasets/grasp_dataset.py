import argparse
import chainer
import imgaug.augmenters as iaa
from imgaug.parameters import Deterministic
import numpy as np
import os
import os.path as osp
import scipy.misc
import skimage.color
from sklearn.model_selection import train_test_split
import yaml


filepath = osp.dirname(osp.realpath(__file__))


class GraspDatasetBase(chainer.dataset.DatasetMixin):

    yamlpath = osp.join(filepath, '../../yaml/label_names.yaml')

    def __init__(self, split='train', random_state=1234):
        super(GraspDatasetBase, self).__init__()

        with open(self.yamlpath) as f:
            self.label_names = yaml.load(f)
        self.split = split

        st = lambda x: iaa.Sometimes(0.3, x)  # NOQA

        ids = os.listdir(self.datadir)
        ids = [d for d in ids if osp.isdir(osp.join(self.datadir, d))]
        ids_train, ids_valid = train_test_split(
            ids, test_size=0.1, random_state=random_state)
        self._ids = {'all': ids, 'train': ids_train, 'valid': ids_valid}

    def __len__(self):
        return len(self._ids[self.split])

    @classmethod
    def label_names(cls):
        with open(cls.yamlpath) as f:
            label_names = yaml.load(f)
        return label_names


class GraspDataset(GraspDatasetBase):

    datadir = osp.join(
        filepath, '../../data/training_data/', '20170829_195648')

    def __init__(
            self, split='train', random_state=1234,
            imgaug=True, threshold=0.3
    ):
        super(GraspDataset, self).__init__(split, random_state)
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
        grasp_img = loaded['grasp']
        if self.aug:
            img = self.color_aug.augment_image(img)
            aug = self.aug.to_deterministic()
            aug[0].order = Deterministic(1)
            img = aug.augment_image(img)
            grasp_img = aug.augment_image(grasp_img)
            aug[0].order = Deterministic(0)
            label = aug.augment_image(label)
        grasp_img = grasp_img > (self.threshold * 255.0)
        grasp_img = grasp_img.astype(np.int32)
        return img, label, grasp_img


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--aug', '-a', action='store_true')
    args = parser.parse_args()

    imgaug = args.aug
    dataset = GraspDataset('all', imgaug=imgaug)

    for i in range(0, 10):
        img, label, grasp_img = dataset.get_example(i)

        # label_viz
        label_viz = skimage.color.label2rgb(label, img, bg_label=0)
        label_viz = (label_viz * 255).astype(np.int32)

        # grasp_viz
        grasp_viz = img.copy()
        grasp_img_viz = (grasp_img * 255).astype(np.int32)
        grasp_img_viz = np.repeat(grasp_img_viz[:, :, np.newaxis], 3, 2)
        grasp_img_viz = grasp_img_viz * np.array([1, 0, 0])
        grasp_viz = (0.3 * grasp_viz + 0.7 * grasp_img_viz).astype(np.int32)

        viz = np.concatenate((img, label_viz, grasp_viz), axis=1)
        scipy.misc.imshow(viz)
