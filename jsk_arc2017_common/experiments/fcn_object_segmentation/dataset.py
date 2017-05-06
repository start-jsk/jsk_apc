#!/usr/bin/env python

import os
import os.path as osp

import click
import cv2
import fcn
import imgaug.augmenters as iaa
import lblaug
import numpy as np
import skimage.io
from sklearn.model_selection import train_test_split
import torch.utils.data
import yaml


this_dir = osp.dirname(osp.abspath(__file__))


def get_object_names():
    with open(osp.join(this_dir, '../../config/label_names.yaml')) as f:
        return yaml.load(f)['label_names']


class ARC2017Base(torch.utils.data.Dataset):

    class_names = np.array(get_object_names())
    mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])

    def transform(self, img, lbl):
        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float64)
        img -= self.mean_bgr
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()
        return img, lbl

    def untransform(self, img, lbl):
        img = img.numpy()
        img = img.transpose(1, 2, 0)
        img += self.mean_bgr
        img = img.astype(np.uint8)
        img = img[:, :, ::-1]
        lbl = lbl.numpy()
        return img, lbl

    def visualize(self, img, lbl):
        lbl = lbl.copy()
        lbl[lbl == -1] = 0
        lbl_viz = fcn.utils.draw_label(
            lbl, img, n_class=len(self.class_names),
            label_titles=dict(enumerate(self.class_names)))
        viz = fcn.utils.get_tile_image([img, lbl_viz], tile_shape=(1, 2))
        return viz


class JSKV1(ARC2017Base):

    aug = lblaug.LabelAugmentation(object_labels=range(1, 41), region_label=41)

    def __init__(self, split='train', transform=True, aug='standard'):
        self.split = split
        self._transform = transform
        self.aug_method = aug
        dataset_dir = osp.expanduser('~/data/projects/arc2017/Data/JSKV1')
        ids = []
        for scene_dir in os.listdir(dataset_dir):
            scene_dir = osp.join(dataset_dir, scene_dir)
            if osp.exists(osp.join(scene_dir, 'label.npz')):
                ids.append(scene_dir)
        self._ids = {}
        self._ids['train'], self._ids['valid'] = \
            train_test_split(ids, test_size=0.25, random_state=1)

    def __len__(self):
        return len(self._ids[self.split])

    def __getitem__(self, i, aug='train'):
        scene_dir = self._ids[self.split][i]
        img_file = osp.join(scene_dir, 'image.jpg')
        img = skimage.io.imread(img_file)
        lbl_file = osp.join(scene_dir, 'label.npz')
        lbl = np.load(lbl_file)['arr_0']
        if aug == self.split:
            if self.aug_method == 'standard':
                img, lbl = self.aug.augment_labeled_image(img, lbl)
            elif self.aug_method == 'stack':
                transform = self._transform
                self._transform = False
                imgs, lbls = [], []
                for j in np.random.randint(0, len(self), 10):
                    img_j, lbl_j = self.__getitem__(j, aug=None)
                    imgs.append(img_j)
                    lbls.append(lbl_j)
                self._transform = transform
                img, lbl = self.aug.stack_labeled_images(img, lbl, imgs, lbls)
                img, lbl = self.aug.augment_labeled_image(img, lbl)
            else:
                raise ValueError
        if self._transform:
            return self.transform(img, lbl)
        else:
            return img, lbl


class JSKARC2017From16(JSKV1):

    def __init__(self, split='train', transform=True, aug='standard'):
        self.split = split
        self._transform = transform
        self.aug_method = aug
        dataset_dir = osp.join(
            this_dir, '../../data/datasets/JSKARC2017From16')
        ids = []
        for scene_dir in os.listdir(dataset_dir):
            if scene_dir == '1466804951244465112':
                # Wrong annotation
                continue
            scene_dir = osp.join(dataset_dir, scene_dir)
            ids.append(scene_dir)
        self._ids = {}
        self._ids['train'], self._ids['valid'] = \
            train_test_split(ids, test_size=0.25, random_state=1)


class DatasetV1(ARC2017Base):

    def __init__(self, split='train', transform=True, aug='standard'):
        self.datasets = [
            JSKV1(split, transform, aug),
            JSKARC2017From16(split, transform, aug),
        ]

    def __len__(self):
        return sum(len(d) for d in self.datasets)

    @property
    def split(self):
        split = self.datasets[0].split
        assert all(d.split == split for d in self.datasets)
        return split

    @split.setter
    def split(self, value):
        for d in self.datasets:
            d.split = value

    def __getitem__(self, index):
        skipped = 0
        for dataset in self.datasets:
            current_index = index - skipped
            if current_index < len(dataset):
                return dataset[current_index]
            skipped += len(dataset)


@click.command()
@click.option('-d', '--dataset', default='V1',
              type=click.Choice(['V1', 'JSKV1', 'JSKARC2017From16']),
              show_default=True)
@click.option('--aug', default='standard',
              type=click.Choice(['standard', 'stack']), show_default=True)
def main(**args):
    if args['dataset'] == 'V1':
        dataset = DatasetV1(transform=False, aug=args['aug'])
        dataset_valid = DatasetV1(split='valid')
    elif args['dataset'] == 'JSKV1':
        dataset = JSKV1(transform=False, aug=args['aug'])
        dataset_valid = JSKV1(split='valid')
    else:
        dataset = JSKARC2017From16(transform=False, aug=args['aug'])
        dataset_valid = JSKARC2017From16(split='valid')
    print('Size of %s: train: %d, valid: %d' %
          (args['dataset'], len(dataset), len(dataset_valid)))
    del dataset_valid
    for i in xrange(len(dataset)):
        img, lbl = dataset[i]
        viz = dataset.visualize(img, lbl)
        cv2.imshow(args['dataset'], viz[:, :, ::-1])
        if cv2.waitKey(0) == ord('q'):
            break


if __name__ == '__main__':
    main()
