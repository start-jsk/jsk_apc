from __future__ import division
from __future__ import print_function

import os.path as osp
import shutil
import tempfile

import chainer
import imgaug.augmenters as iaa
import imgaug.imgaug as ia
from imgaug.parameters import Deterministic
import numpy as np
import skimage.io


import grasp_fusion_lib.data


class SuctionDataset(chainer.dataset.DatasetMixin):

    root_dir = osp.expanduser(
        '~/data/grasp_fusion_lib/grasp_fusion/suction_dataset')
    channel_names = np.array(['good'])

    def __init__(self, split, augmentation=False):
        assert split in ['train', 'test']
        self.split = split
        self._augmentation = augmentation

        if not osp.exists(self.root_dir):
            self.download()

        if self.split == 'train':
            split_file = osp.join(self.root_dir, 'train-split.txt')
        else:
            split_file = osp.join(self.root_dir, 'test-split.txt')
        self._data_ids = np.loadtxt(split_file, dtype=str)

    def __len__(self):
        return len(self._data_ids)

    @staticmethod
    def _augment(img, depth, label):
        st = lambda x: iaa.Sometimes(0.5, x)  # NOQA
        augmentations = [
            st(iaa.WithChannels([0, 1], iaa.Multiply([1, 1.5]))),
            st(
                iaa.InColorspace(
                    'HSV',
                    children=iaa.WithChannels(
                        [1, 2], iaa.Multiply([0.5, 2])
                    ),
                )
            ),
            (iaa.GaussianBlur(sigma=[0, 1])),
            iaa.Sometimes(0.9, iaa.Dropout(p=(0, 0.4), name='dropout')),
            # iaa.CoarseDropout(p=(0, 0.1), size_percent=0.5, name='dropout'),
            iaa.Sometimes(0.9, iaa.Affine(
                order=1,
                cval=0,
                scale=1,
                translate_px=(-96, 96),
                rotate=(-180, 180),
                mode='constant',
            )),
        ]
        aug = iaa.Sequential(augmentations, random_order=True)

        def activator_imgs(images, augmenter, parents, default):
            if isinstance(augmenter, iaa.Affine):
                augmenter.order = Deterministic(1)
                augmenter.cval = Deterministic(0)
            return True

        def activator_depths(images, augmenter, parents, default):
            white_lists = (iaa.Affine, iaa.Sequential, iaa.Sometimes)
            if not isinstance(augmenter, white_lists):
                return False
            if isinstance(augmenter, iaa.Affine):
                augmenter.order = Deterministic(1)
                augmenter.cval = Deterministic(0)
            return True

        def activator_lbls(images, augmenter, parents, default):
            white_lists = (iaa.Affine, iaa.Sequential, iaa.Sometimes)
            if not isinstance(augmenter, white_lists):
                return False
            if isinstance(augmenter, iaa.Affine):
                augmenter.order = Deterministic(0)
                augmenter.cval = Deterministic(-1)
            return True

        aug = aug.to_deterministic()
        img = aug.augment_image(
            img, hooks=ia.HooksImages(activator=activator_imgs)
        )
        depth = aug.augment_image(
            depth, hooks=ia.HooksImages(activator=activator_depths)
        )
        label = aug.augment_image(
            label, hooks=ia.HooksImages(activator=activator_lbls)
        )

        return img, depth, label

    def get_example(self, i):
        data_id = self._data_ids[i]
        basename = data_id.split('-')[0]

        # RGB image
        img_file = osp.join(
            self.root_dir, 'heightmap-color2', (basename + '.png'))
        img = skimage.io.imread(img_file)
        assert img.dtype == np.uint8

        # Depth image
        depth_file = osp.join(
            self.root_dir, 'heightmap-depth2', (basename + '.png'))
        depth = skimage.io.imread(depth_file)
        if depth.dtype == np.uint8:
            assert 0 <= depth.min() and depth.max() <= 255
            depth.dtype = np.uint16
        assert depth.dtype == np.uint16
        depth = depth.astype(np.float32) / 10000.

        # Label image
        suction_file = osp.join(
            self.root_dir, 'heightmap-suction2', (basename + '.png'))
        suction = skimage.io.imread(suction_file)
        assert suction.dtype == np.uint8
        label = suction.astype(np.int32)
        label[label == 255] = -1
        label[label == 128] = 1

        H, W, C = img.shape

        # FIXME: Some depth images are invalid.
        if depth.shape != (H, W):
            depth = np.zeros((H, W), dtype=np.float32)
            label = np.full((H, W), -1, dtype=np.int32)

        assert C == 3
        assert depth.shape == (H, W)
        assert label.shape == (H, W)
        assert np.isin(label, (-1, 0, 1)).all()

        if self._augmentation:
            img, depth, label = self._augment(img, depth, label)

            H, W, C = img.shape
            assert C == 3
            assert depth.shape == (H, W)
            assert label.shape == (H, W)
            assert np.isin(label, (-1, 0, 1)).all()

        # 0.03 is assigned for missing depth in get_heightmaps.py
        label[depth <= 0.03] = 0

        return img, depth, label

    def visualize(self, index):
        img, depth, label = self[index]
        depth = grasp_fusion_lib.image.colorize_depth(
            depth, min_value=0, max_value=0.3)
        assert np.isin(label, (-1, 0, 1)).all()
        label_names = ['not %s' % self.channel_names[0], self.channel_names[0]]
        label = grasp_fusion_lib.image.label2rgb(
            label, img, label_names=label_names)
        viz = grasp_fusion_lib.image.tile([img, depth, label], (1, 3))
        # For small window
        return grasp_fusion_lib.image.resize(viz, size=600 * 600)

    @classmethod
    def download(cls):
        url = 'http://vision.princeton.edu/projects/2017/arc/downloads/suction-based-grasping-dataset.zip'  # NOQA
        path = cls.root_dir + '.zip'

        def postprocess(path):
            tmp_dir = tempfile.mktemp()
            grasp_fusion_lib.data.extractall(path, to=tmp_dir)
            shutil.move(osp.join(tmp_dir, 'data'), cls.root_dir)
            shutil.rmtree(tmp_dir)

        grasp_fusion_lib.data.download(
            url=url,
            path=path,
            md5='f1e18eb1b707099378d59e845cc42c7e',
            postprocess=postprocess,
        )
