from __future__ import division
from __future__ import print_function

import os.path as osp
import shutil
import tempfile
import warnings

import chainer
import imgaug.augmenters as iaa
import imgaug.imgaug as ia
from imgaug.parameters import Deterministic
import numpy as np
import skimage.io

import grasp_fusion_lib.data


class PinchDataset(chainer.dataset.DatasetMixin):

    root_dir = osp.expanduser(
        '~/data/grasp_fusion_lib/grasp_fusion/pinch_dataset')
    channel_names = None

    def __init__(self, split, augmentation=False, resolution=30):
        assert split in ['train', 'test']
        self.split = split
        self._augmentation = augmentation

        assert isinstance(resolution, int)
        assert 0 < resolution < 180
        # assert 180 % resolution == 0
        self._resolution = resolution
        self.channel_names = np.array([
            'good_%d' % (i * resolution)
            for i in range(int(round(180. / resolution)))
        ])

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
    def _augment(img, depth, lines_good, lines_bad):
        keypoints = []
        for lines in [lines_good, lines_bad]:
            for x0, y0, x1, y1 in lines:
                keypoints.append(ia.Keypoint(x=x0, y=y0))
                keypoints.append(ia.Keypoint(x=x1, y=y1))
        keypoints = ia.KeypointsOnImage(keypoints, shape=img.shape)

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
            if (not isinstance(augmenter, white_lists) and
                    augmenter.name != 'dropout'):
                return False
            if isinstance(augmenter, iaa.Affine):
                augmenter.order = Deterministic(1)
                augmenter.cval = Deterministic(0)
            return True

        aug = aug.to_deterministic()
        img = aug.augment_image(
            img, hooks=ia.HooksImages(activator=activator_imgs)
        )
        depth = aug.augment_image(
            depth, hooks=ia.HooksImages(activator=activator_depths)
        )
        keypoints = aug.augment_keypoints([keypoints])[0]

        keypoints = keypoints.get_coords_array()
        keypoints = keypoints.reshape(-1, 2, 2).reshape(-1, 4)
        lines_good = keypoints[:len(lines_good)]
        lines_bad = keypoints[len(lines_good):]

        return img, depth, lines_good, lines_bad

    def _points_to_label(self, label, lines, cval):
        H, W = label.shape[:2]
        for x0, y0, x1, y1 in lines:
            if not ((0 <= x0 <= W) and (0 <= x1 <= W) and
                    (0 <= y0 <= H) and (0 <= y1 <= H)):
                continue

            degree = np.rad2deg(np.arctan2(y1 - y0, x1 - x0))
            if degree < 0.0:
                degree += 180.0  # range: [-180, 180] -> [0, 180]
            degree_id = int(round(degree / self._resolution))
            if degree_id == int(round(180. / self._resolution)):
                degree_id = 0

            cx = int(round((x0 + x1) / 2))
            cy = int(round((y0 + y1) / 2))
            # 0.002m = 2mm (voxel size), so circle radius is 2cm
            rr, cc = skimage.draw.circle(cy, cx, 10, shape=(H, W))
            label[rr, cc, degree_id] = cval

    @staticmethod
    def _load_label_file(label_file):
        with warnings.catch_warnings():
            # filter warnings for empty txt
            warnings.simplefilter('ignore')
            lines = np.loadtxt(label_file, dtype=int, ndmin=2)
            return lines

    def get_example(self, i):
        # RGB image
        img_file = osp.join(
            self.root_dir, 'heightmap-color', (self._data_ids[i] + '.png'))
        img = skimage.io.imread(img_file)
        assert img.dtype == np.uint8

        # Depth image
        depth_file = osp.join(
            self.root_dir, 'heightmap-depth', (self._data_ids[i] + '.png'))
        depth = skimage.io.imread(depth_file)
        if depth.dtype == np.uint8:
            assert 0 <= depth.min() and depth.max() <= 255
            depth.dtype = np.uint16
        depth = depth.astype(np.float32) / 10000.

        # Label image
        label_good_file = osp.join(
            self.root_dir, 'label', (self._data_ids[i] + '.good.txt'))
        lines_good = self._load_label_file(label_good_file)

        label_bad_file = osp.join(
            self.root_dir, 'label', (self._data_ids[i] + '.bad.txt'))
        lines_bad = self._load_label_file(label_bad_file)

        if self._augmentation:
            img, depth, lines_good, lines_bad = self._augment(
                img, depth, lines_good, lines_bad
            )

        # label = np.zeros(img.shape[:2], dtype=np.int32)
        # 6 = 180 degree / 30 degree
        label = - np.ones(
            (img.shape[0], img.shape[1], int(round(180. / self._resolution))),
            dtype=np.int32,
        )
        self._points_to_label(label, lines_good, 1)
        self._points_to_label(label, lines_bad, 0)

        # 0.03 is assigned for missing depth in get_heightmaps.py
        label[depth <= 0.03] = 0

        return img, depth, label

    def visualize(self, index):
        img, depth, label = self[index]
        depth = grasp_fusion_lib.image.colorize_depth(
            depth, min_value=0, max_value=0.3)

        vizs = []
        for degree_id in range(label.shape[2]):
            label_d = label[:, :, degree_id]
            viz = grasp_fusion_lib.image.label2rgb(
                label_d,
                img,
                label_names=[None, 'good_%d' % (degree_id * self._resolution)],
            )
            vizs.append(viz)
        label = grasp_fusion_lib.image.tile(vizs, boundary=True)

        img = grasp_fusion_lib.image.resize(img, height=label.shape[0])
        depth = grasp_fusion_lib.image.resize(depth, height=label.shape[0])

        viz = np.hstack([img, depth, label])
        # For small window
        return grasp_fusion_lib.image.resize(viz, size=1200 * 1200)

    @classmethod
    def download(cls):
        url = 'http://vision.princeton.edu/projects/2017/arc/downloads/parallel-jaw-grasping-dataset.zip'  # NOQA
        path = cls.root_dir + '.zip'

        def postprocess(path):
            tmp_dir = tempfile.mktemp()
            grasp_fusion_lib.data.extractall(path, to=tmp_dir)
            shutil.move(osp.join(tmp_dir, 'data'), cls.root_dir)
            shutil.rmtree(tmp_dir)

        grasp_fusion_lib.data.download(
            url=url,
            path=path,
            md5='3c3d91cd61033129bcd0d95ee3b65944',
            postprocess=postprocess,
        )
