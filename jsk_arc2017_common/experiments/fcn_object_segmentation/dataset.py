#!/usr/bin/env python

import os
import os.path as osp

import click
import cv2
import fcn
import numpy as np
import skimage.io
import torch.utils.data

import rospkg


PKG_PATH = rospkg.RosPack().get_path('jsk_arc2017_common')


def get_object_names():
    object_names = ['__background__']
    with open(osp.join(PKG_PATH, 'data/names/objects.txt')) as f:
        object_names += [x.strip() for x in f]
    object_names.append('__shelf__')
    return object_names


class ARC2017Base(torch.utils.data.Dataset):

    object_names = np.array(get_object_names())
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
        viz = fcn.utils.draw_label(
            lbl, img, n_class=len(self.object_names),
            label_titles=dict(enumerate(self.object_names)))
        return viz


class JSKV1(ARC2017Base):

    def __init__(self, transform=True):
        self._transform = transform
        dataset_dir = osp.join(PKG_PATH, 'data/datasets/JSKV1')
        self._ids = []
        for scene_dir in os.listdir(dataset_dir):
            scene_dir = osp.join(dataset_dir, scene_dir)
            self._ids.append(scene_dir)

    def __len__(self):
        return len(self._ids)

    def __getitem__(self, i):
        scene_dir = self._ids[i]
        img_file = osp.join(scene_dir, 'image.jpg')
        img = skimage.io.imread(img_file)
        lbl_file = osp.join(scene_dir, 'label.npz')
        lbl = np.load(lbl_file)['arr_0']
        if self._transform:
            return self.transform(img, lbl)
        else:
            return img, lbl


def main():
    dataset = JSKV1(transform=False)
    for i in xrange(len(dataset)):
        img, lbl = dataset[i]
        viz = dataset.visualize(img, lbl)
        cv2.imshow('JSKV1', viz)
        if cv2.waitKey(0) == ord('q'):
            break


if __name__ == '__main__':
    main()
