import collections
import os.path as osp

import numpy as np
import PIL.Image

from . import config
from .base import ClassSegmentationDatasetBase
import grasp_fusion_lib.data


class VOCClassSegmentationDataset(ClassSegmentationDatasetBase):

    _class_names = np.array([
        'background',
        'aeroplane',
        'bicycle',
        'bird',
        'boat',
        'bottle',
        'bus',
        'car',
        'cat',
        'chair',
        'cow',
        'diningtable',
        'dog',
        'horse',
        'motorbike',
        'person',
        'potted plant',
        'sheep',
        'sofa',
        'train',
        'tv/monitor',
    ])
    _class_names.setflags(write=False)

    def __init__(self, split):
        assert split in ['train', 'val']
        super(VOCClassSegmentationDataset, self).__init__(split=split)

        dataset_dir = osp.join(config.ROOT_DIR, 'VOC/VOCdevkit/VOC2012')
        if not osp.exists(dataset_dir):
            self.download()

        # VOC20XX is subset of VOC2012
        self._files = collections.defaultdict(list)
        for split in ['train', 'val']:
            imgsets_file = osp.join(
                dataset_dir, 'ImageSets/Segmentation/%s.txt' % split)
            for did in open(imgsets_file):
                did = did.strip()
                img_file = osp.join(dataset_dir, 'JPEGImages/%s.jpg' % did)
                lbl_file = osp.join(
                    dataset_dir, 'SegmentationClass/%s.png' % did)
                self._files[split].append({
                    'img': img_file,
                    'lbl': lbl_file,
                })

    def download(self):
        url = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar'  # NOQA
        grasp_fusion_lib.data.download(
            url=url,
            path=osp.join(config.ROOT_DIR, 'VOC', osp.basename(url)),
            md5='6cd6e144f989b92b3379bac3b3de84fd',
            postprocess=grasp_fusion_lib.data.extractall,
        )

    def __len__(self):
        return len(self._files[self.split])

    def __getitem__(self, index):
        data_file = self._files[self.split][index]
        # img
        img_file = data_file['img']
        img = PIL.Image.open(img_file)
        img = np.array(img, dtype=np.uint8)
        # lbl
        lbl_file = data_file['lbl']
        lbl = PIL.Image.open(lbl_file)
        lbl = np.array(lbl, dtype=np.int32)
        lbl[lbl == 255] = -1
        return img, lbl
