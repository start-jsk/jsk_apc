import os.path as osp

import numpy as np
import PIL.Image
import PIL.ImageDraw
import pycocotools.coco
import pycocotools.mask
import skimage.io

import instance_occlsegm_lib.data
from instance_occlsegm_lib.datasets.base import ClassSegmentationDatasetBase
from instance_occlsegm_lib.datasets import config


class COCOClassSegmentationDataset(ClassSegmentationDatasetBase):

    _root_dir = osp.join(config.ROOT_DIR, 'COCO')

    def __init__(self, split):
        assert split in ['train', 'val', 'minival']
        self._split = split
        split_ann = '%s2014' % split
        split_dir = 'val2014' if split.endswith('val') else 'train2014'

        ann_file = osp.join(
            self._root_dir, 'annotations/instances_%s.json' % split_ann)
        if not osp.exists(ann_file):
            self.download()
        self.coco = pycocotools.coco.COCO(ann_file)

        self.img_fname = osp.join(
            self._root_dir, split_dir, 'COCO_%s_{:012}.jpg' % split_dir)

        # setup class_names
        labels = self.coco.loadCats(self.coco.getCatIds())
        max_label = max(labels, key=lambda x: x['id'])['id']
        n_label = max_label + 1
        self._class_names = [None] * n_label
        for label in labels:
            self._class_names[label['id']] = label['name']
        self._class_names[0] = '__background__'

        self.img_ids = self.coco.getImgIds()

    def download(self):
        data = [
            (None,  # '0da8c0bd3d6becc4dcb32757491aca88',
             'http://msvocds.blob.core.windows.net/coco2014/train2014.zip',
             'train2014.zip'),
            (None,  # 'a3d79f5ed8d289b7a7554ce06a5782b3',
             'http://msvocds.blob.core.windows.net/coco2014/val2014.zip',
             'val2014.zip'),
            ('395a089042d356d97017bf416e4e99fb',
             'https://dl.dropboxusercontent.com/s/o43o90bna78omob/instances_minival2014.json.zip',  # NOQA
             'annotations/instances_minival2014.json.zip'),
            ('59582776b8dd745d649cd249ada5acf7',
             'http://msvocds.blob.core.windows.net/annotations-1-0-3/instances_train-val2014.zip',  # NOQA
             'instances_train-val2014.zip'),
        ]
        for md5, url, basename in data:
            instance_occlsegm_lib.data.download(
                url=url,
                path=osp.join(self._root_dir, basename),
                md5=md5,
                quiet=False,
                postprocess=instance_occlsegm_lib.data.extractall,
            )

    def __getitem__(self, index):
        img_id = self.img_ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        img_fname = self.img_fname.format(img_id)
        img = skimage.io.imread(img_fname)

        lbl = self._annotations_to_label(anns, img.shape[0], img.shape[1])

        return img, lbl

    @staticmethod
    def _annotations_to_label(anns, height, width):
        label = np.zeros((height, width), dtype=np.int32)
        label.fill(0)
        for ann in anns:
            if 'segmentation' not in ann:
                continue
            if isinstance(ann['segmentation'], list):
                # polygon
                for seg in ann['segmentation']:
                    mask = np.zeros((height, width), dtype=np.uint8)
                    mask = PIL.Image.fromarray(mask)
                    xy = np.array(seg).reshape(-1, 2).tolist()
                    xy = [tuple(xiyi) for xiyi in xy]
                    PIL.ImageDraw.Draw(mask).polygon(xy=xy, outline=1, fill=1)
                    mask = np.array(mask)
                    label[mask == 1] = ann['category_id']
            else:
                # mask
                if isinstance(ann['segmentation']['counts'], list):
                    rle = pycocotools.mask.frPyObjects(
                        [ann['segmentation']], height, width)
                else:
                    rle = [ann['segmentation']]
                mask = pycocotools.mask.decode(rle)[:, :, 0]
                label[mask == 1] = ann['category_id']
        return label

    def __len__(self):
        return len(self.img_ids)
