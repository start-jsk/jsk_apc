import glob
import os
import os.path as osp
import re

import chainer
import numpy as np
import PIL.Image
import skimage.io
try:
    from sklearn.model_selection import train_test_split
except ImportError:
    from sklearn.cross_validation import train_test_split

from .base import class_names_apc2016
import instance_occlsegm_lib.data
from instance_occlsegm_lib.datasets import config


class JskAPC2016Dataset(chainer.dataset.DatasetMixin):

    class_names = class_names_apc2016
    _root_dir = osp.join(config.ROOT_DIR, 'APC2016')

    def __init__(self, split):
        assert split in ['all', 'train', 'valid']
        self.split = split
        self._init_ids()

    def _init_ids(self):
        ids = []
        # APC2016rbo
        dataset_dir = osp.join(self._root_dir, 'APC2016rbo')
        if not osp.exists(dataset_dir):
            self.download()
        for img_file in os.listdir(dataset_dir):
            if not re.match(r'^.*_[0-9]*_bin_[a-l].jpg$', img_file):
                continue
            data_id = osp.splitext(img_file)[0]
            ids.append(('rbo', data_id))
        # APC2016seg
        dataset_dir = osp.join(self._root_dir, 'annotated')
        if not osp.exists(dataset_dir):
            self.download()
        for scene_dir in os.listdir(dataset_dir):
            if osp.isdir(scene_dir):
                ids.append(('seg', scene_dir))
        ids_train, ids_valid = train_test_split(
            ids, test_size=0.25, random_state=5)
        self._ids = {'all': ids, 'train': ids_train, 'valid': ids_valid}

    def __len__(self):
        return len(self._ids[self.split])

    def download(self):
        instance_occlsegm_lib.data.download(
            url='https://drive.google.com/uc?id=0B9P1L--7Wd2vSV9oLTd1U2I3TDg',
            path=osp.join(self._root_dir, 'APC2016rbo.tgz'),
            md5='efd7f1d5420636ee2b2827e7e0f5d1ac',
            postprocess=instance_occlsegm_lib.data.extractall,
        )
        instance_occlsegm_lib.data.download(
            url='https://drive.google.com/uc?id=0B9P1L--7Wd2vaExFU1AxWHlMdTg',
            path=osp.join(self._root_dir, 'APC2016jsk.tgz'),
            md5='8f1641f52fff90154533f84b9eb111a5',
            postprocess=instance_occlsegm_lib.data.extractall,
        )

    def get_example(self, i):
        data_type, data_id = self._ids[self.split][i]
        if data_type == 'seg':
            dataset_dir = osp.join(self._root_dir, 'annotated')
            img_file = osp.join(dataset_dir, data_id, 'image.png')
            label_file = osp.join(dataset_dir, data_id, 'label.png')
            img = skimage.io.imread(img_file)
            assert img.dtype == np.uint8
            label = np.array(PIL.Image.open(label_file), dtype=np.int32)
            label[label == 255] = -1
        elif data_type == 'rbo':
            dataset_dir = osp.join(self._root_dir, 'APC2016rbo')
            img_file = osp.join(dataset_dir, data_id + '.jpg')
            img = skimage.io.imread(img_file)

            label = np.zeros(img.shape[:2], dtype=np.int32)

            shelf_bin_mask_file = osp.join(dataset_dir, data_id + '.pbm')
            shelf_bin_mask = skimage.io.imread(
                shelf_bin_mask_file, as_gray=True
            )
            label[shelf_bin_mask < 127] = -1

            mask_glob = osp.join(dataset_dir, data_id + '_*.pbm')
            for mask_file in glob.glob(mask_glob):
                mask_id = osp.splitext(osp.basename(mask_file))[0]
                mask = skimage.io.imread(mask_file, as_gray=True)
                label_name = mask_id[len(data_id + '_'):]
                label_value = self.class_names.index(label_name)
                label[mask > 127] = label_value
        else:
            raise ValueError
        return img, label
