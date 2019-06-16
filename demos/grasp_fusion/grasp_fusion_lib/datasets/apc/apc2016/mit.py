import itertools
import os
import os.path as osp

import chainer
import numpy as np
import skimage.io
try:
    from sklearn.model_selection import train_test_split
except ImportError:
    from sklearn.cross_validation import train_test_split

from .base import class_names_apc2016
import grasp_fusion_lib.data
import grasp_fusion_lib.image


def ids_from_scene_dir(scene_dir, empty_scene_dir):
    for i_frame in itertools.count():
        empty_file = osp.join(
            empty_scene_dir, 'frame-{:06}.color.png'.format(i_frame))
        rgb_file = osp.join(
            scene_dir, 'frame-{:06}.color.png'.format(i_frame))
        segm_file = osp.join(
            scene_dir, 'segm/frame-{:06}.segm.png'.format(i_frame))
        if not (osp.exists(rgb_file) and osp.exists(segm_file)):
            break
        data_id = (empty_file, rgb_file, segm_file)
        yield data_id


def bin_id_from_scene_dir(scene_dir):
    caminfo = open(osp.join(scene_dir, 'cam.info.txt')).read()
    loc = caminfo.splitlines()[0].split(': ')[-1]
    if loc == 'shelf':
        bin_id = caminfo.splitlines()[1][-1]
    else:
        bin_id = 'tote'
    return bin_id


class MitAPC2016Dataset(chainer.dataset.DatasetMixin):

    class_names = class_names_apc2016
    datasets_dir = osp.expanduser('~/data/datasets/APC2016')

    def __init__(self, split, locations=('shelf', 'tote')):
        assert split in ['all', 'train', 'valid']
        self.split = split
        assert all(loc in ['shelf', 'tote'] for loc in locations)
        self._locations = locations
        self.dataset_dir = osp.join(self.datasets_dir, 'benchmark')
        if not osp.exists(self.dataset_dir):
            self.download()
        self._init_ids()

    def __len__(self):
        return len(self._ids[self.split])

    def _init_ids(self):
        data_ids = []
        # office
        contain_dir = osp.join(self.dataset_dir, 'office/test')
        for loc in self._locations:
            loc_dir = osp.join(contain_dir, loc)
            data_ids += self._get_ids_from_loc_dir('office', loc_dir)
        # warehouse
        contain_dir = osp.join(self.dataset_dir, 'warehouse')
        for sub in ['practice', 'competition']:
            sub_contain_dir = osp.join(contain_dir, sub)
            for loc in self._locations:
                loc_dir = osp.join(sub_contain_dir, loc)
                data_ids += self._get_ids_from_loc_dir('warehouse', loc_dir)
        ids_train, ids_valid = train_test_split(
            data_ids, test_size=0.25, random_state=5)
        self._ids = {'all': data_ids, 'train': ids_train, 'valid': ids_valid}

    def _get_ids_from_loc_dir(self, env, loc_dir):
        assert env in ('office', 'warehouse')
        loc = osp.basename(loc_dir)
        data_ids = []
        for scene_dir in os.listdir(loc_dir):
            scene_dir = osp.join(loc_dir, scene_dir)
            bin_id = bin_id_from_scene_dir(scene_dir)
            empty_dir = osp.join(
                self.dataset_dir, env, 'empty', loc, 'scene-{}'.format(bin_id))
            data_ids += list(ids_from_scene_dir(scene_dir, empty_dir))
        return data_ids

    def _load_from_id(self, data_id):
        empty_file, rgb_file, segm_file = data_id
        img = skimage.io.imread(rgb_file)
        img_empty = skimage.io.imread(empty_file)
        # Label value is multiplied by 9:
        #   ex) 0: 0/6=0 (background), 54: 54/6=9 (dasani_bottle_water)
        lbl = skimage.io.imread(segm_file, as_gray=True) / 6
        lbl = lbl.astype(np.int32)
        # infer bin mask
        mask_fg = lbl > 0
        if mask_fg.sum() == 0:
            lbl[...] = -1
        else:
            y1, x1, y2, x2 = grasp_fusion_lib.image.masks_to_bboxes([mask_fg])[
                0]
            mask_bin = np.zeros_like(mask_fg)
            mask_bin[y1:y2, x1:x2] = True
            lbl[~mask_bin] = -1
        # copy object region in rgb image
        img_empty[mask_fg] = img[mask_fg]
        return img_empty, lbl

    def __getitem__(self, i):
        data_id = self._ids[self.split][i]
        img, lbl = self._load_from_id(data_id)
        return img, lbl

    def download(self):
        # XXX: this is optional
        # path = osp.join(self.datasets_dir, 'APC2016mit_training.zip')
        # fcn.data.cached_download(
        #     url='https://drive.google.com/uc?id=0B4mCa-2YGnp7ZEMwcW5rcVBpeG8',  # NOQA
        #     path=path,
        # )
        grasp_fusion_lib.data.download(
            url='https://drive.google.com/uc?id=0B9P1L--7Wd2vZHlSQjJSV0x4eXc',
            path=osp.join(self.datasets_dir, 'APC2016mit_benchmark.zip'),
            md5='bdb56b152a7cec0e652898338e519e79',
            postprocess=grasp_fusion_lib.data.extractall,
        )
