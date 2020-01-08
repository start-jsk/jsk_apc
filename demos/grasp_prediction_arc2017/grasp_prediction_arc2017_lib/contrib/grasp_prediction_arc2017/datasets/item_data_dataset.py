import copy
import math
import os.path as osp

import chainer
import cv2
import numpy as np

import grasp_fusion_lib
from grasp_fusion_lib.datasets.apc.arc2017 import class_names_arc2017
from grasp_fusion_lib.datasets.apc.arc2017 import JskARC2017DatasetV3
from grasp_fusion_lib.datasets.apc.arc2017 import load_item_data


def get_shelf_data_hasegawa_iros2018():
    import skimage.io
    path = osp.expanduser('~/data/hasegawa_iros2018/item_data/table/image.jpg')
    if not osp.exists(path):
        grasp_fusion_lib.data.download(
            url='https://drive.google.com/uc?id=15e24rM1ibVO9GTz9VUmzF2abppsudFpU',  # NOQA
            path=path,
            md5='3b32a48aa265e7896417af34a501ec9b',
        )
    img = skimage.io.imread(path)
    lbl = np.ones(img.shape[:2], dtype=np.int32) * 41
    return [img], [lbl]


class ItemDataDataset(chainer.dataset.DatasetMixin):

    def __init__(self,
                 split,
                 item_data_dir,
                 bg_from_dataset_ratio=0.7,
                 project='wada_icra2018',
                 ):
        assert split in ['train', 'valid']
        self.split = split
        assert item_data_dir
        self.project = project
        if self.project == 'wada_icra2018':
            object_names, self.object_data = load_item_data(item_data_dir)
        elif (self.project == 'hasegawa_iros2018' or
              self.project == 'hasegawa_master_thesis'):
            object_names, self.object_data = load_item_data(
                item_data_dir, target_size=360)
        else:
            raise ValueError
        self._bg_from_dataset_ratio = bg_from_dataset_ratio
        # class_names
        self.class_names = ['__background__'] + object_names[:]
        # label conversion for self._dataset
        self.class_id_map = {}
        for cls_id_from, cls_name in enumerate(class_names_arc2017):
            if cls_id_from == 41:
                cls_id_to = 0
            else:
                try:
                    cls_id_to = self.class_names.index(cls_name)
                except ValueError:
                    cls_id_to = -1
            self.class_id_map[cls_id_from] = cls_id_to
        # shelf templates
        self._dataset = JskARC2017DatasetV3(split=self.split)
        if self.project == 'wada_icra2018':
            from grasp_fusion_lib.datasets.apc.arc2017.jsk import (
                get_shelf_data
            )
            get_shelf_data_func = get_shelf_data
        elif (self.project == 'hasegawa_iros2018' or
              self.project == 'hasegawa_master_thesis'):
            get_shelf_data_func = get_shelf_data_hasegawa_iros2018
        self.shelf_img, self.shelf_lbl = get_shelf_data_func()

    def __len__(self):
        return 100  # fixed size

    def get_example(self, i):
        if self.split == 'valid':
            random_state = np.random.RandomState(i)
        else:
            random_state = np.random.RandomState(np.random.randint(0, 10 ** 7))

        if random_state.rand() < self._bg_from_dataset_ratio:
            index_shelf = random_state.randint(0, len(self._dataset))
            img_shelf, lbl_shelf = self._dataset[index_shelf]
        else:
            index_shelf = random_state.randint(0, len(self.shelf_img))
            img_shelf = self.shelf_img[index_shelf]
            lbl_shelf = self.shelf_lbl[index_shelf]

        mask_labeled = lbl_shelf != -1
        x1, y1, x2, y2 = grasp_fusion_lib.image.mask_to_bbox(mask_labeled)
        img_shelf = img_shelf[y1:y2, x1:x2]
        lbl_shelf = lbl_shelf[y1:y2, x1:x2]

        scale = (500. * 500.) / (img_shelf.shape[0] * img_shelf.shape[1])
        scale = math.sqrt(scale)
        img_shelf = cv2.resize(img_shelf, None, None, fx=scale, fy=scale)
        lbl_shelf = cv2.resize(lbl_shelf, None, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_NEAREST)

        # transform class ids (label values)
        img_shelf_converted = img_shelf.copy()
        lbl_shelf_converted = lbl_shelf.copy()
        for cls_id_from, cls_id_to in self.class_id_map.items():
            mask = lbl_shelf == cls_id_from
            if cls_id_to == -1:
                img_shelf_converted[mask] = 0
            cls_id_to = 0 if cls_id_to == -1 else cls_id_to
            lbl_shelf_converted[mask] = cls_id_to
        img_shelf = img_shelf_converted
        lbl_shelf = lbl_shelf_converted
        lbl_shelf[lbl_shelf == 41] = 0  # __shelf__ -> __background__

        object_data = copy.deepcopy(self.object_data)
        random_state.shuffle(object_data)
        if self.project == 'wada_icra2018':
            object_data = grasp_fusion_lib.aug.augment_object_data(
                object_data, random_state)
        elif (self.project == 'hasegawa_iros2018' or
              self.project == 'hasegawa_master_thesis'):
            object_data = grasp_fusion_lib.aug.augment_object_data(
                object_data, random_state, scale=(0.15, 0.8))
        stacked = grasp_fusion_lib.aug.stack_objects(
            img_shelf, lbl_shelf, object_data,
            region_label=0, random_state=random_state)
        if self.project == 'wada_icra2018':
            stacked = next(grasp_fusion_lib.aug.augment_object_data(
                [stacked], random_state, fit_output=False))
        elif (self.project == 'hasegawa_iros2018' or
              self.project == 'hasegawa_master_thesis'):
            stacked = next(grasp_fusion_lib.aug.augment_object_data(
                [stacked], random_state, fit_output=False,
                scale=(0.15, 0.8)))
        return stacked['img'], stacked['lbl'], stacked['lbl_suc']


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-p', '--project',
                        choices=['wada_icra2018',
                                 'hasegawa_iros2018',
                                 'hasegawa_master_thesis'],
                        help='project name')
    args = parser.parse_args()

    from grasp_prediction_arc2017_lib.contrib.grasp_prediction_arc2017 import (
        datasets,
    )
    if args.project == 'wada_icra2018':
        item_data_dir = datasets.item_data.pick_re_experiment()
        bg_from_dataset_ratio = 0.7
    elif args.project == 'hasegawa_iros2018':
        item_data_dir = datasets.item_data.hasegawa_iros2018()
        bg_from_dataset_ratio = 0
    elif args.project == 'hasegawa_master_thesis':
        item_data_dir = datasets.item_data.hasegawa_master_thesis()
        bg_from_dataset_ratio = 0
    else:
        raise ValueError

    dataset = ItemDataDataset(
        split='train',
        item_data_dir=item_data_dir,
        bg_from_dataset_ratio=bg_from_dataset_ratio,
        project=args.project,
    )

    def visualize_func(dataset, index):
        img, lbl, lbl_suc = dataset[index]
        lbl_viz = grasp_fusion_lib.datasets.visualize_label(
            lbl, img, class_names=dataset.class_names)
        lbl_suc_viz = grasp_fusion_lib.datasets.visualize_label(
            lbl_suc, img, class_names=['no_suction', 'suction'])
        return np.hstack((img, lbl_viz, lbl_suc_viz))

    grasp_fusion_lib.datasets.view_dataset(dataset, visualize_func)
