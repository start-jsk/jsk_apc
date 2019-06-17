import copy

import chainer
import numpy as np

import chainer_mask_rcnn
import instance_occlsegm_lib.data

from .load_item_data import load_item_data


class ARC2017ItemDataSyntheticInstanceSegmentationDataset(
    chainer.dataset.DatasetMixin
):

    def __init__(
        self,
        item_data_dir,
        do_aug=False,
        aug_level='all',
        exclude_arc2017=False,
        background='tote',
        stack_ratio=0.9,
    ):
        object_names, object_data = load_item_data(
            item_data_dir, skip_known=exclude_arc2017
        )
        self.class_names = object_names
        self.object_data = object_data
        self._do_aug = do_aug
        self._aug_level = aug_level
        assert background in ['tote', 'shelf', 'tote+shelf']
        self._background = background
        self._stack_ratio = stack_ratio  # ex. 0.9, (0.2, 0.9)

    def __len__(self):
        return int(1e3)

    def get_example(self, i):
        random_state = np.random.RandomState()

        imgs, lbls = instance_occlsegm_lib.datasets.apc.arc2017.\
            jsk.get_shelf_data()
        if self._background == 'tote':
            imgs, lbls = imgs[1:2], lbls[1:2]  # only tote
        elif self._background == 'shelf':
            imgs, lbls = imgs[0:1], lbls[0:1]  # only shelf
        else:
            assert self._background == 'tote+shelf'

        index = random_state.choice(np.arange(len(imgs)))
        img, lbl = imgs[index], lbls[index]
        lbl[lbl == 41] = 0
        if self._do_aug and self._aug_level in ['all', 'image']:
            shelf_datum = dict(img=img, lbl=lbl)
            shelf_datum = next(instance_occlsegm_lib.aug.augment_object_data(
                [shelf_datum], random_state=random_state, fit_output=False))
            img, lbl = shelf_datum['img'], shelf_datum['lbl']

        object_data = copy.deepcopy(self.object_data)
        random_state.shuffle(object_data)
        if self._do_aug and self._aug_level in ['all', 'object']:
            object_data = instance_occlsegm_lib.aug.augment_object_data(
                object_data, random_state=random_state)
        stacked = instance_occlsegm_lib.aug.stack_objects(
            img, lbl, object_data,
            region_label=0, n_objects=(0, 16), return_instances=True,
            stack_ratio=self._stack_ratio, random_state=random_state)
        if self._do_aug and self._aug_level in ['all', 'image']:
            stacked = next(instance_occlsegm_lib.aug.augment_object_data(
                [stacked],
                random_state=random_state,
                fit_output=False,
                scale=(0.5, 2.0),
            ))
            keep = stacked['masks'].sum(axis=(1, 2)) > 0
            stacked['masks'] = stacked['masks'][keep]
            stacked['bboxes'] = instance_occlsegm_lib.image.masks_to_bboxes(
                stacked['masks'])
            stacked['labels'] = stacked['labels'][keep]

        img = stacked['img']
        bboxes = stacked['bboxes']
        assert bboxes.dtype == np.int32
        labels = stacked['labels']
        assert labels.dtype == np.int32
        masks = stacked['masks']

        n_bbox = len(bboxes)
        lbl_ins, lbl_cls = chainer_mask_rcnn.utils.instance_boxes2label(
            labels, bboxes, masks)

        keep = np.ones((n_bbox,), dtype=bool)
        lbls = []
        for i in range(0, n_bbox):
            bbox = bboxes[i]
            mask = masks[i]
            mask_visible = lbl_ins == i
            if mask_visible.sum() == 0:
                keep[i] = False
                continue
            mask_invisible = (~mask_visible) & mask
            # create lbl
            # -1: out of region
            #  0: background
            #  1: visible
            #  2: invisible
            lbl = - np.ones(mask.shape[:2], dtype=np.int32)
            y1, x1, y2, x2 = bbox
            lbl[y1:y2, x1:x2] = 0
            lbl[mask_visible] = 1
            lbl[mask_invisible] = 2
            lbls.append(lbl)
        lbls = np.asarray(lbls, dtype=np.int32)
        bboxes = bboxes[keep]
        labels = labels[keep]

        bboxes = bboxes.astype(np.float32, copy=False)
        labels -= 1

        return img, bboxes, labels, lbls
