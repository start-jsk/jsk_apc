#!/usr/bin/env python

import glob

import numpy as np

import chainer_mask_rcnn as mrcnn
import instance_occlsegm_lib

import contrib


def main():
    class_names = contrib.core.get_class_names()

    # get lbl_clss from sequential frames
    imgs = []
    lbl_clss = []
    class_ids_timeline = []
    whole_masks = {}
    for json_file in sorted(glob.glob('*.json')):
        img, lbl_ins, lbl_cls = contrib.core.load_json_file(json_file)

        class_ids_ = None
        lbl_cls_ = None
        if class_ids_timeline:
            class_ids_ = class_ids_timeline[-1]
        if lbl_clss:
            lbl_cls_ = lbl_clss[-1]

        class_ids = np.unique(lbl_cls)
        class_ids = class_ids[class_ids != 0]
        class_ids_timeline.append(class_ids)

        if class_ids_ is not None:
            picked_class_ids = class_ids_[~np.isin(class_ids_, class_ids)]
            for class_id in picked_class_ids:
                mask_whole = lbl_cls_ == class_id
                assert class_id not in whole_masks
                whole_masks[class_id] = mask_whole

        imgs.append(img)
        lbl_clss.append(lbl_cls)

    # left objects
    for class_id in class_ids:
        mask_whole = lbl_cls == class_id
        whole_masks[class_id] = mask_whole

    # visualize
    for img, lbl_cls in zip(imgs, lbl_clss):
        bboxes = []
        labels = []
        masks = []
        for class_id in np.unique(lbl_cls):
            if class_id in [-1, 0]:
                continue
            assert class_id in whole_masks
            mask_whole = whole_masks[class_id]
            mask_visible = lbl_cls == class_id
            if mask_visible.sum() == 0:
                continue
            x1, y1, x2, y2 = instance_occlsegm_lib.image.mask_to_bbox(
                mask_whole)
            mask_invisible = np.bitwise_and(mask_whole, ~mask_visible)
            bboxes.append((y1, x1, y2, x2))
            labels.append(class_id)
            mask = np.zeros(mask_whole.shape[:2], dtype=np.int32)
            # mask[mask_bg] = 0
            mask[mask_visible] = 1
            mask[mask_invisible] = 2
            masks.append(mask)
        bboxes = np.asarray(bboxes, dtype=np.int32)
        labels = np.asarray(labels, dtype=np.int32)
        masks = np.asarray(masks, dtype=np.int32)

        captions = [class_names[l] for l in labels]
        viz1 = mrcnn.utils.draw_instance_bboxes(
            img, bboxes, labels, n_class=41, masks=masks == 1,
            captions=captions)
        viz2 = mrcnn.utils.draw_instance_bboxes(
            img, bboxes, labels, n_class=41, masks=masks == 2,
            captions=captions)

        viz = instance_occlsegm_lib.image.tile([img, viz1, viz2])
        instance_occlsegm_lib.io.imshow(viz)
        if instance_occlsegm_lib.io.waitkey() == ord('q'):
            break


if __name__ == '__main__':
    main()
