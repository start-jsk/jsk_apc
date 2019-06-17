#!/usr/bin/env python

import glob

import numpy as np

import instance_occlsegm_lib

import chainer_mask_rcnn as mrcnn

import contrib


def main():
    class_names = contrib.core.get_class_names()

    bboxes_, labels_, masks_ = None, None, None
    json_files = sorted(glob.glob('*.json'))
    for json_file in json_files:
        img, lbl_ins, lbl_cls = contrib.core.load_json_file(json_file)

        labels, bboxes, masks = mrcnn.utils.label2instance_boxes(
            lbl_ins, lbl_cls, return_masks=True)

        if bboxes_ is None:
            masks_inv = None
        else:
            masks_inv = []
            for i in range(len(bboxes)):
                mask = masks[i]
                label = labels[i]

                mask_inv = np.zeros_like(mask)
                for j in range(len(bboxes_)):
                    label_ = labels_[j]
                    mask_ = masks_[j]
                    if label_ == label:
                        continue
                    mask_inv = np.bitwise_or(
                        mask_inv,
                        np.bitwise_and(mask, mask_),  # invisible region
                    )
                masks_inv.append(mask_inv)

        captions = [class_names[l] for l in labels]
        viz1 = mrcnn.utils.draw_instance_bboxes(
            img, bboxes, labels, n_class=len(class_names),
            masks=masks, captions=captions)
        viz2 = mrcnn.utils.draw_instance_bboxes(
            img, bboxes, labels, n_class=len(class_names),
            masks=masks_inv, captions=captions)
        tile = instance_occlsegm_lib.image.tile([img, viz1, viz2])
        instance_occlsegm_lib.io.imshow(tile)
        key = instance_occlsegm_lib.io.waitkey()
        if key == ord('q'):
            break

        bboxes_, masks_, labels_ = bboxes, masks, labels


if __name__ == '__main__':
    main()
