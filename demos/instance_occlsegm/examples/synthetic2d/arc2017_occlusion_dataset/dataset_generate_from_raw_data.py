#!/usr/bin/env python

import glob
import os
import os.path as osp

import numpy as np
import skimage.io
import tqdm

import instance_occlsegm_lib

import contrib


here = osp.dirname(osp.abspath(__file__))


def main():
    raw_data_dir = osp.join(here, 'annotation_raw_data/20180204_annotated')
    dataset_dir = osp.join(here, 'dataset_data/20180204')

    if osp.exists(dataset_dir):
        print('Dataset dir already exists: %s' % dataset_dir)
        return
    os.makedirs(dataset_dir)

    class_names = contrib.core.get_class_names()
    with open(osp.join(dataset_dir, 'class_names.txt'), 'w') as f:
        for class_name in class_names:
            f.write('%s\n' % class_name)

    for video_id in tqdm.tqdm(sorted(os.listdir(raw_data_dir))):
        if video_id == '__invalid__':
            continue

        video_dir = osp.join(raw_data_dir, video_id)
        json_files = sorted(glob.glob(osp.join(video_dir, '*.json')))

        # get imgs, lbl_clss, whole_masks  from sequential frames
        imgs = []
        lbl_clss = []
        class_ids_timeline = []
        whole_masks = {}
        for json_file in json_files:
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
                # picked object should be unoccluded
                for class_id in picked_class_ids:
                    mask_whole = lbl_cls_ == class_id
                    assert class_id not in whole_masks
                    whole_masks[class_id] = mask_whole

            imgs.append(img)
            lbl_clss.append(lbl_cls)

        # left objects should be unoccluded
        for class_id in class_ids:
            mask_whole = lbl_cls == class_id
            whole_masks[class_id] = mask_whole

        for frame_id, (img, lbl_cls) in enumerate(zip(imgs, lbl_clss)):
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

                # 0: bg, 1: visible, 2: invisible
                mask = np.zeros(mask_whole.shape[:2], dtype=np.int32)
                # mask[mask_bg] = 0
                mask[mask_visible] = 1
                mask[mask_invisible] = 2
                masks.append(mask)
            bboxes = np.asarray(bboxes, dtype=np.int32)
            labels = np.asarray(labels, dtype=np.int32)
            masks = np.asarray(masks, dtype=np.int32)

            out_video_dir = osp.join(dataset_dir, video_id)
            if not osp.exists(out_video_dir):
                os.makedirs(out_video_dir)
            skimage.io.imsave(
                osp.join(out_video_dir, '%08d.jpg' % frame_id), img)
            np.savez_compressed(
                osp.join(out_video_dir, '%08d.npz' % frame_id),
                img=img, lbl_cls=lbl_cls,
                bboxes=bboxes, labels=labels, masks=masks)


if __name__ == '__main__':
    main()
