#!/usr/bin/env python

import argparse
import copy
import json
import os
import os.path as osp

import chainer_mask_rcnn as cmr
import labelme
import numpy as np
import skimage.io
import yaml

import instance_occlsegm_lib


here = osp.dirname(osp.abspath(__file__))


def get_class_names():
    class_names = []
    with open(osp.join(here, 'labelmerc')) as f:
        data = yaml.load(f)
        for l in data['labels']:
            class_id, class_name = l.split(': ')
            class_id = int(class_id)
            if class_name == '__ignore__':
                continue
            assert len(class_names) == class_id
            class_names.append(class_name)
    return class_names


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--view', action='store_true', help='view')
    args = parser.parse_args()

    class_names = get_class_names()

    annotated_dir = osp.join(here, 'data/annotation_raw_data/20180730')

    if not args.view:
        dataset_dir = osp.join(here, 'data/dataset_data/20180730')
        os.makedirs(dataset_dir)
        with open(osp.join(dataset_dir, 'class_names.txt'), 'w') as f:
            for class_name in class_names:
                f.write(class_name + '\n')

    for video_dir in sorted(os.listdir(annotated_dir)):
        video_dir = osp.join(annotated_dir, video_dir)

        objects = []
        for frame_dir in sorted(os.listdir(video_dir), reverse=True):
            frame_dir = osp.join(video_dir, frame_dir)
            print('-' * 79)
            print('frame_dir: ', frame_dir)

            anno_file = osp.join(frame_dir, 'image.json')
            with open(anno_file) as f:
                data = json.load(f)

            if data['flags']['skip']:
                print('skipping..')
                continue

            img_file = osp.join(osp.dirname(anno_file), data['imagePath'])
            img = skimage.io.imread(img_file)

            labels = []
            masks = []
            bboxes = []
            objects_old = copy.deepcopy(objects)
            objects_new = []
            for shape in data['shapes']:
                label_i = int(shape['label'].split(': ')[0])
                if label_i == 0:
                    continue

                # object whole region is visible:
                # vi (visible) = vc = vi + oc (occluded)
                mask_vi_i = labelme.utils.polygons_to_mask(
                    img.shape, shape['points'],
                )
                # 0: background
                # 1: visible
                # 2: occluded
                mask_i = np.zeros((img.shape[0], img.shape[1]), dtype=np.int32)
                mask_i[mask_vi_i] = 1
                labels.append(label_i)
                masks.append(mask_i)
                bboxes.append(cmr.utils.mask_to_bbox(mask_i))

                objects_old_ = []
                for label_j, mask_j in objects_old:
                    mask_oc_j = np.bitwise_or(
                        np.bitwise_and(mask_vi_i, np.isin(mask_j, [1, 2])),
                        mask_j == 2,
                    )
                    mask_vi_j = np.bitwise_and(~mask_oc_j, mask_j == 1)
                    mask_j[...] = 0
                    mask_j[mask_vi_j] = 1
                    mask_j[mask_oc_j] = 2
                    objects_old_.append((label_j, mask_j))
                objects_old = objects_old_

                objects_new.append((label_i, mask_i))

            objects = []
            for label_j, mask_j in objects_old:
                labels.append(label_j)
                masks.append(mask_j)
                bboxes.append(cmr.utils.mask_to_bbox(mask_j))
                objects.append((label_j, mask_j))
            objects.extend(objects_new)

            labels = np.asarray(labels, dtype=np.int32)
            masks = np.asarray(masks, dtype=np.int32)
            bboxes = np.asarray(bboxes, dtype=np.float32)

            N = len(labels)
            H, W = img.shape[:2]
            assert labels.shape == (N,)
            assert labels.dtype == np.int32
            assert masks.shape == (N, H, W), masks.shape
            assert masks.dtype == np.int32
            assert bboxes.shape == (N, 4), bboxes.shape
            assert bboxes.dtype == np.float32

            captions = ['{:2}: {:s}'.format(l, class_names[l]) for l in labels]
            viz_vis = cmr.utils.draw_instance_bboxes(
                img,
                bboxes,
                labels + 1,
                n_class=len(class_names) + 1,
                masks=masks == 1,
                captions=captions,
                bg_class=0,
            )
            viz_occ = cmr.utils.draw_instance_bboxes(
                img,
                bboxes,
                labels + 1,
                n_class=len(class_names) + 1,
                masks=masks == 2,
                captions=captions,
                bg_class=0,
            )
            viz = np.hstack((img, viz_vis, viz_occ))

            if args.view:
                for i in range(len(bboxes)):
                    print(captions[i])
                    draw = np.zeros((len(bboxes),), dtype=bool)
                    draw[i] = True
                    viz_vi = cmr.utils.draw_instance_bboxes(
                        img,
                        bboxes,
                        labels,
                        n_class=len(class_names),
                        masks=masks == 1,
                        captions=captions,
                        bg_class=0,
                        draw=draw,
                    )
                    viz_oc = cmr.utils.draw_instance_bboxes(
                        img,
                        bboxes,
                        labels,
                        n_class=len(class_names),
                        masks=masks == 2,
                        captions=captions,
                        bg_class=0,
                        draw=draw,
                    )
                    viz = instance_occlsegm_lib.image.tile(
                        [img, viz_vi, viz_oc])
                    viz = instance_occlsegm_lib.image.resize(viz, width=1500)
                    instance_occlsegm_lib.io.imshow(viz)
                    if instance_occlsegm_lib.io.waitkey() == ord('q'):
                        return
            else:
                out_video_dir = osp.join(dataset_dir, osp.basename(video_dir))
                if not osp.exists(out_video_dir):
                    os.makedirs(out_video_dir)

                frame_name = osp.basename(frame_dir)
                img_file = osp.join(out_video_dir, frame_name + '.jpg')
                skimage.io.imsave(img_file, img)
                print('Saved to: {}'.format(img_file))
                npz_file = osp.join(out_video_dir, frame_name + '.npz')
                np.savez_compressed(
                    npz_file, bboxes=bboxes, labels=labels, masks=masks
                )
                print('Saved to: {}'.format(npz_file))


if __name__ == '__main__':
    main()
