#!/usr/bin/env python

import chainer_mask_rcnn
import instance_occlsegm_lib

import contrib


if __name__ == '__main__':
    dataset = contrib.datasets.ARC2017SyntheticInstancesDataset(do_aug=True)

    def visualize_func(dataset, index):
        img, bboxes, labels, lbls = dataset[index]
        class_names = dataset.class_names
        captions = [class_names[l] for l in labels]

        vizs = []
        for bbox, label, lbl, caption in \
                zip(bboxes, labels, lbls, captions):
            mask_visible = lbl == 1
            mask_invisible = lbl == 2
            viz = chainer_mask_rcnn.utils.draw_instance_bboxes(
                img, [bbox], [label], n_class=len(class_names),
                masks=[mask_visible], captions=[caption])
            vizs.append(viz)
            viz = chainer_mask_rcnn.utils.draw_instance_bboxes(
                img, [bbox], [label], n_class=len(class_names),
                masks=[mask_invisible], captions=[caption])
            vizs.append(viz)
        viz = instance_occlsegm_lib.image.tile(vizs, (len(vizs) // 8, 8))
        return viz

    instance_occlsegm_lib.datasets.view_dataset(dataset, visualize_func)
    # viz = instance_occlsegm_lib.image.resize(viz, size=1000 * 1000)
    # instance_occlsegm_lib.io.imshow(viz)
    # instance_occlsegm_lib.io.waitkey()
