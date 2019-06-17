import numpy as np

import instance_occlsegm_lib


def visualize_panoptic_occlusion_segmentation(
    img, bboxes, labels, masks, lbl_vis, lbl_occ, fg_class_names,
):
    viz_ins = instance_occlsegm_lib.datasets.visualize_instance_segmentation(
        img, bboxes, labels, masks, fg_class_names, n_mask_class=3
    )
    class_names = ['__background__'] + list(fg_class_names)
    viz_sem = visualize_occlusion_segmentation(
        img, lbl_vis, lbl_occ, class_names
    )
    viz = instance_occlsegm_lib.image.tile([viz_ins, viz_sem])
    return viz


def view_panoptic_occlusion_segmentation_dataset(data):

    def visualize_func(data, index):
        img, bboxes, labels, masks, lbl_vis, lbl_occ = data[index]
        return visualize_panoptic_occlusion_segmentation(
            img, bboxes, labels, masks, lbl_vis, lbl_occ, data.class_names
        )

    instance_occlsegm_lib.datasets.view_dataset(data, visualize_func)


def visualize_occlusion_segmentation(img, lbl_vis, lbl_occ, class_names):
    n_class = len(class_names)

    # visible
    viz_vis = instance_occlsegm_lib.image.label2rgb(
        lbl_vis, img, label_names=class_names, thresh_suppress=0.01
    )
    viz_vis = instance_occlsegm_lib.image.tile([img, viz_vis])

    # occluded
    vizs = []
    for fg_label in range(n_class - 1):
        prob_vis_l = (lbl_vis == (fg_label + 1)).astype(np.float32)
        ignore = prob_vis_l < 0
        prob_vis_l[ignore] = 0
        viz1 = instance_occlsegm_lib.image.colorize_heatmap(prob_vis_l)
        viz1 = instance_occlsegm_lib.image.overlay_color_on_mono(viz1, img)
        if ignore.sum() > 0:
            viz1[ignore] = np.random.randint(
                0, 255, (ignore.sum(), 3), dtype=np.uint8
            )

        lbl_occ_l = lbl_occ[:, :, fg_label]
        ignore = lbl_occ_l < 0
        lbl_occ_l[ignore] = 0
        viz2 = instance_occlsegm_lib.image.colorize_heatmap(lbl_occ_l)
        viz2 = instance_occlsegm_lib.image.overlay_color_on_mono(viz2, img)
        if ignore.sum() > 0:
            viz2[ignore] = np.random.randint(
                0, 255, (ignore.sum(), 3), dtype=np.uint8
            )

        viz = instance_occlsegm_lib.image.tile([viz1, viz2])
        vizs.append(viz)
    viz_occ = instance_occlsegm_lib.image.tile(vizs, boundary=True)
    viz_occ = instance_occlsegm_lib.image.resize(viz_occ, width=1500)

    return instance_occlsegm_lib.image.tile([viz_vis, viz_occ], shape=(2, 1))


def view_occlusion_segmentation_dataset(data):
    instance_occlsegm_lib.datasets.view_dataset(
        data,
        lambda d, i: visualize_occlusion_segmentation(
            *d[i], class_names=d.class_names
        )
    )
