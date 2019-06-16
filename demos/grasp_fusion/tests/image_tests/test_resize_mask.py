import fcn
import numpy as np

import grasp_fusion_lib


def test_resize_mask():
    _, lbl, _ = grasp_fusion_lib.data.voc_image()
    mask = lbl == 1  # airplane

    H, W = mask.shape[:2]
    size = 50 * 50

    # Naive label resize
    mask_a = grasp_fusion_lib.image.resize(
        mask.astype(int), size=size, interpolation=0,
    ).astype(bool)
    mask_a = grasp_fusion_lib.image.resize(
        mask_a.astype(int), height=H, width=W, interpolation=0,
    ).astype(bool)

    # Channel space label resize
    mask_b = grasp_fusion_lib.image.resize_mask(mask, size=size)
    mask_b = grasp_fusion_lib.image.resize_mask(mask_b, height=H, width=W)

    mask_c = mask.copy()

    mask = mask.astype(np.int32)
    mask_a = mask_a.astype(np.int32)
    mask_b = mask_b.astype(np.int32)
    mask_c = mask_c.astype(np.int32)

    # Visualization
    viz_a = grasp_fusion_lib.image.label2rgb(mask_a)
    viz_b = grasp_fusion_lib.image.label2rgb(mask_b)
    viz_c = grasp_fusion_lib.image.label2rgb(mask_c)
    viz = grasp_fusion_lib.image.tile([viz_a, viz_b, viz_c])
    grasp_fusion_lib.io.imsave('/tmp/viz.jpg', viz)

    # Test that channel space label resize is better than naive one
    acc_a = fcn.utils.label_accuracy_score(mask, mask_a, n_class=2)
    acc_b = fcn.utils.label_accuracy_score(mask, mask_b, n_class=2)
    acc_c = fcn.utils.label_accuracy_score(mask, mask_c, n_class=2)
    for metric_i in zip(acc_a, acc_b, acc_c):
        assert metric_i[0] < metric_i[1] < metric_i[2]
