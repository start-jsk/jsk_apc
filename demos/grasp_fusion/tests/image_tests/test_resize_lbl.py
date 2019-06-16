import fcn

import grasp_fusion_lib


def test_resize_lbl():
    _, lbl, _ = grasp_fusion_lib.data.voc_image()

    H, W = lbl.shape[:2]
    size = 50 * 50

    # Naive label resize
    lbl_a = grasp_fusion_lib.image.resize(lbl, size=size, interpolation=0)
    lbl_a = grasp_fusion_lib.image.resize(
        lbl_a, height=H, width=W, interpolation=0)

    # Channel space label resize
    lbl_b = grasp_fusion_lib.image.resize_lbl(lbl, size=size)
    lbl_b = grasp_fusion_lib.image.resize_lbl(lbl_b, height=H, width=W)

    lbl_c = lbl.copy()

    # Visualization
    viz_a = grasp_fusion_lib.image.label2rgb(lbl_a)
    viz_b = grasp_fusion_lib.image.label2rgb(lbl_b)
    viz_c = grasp_fusion_lib.image.label2rgb(lbl_c)
    viz = grasp_fusion_lib.image.tile([viz_a, viz_b, viz_c])
    grasp_fusion_lib.io.imsave('/tmp/viz.jpg', viz)

    # Test that channel space label resize is better than naive one
    lbl[lbl == 255] = -1
    lbl_a[lbl_a == 255] = 0
    lbl_b[lbl_b == 255] = 0
    lbl_c[lbl_c == 255] = 0
    acc_a = fcn.utils.label_accuracy_score(lbl, lbl_a, n_class=21)
    acc_b = fcn.utils.label_accuracy_score(lbl, lbl_b, n_class=21)
    acc_c = fcn.utils.label_accuracy_score(lbl, lbl_c, n_class=21)
    for metric_i in zip(acc_a, acc_b, acc_c):
        assert metric_i[0] < metric_i[1] < metric_i[2]
