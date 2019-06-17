import chainer_mask_rcnn as cmr
import numpy as np
from scipy.ndimage import center_of_mass
from skimage.segmentation import slic
from skimage.transform import rescale
from skimage.transform import rotate

import instance_occlsegm_lib


def find_space(
    img,
    bboxes,
    labels,
    masks,
    obstacles,
    target_id,
    pick,
    n_times=10,
    mask_fg=None,
    debug=0,
):
    lbl_ins, _ = cmr.utils.instance_boxes2label(
        labels + 1, bboxes, masks=np.isin(masks, (1,))
    )
    lbl_ins2, _ = cmr.utils.instance_boxes2label(
        labels[np.arange(len(labels)) != pick] + 1,
        bboxes[np.arange(len(labels)) != pick],
        masks=np.isin(masks, (1, 2)),
    )

    # objects_in_graph = [o + 1 for o in objects_in_graph]
    mask = lbl_ins == pick
    cy, cx = center_of_mass(mask)
    xmin, ymin, xmax, ymax = instance_occlsegm_lib.image.mask_to_bbox(mask)
    mask_obj = mask[ymin:ymax, xmin:xmax]
    # plt.imshow(mask_obj, cmap='gray')
    # plt.show()
    segments = slic(lbl_ins, n_segments=50)
    mask_placable = lbl_ins == -1
    if mask_fg is not None:
        mask_placable = np.bitwise_and(mask_placable, mask_fg)
    # plt.imshow(lbl_cls2)
    # plt.imshow(mask_placable)
    # plt.show()
    disabled = np.unique(segments[~mask_placable])
    for s in disabled:
        segments[segments == s] = -1
    # plt.imshow(segments)
    # plt.imshow(mask_placable, cmap='gray')
    # plt.show()
    distances = []
    for s in np.unique(segments):
        if s == -1:
            continue
        mask_s = segments == s
        cy_s, cx_s = center_of_mass(mask_s)
        d = np.sqrt((cx - cx_s) ** 2 + (cy - cy_s) ** 2)
        distances.append((s, d))
    distances = sorted(distances, key=lambda x: x[1])
    for l, d in distances[:n_times]:
        R = 8
        for r in range(0, R):
            mask_obj_r0 = rotate(
                mask_obj, resize=True, angle=r * (360. / R), order=0
            )
            #  mask_obj_r1 = rescale(
            #     mask_obj_r0,
            #     1.5,
            #     mode='constant',
            #     multichannel=False,
            #     anti_aliasing=False,
            # )
            mask_obj_r1 = rescale(mask_obj_r0, 1.1, mode='constant')
            mask_obj_r1 = mask_obj_r1 >= 0.5

            # plt.subplot(121)
            # plt.imshow(mask_obj_r0)
            # plt.subplot(122)
            # plt.imshow(mask_obj_r1)
            # plt.show()

            H, W = mask.shape[:2]

            mask_s = segments == l
            cy_s, cx_s = center_of_mass(mask_s)

            def get_mask_t(mask_obj_r):
                h, w = mask_obj_r.shape[:2]
                cy_o, cx_o = center_of_mass(mask_obj_r)
                dymax = mask_obj_r.shape[0] - cy_o
                # dymin = 0 - cy_o
                dxmax = mask_obj_r.shape[1] - cx_o
                # dxmin = 0 - cx_o
                ymax_t = int(cy_s + dymax)
                ymin_t = ymax_t - h
                # ymin_t = int(cy_s + dymin)
                xmax_t = int(cx_s + dxmax)
                xmin_t = xmax_t - w
                # xmin_t = int(cx_s + dxmin)
                if not (0 <= ymax_t <= H and 0 <= ymin_t <= H and
                        0 <= xmax_t <= W and 0 <= xmin_t <= W):
                    return None
                mask_t = np.zeros_like(mask)
                mask_t[ymin_t:ymax_t, xmin_t:xmax_t] = mask_obj_r
                return mask_t

            mask_t1 = get_mask_t(mask_obj_r1)
            mask_t0 = get_mask_t(mask_obj_r0)

            if mask_t0 is None or mask_t1 is None:
                continue

            # instance_occlsegm_lib.io.tileimg([
            #     mask_t1,
            #     mask_placable,
            #     np.bitwise_or(mask_t1, mask_placable),
            #     np.bitwise_and(mask_t1, ~mask_placable)
            # ])
            # instance_occlsegm_lib.io.show()
            if 1. * np.sum(mask_t1 & ~mask_placable) / mask_t1.sum() < 0.05:
                if debug:
                    instance_occlsegm_lib.io.tileimg([
                        img,
                        mask,
                        mask_placable,
                        np.bitwise_or(mask_t1, ~mask_placable),
                        np.bitwise_or(mask_t0, ~mask_placable),
                        mask_t0
                    ])
                    instance_occlsegm_lib.io.show()
                return mask_t0
            # plt.imshow(segments)
            # plt.plot([cx_s], [cy_s], 'o', color='r')
            # plt.show()
