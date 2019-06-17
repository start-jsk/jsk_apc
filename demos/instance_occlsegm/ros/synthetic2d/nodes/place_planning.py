#!/usr/bin/env python
# flake8: noqa

import numpy as np


def get_place_mask(img, bboxes, labels, masks, obstacles, target_id, pick, debug=0, n_times=10, mask_fg=None):
    import chainer_mask_rcnn as cmr
    import instance_occlsegm_lib
    from skimage.segmentation import slic
    from scipy.ndimage import center_of_mass
    from skimage.transform import rotate
    from skimage.transform import rescale

    lbl_ins, _ = cmr.utils.instance_boxes2label(labels + 1, bboxes, masks=np.isin(masks, (1,)))
    lbl_ins2, _ = cmr.utils.instance_boxes2label(
        labels[np.arange(len(labels)) != pick] + 1,
        bboxes[np.arange(len(labels)) != pick],
        masks=np.isin(masks, (1, 2)))

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
            mask_obj_r0 = rotate(mask_obj, resize=True, angle=r * (360. / R), order=0)
            # mask_obj_r1 = rescale(mask_obj_r0, 1.5, mode='constant', multichannel=False, anti_aliasing=False)
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
                dymin = 0 - cy_o
                dxmax = mask_obj_r.shape[1] - cx_o
                dxmin = 0 - cx_o
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


def main():
    data = np.load('book_and_tennis_ball.npz')
    img = data['img']
    bboxes = data['bboxes']
    labels = data['labels']
    masks = data['masks']
    objects_in_graph = [34, 7]
    target = 34

    get_place_mask(img, bboxes, labels, masks, objects_in_graph, target, debug=1)


if __name__ == '__main__':
    main()

# def apply_pca(mask):
#     from sklearn.decomposition import PCA
#
#     pca = PCA()
#     xy = np.argwhere(mask)
#     pca.fit(xy)
#     xy_trans = pca.fit_transform(xy)
#     cy, cx = pca.mean_
#
#     axis0_min = xy_trans[:, 0].min()
#     axis0_max = xy_trans[:, 0].max()
#     axis1_min = xy_trans[:, 1].min()
#     axis1_max = xy_trans[:, 1].max()
#
#     yx0_max = axis0_max * pca.components_[0] + pca.mean_
#     yx0_min = axis0_min * pca.components_[0] + pca.mean_
#     yx1_max = axis1_max * pca.components_[1] + pca.mean_
#     yx1_min = axis1_min * pca.components_[1] + pca.mean_
#
#     # visualize
#     viz = img.copy()
#     cv2.circle(viz, (int(cx), int(cy)), radius=5, color=(0, 255, 0), thickness=-1)
#     # long axis
#     cv2.line(viz, (int(yx0_min[1]), int(yx0_min[0])), (int(yx0_max[1]), int(yx0_max[0])), color=(0, 255, 0), thickness=1)
#     cv2.circle(viz, (int(yx0_max[1]), int(yx0_max[0])), radius=5, color=(0, 0, 255), thickness=-1)
#     cv2.circle(viz, (int(yx0_min[1]), int(yx0_min[0])), radius=5, color=(255, 0, 0), thickness=-1)
#     # short axis
#     cv2.line(viz, (int(yx1_min[1]), int(yx1_min[0])), (int(yx1_max[1]), int(yx1_max[0])), color=(0, 255, 0), thickness=1)
#     cv2.circle(viz, (int(yx1_max[1]), int(yx1_max[0])), radius=5, color=(0, 0, 255), thickness=-1)
#     cv2.circle(viz, (int(yx1_min[1]), int(yx1_min[0])), radius=5, color=(255, 0, 0), thickness=-1)
#     plt.imshow(viz)
#     plt.show()
#
#     viz = img.copy()
#     mask_flat = mask.flatten()
#     index = np.random.choice(np.argwhere(~mask_flat)[:, 0])
#     x = index % mask.shape[1]
#     y = index // mask.shape[1]
#     cv2.circle(viz, (x, y), radius=5, color=(0, 255, 0), thickness=-1)
#     plt.imshow(viz)
#     plt.show()
