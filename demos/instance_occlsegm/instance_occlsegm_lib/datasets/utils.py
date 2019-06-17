import cv2
import numpy as np

import chainer_mask_rcnn

from .. import image as image_module


def visualize_label(lbl, img, class_names=None):
    lbl_viz1 = image_module.label2rgb(
        lbl, label_names=class_names, thresh_suppress=0.01)
    lbl_viz2 = image_module.label2rgb(
        lbl, img, label_names=class_names, thresh_suppress=0.01)
    return image_module.tile([img, lbl_viz1, lbl_viz2], (1, 3))


def visualize_heatmap(hmp, img, lbl=None):
    hmp_viz1 = image_module.colorize_heatmap(hmp)
    hmp_viz2 = image_module.overlay_color_on_mono(hmp_viz1, img, alpha=.5)
    if lbl is not None:
        mask_unlabeled = lbl == -1
        viz_unlabeled = np.random.random(size=(mask_unlabeled.sum(), 3)) * 255
        hmp_viz1[mask_unlabeled] = viz_unlabeled
        hmp_viz2[mask_unlabeled] = viz_unlabeled
    return image_module.tile([img, hmp_viz1, hmp_viz2], (1, 3))


def visualize_segmentation(img, lbl_true, lbl_pred, class_names=None):
    lbl_pred = lbl_pred.copy()
    lbl_pred[lbl_true == -1] = -1
    viz_true = visualize_label(lbl_true, img, class_names)
    viz_pred = visualize_label(lbl_pred, img, class_names)
    viz = image_module.tile([viz_true, viz_pred], shape=(2, 1))
    return viz


def view_dataset(dataset, visualize_func=None):
    try:
        assert dataset._transform is not True
    except AttributeError:
        pass

    split = getattr(dataset, 'split', '<unknown>')
    print("Showing dataset '%s' (%d) with split '%s'." %
          (dataset.__class__.__name__, len(dataset), split))

    index = 0
    speed = 1
    while True:
        if visualize_func is None:
            viz = dataset[index]
        else:
            viz = visualize_func(dataset, index)
        if viz is None:
            index += speed
            continue
        cv2.imshow(dataset.__class__.__name__, viz[:, :, ::-1])

        k = cv2.waitKey(0)
        if k == ord('q'):
            break
        elif k == ord('n'):
            if index == len(dataset) - 1:
                print('WARNING: reached edge index of dataset: %d' % index)
                continue
            speed = 1
        elif k == ord('p'):
            if index == 0:
                print('WARNING: reached edge index of dataset: %d' % index)
                continue
            speed = -1
        else:
            continue
        index += speed


def view_class_seg_dataset(dataset):

    def visualize_func(dataset, index):
        img, lbl = dataset[index]
        print('[{:08d}] Labels: {}'.format(index, np.unique(lbl)))
        return visualize_label(lbl, img, class_names=dataset.class_names)

    return view_dataset(dataset, visualize_func)


def visualize_instance_segmentation(
    img, bboxes, labels, lbls, class_names, n_mask_class=2,
):
    captions = [class_names[l] for l in labels]

    n_bbox = len(bboxes)

    vizs = []
    for c in range(n_mask_class):
        viz = chainer_mask_rcnn.utils.draw_instance_bboxes(
            img, bboxes, labels + 1, n_class=len(class_names) + 1,
            masks=lbls == c, captions=captions)
        vizs.append(viz)
    viz1 = image_module.tile(vizs, (1, n_mask_class), boundary=True)
    viz1 = image_module.resize(viz1, width=80 * 15)

    vizs = []
    for i in range(n_bbox):
        draw = [False] * n_bbox
        draw[i] = True
        y1, x1, y2, x2 = bboxes[i].astype(int)

        viz = []
        for c in range(n_mask_class):
            viz_c = chainer_mask_rcnn.utils.draw_instance_bboxes(
                img, bboxes, labels + 1, n_class=len(class_names) + 1,
                masks=lbls == c, captions=captions, draw=draw)
            viz.append(viz_c[y1:y2, x1:x2])
        viz = image_module.tile(viz, (n_mask_class, 1))
        viz = image_module.centerize(viz, (300, 80))
        vizs.append(viz)
    if len(vizs) > 0:
        viz2 = image_module.tile(
            vizs, (1, 15), boundary=True, boundary_thickness=1
        )
        viz2 = image_module.resize(viz2, width=80 * 15)
    else:
        viz2 = np.zeros((300, 80 * 15, 3), dtype=np.uint8)
    viz = np.vstack([viz1, viz2])
    return viz


def view_instance_seg_dataset(dataset, n_mask_class=2):
    """View function for instance segmentation dataset.

    Parameters
    ----------
    dataset: class with __getattr__
        Dataset class.
    n_mask_class: int
        Number of mask classes. Default is 2 (background, foreground).
    """

    def visualize_func(dataset, index):
        img, bboxes, labels, lbls = dataset[index]
        return visualize_instance_segmentation(
            img,
            bboxes,
            labels,
            lbls,
            dataset.class_names,
            n_mask_class=n_mask_class
        )

    return view_dataset(dataset, visualize_func)
