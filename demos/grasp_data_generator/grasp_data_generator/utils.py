import fcn
import numpy as np
import os.path as osp
import yaml


filepath = osp.dirname(osp.realpath(__file__))


def get_label_names():
    yamlpath = osp.join(filepath, './datasets/data/label_names.yaml')
    with open(yamlpath) as f:
        label_names = ['__backgroud__'] + yaml.load(f)
    return label_names


def grasp_accuracy(grasp_trues, grasp_preds, threshold=0.5):
    grt_mask = np.array(grasp_trues).astype(np.bool)
    grp_mask = np.array(grasp_preds).astype(np.bool)

    # acc
    acc_mask = (grt_mask == grp_mask).astype(np.int32)
    acc_mask = acc_mask.flatten()
    acc = acc_mask.sum() / float(len(acc_mask))

    # true positive
    tp_mask = np.logical_and(grp_mask, grt_mask)
    tp_mask = tp_mask.astype(np.int32)

    # precision
    p_sum = float(np.nansum(grp_mask.astype(np.int32)))
    if p_sum > 0:
        precision = np.nansum(tp_mask) / p_sum
    else:
        precision = 0.0

    # recall
    t_sum = float(np.nansum(grt_mask.astype(np.int32)))
    if t_sum > 0:
        recall = np.nansum(tp_mask) / t_sum
    else:
        recall = 0.0
    return acc, precision, recall


def visualize(**kwargs):
    img = kwargs.get('img')
    lbl_true = kwargs.get('lbl_true')
    lbl_pred = kwargs.get('lbl_pred')
    single_grasp_true = kwargs.get('single_grasp_true')
    single_grasp_pred = kwargs.get('single_grasp_pred')
    dual_grasp_true = kwargs.get('dual_grasp_true')
    dual_grasp_pred = kwargs.get('dual_grasp_pred')
    n_class = kwargs.get('n_class')
    label_names = kwargs.get('label_names')
    alpha = kwargs.get('alpha')

    if lbl_true is None and lbl_pred is None:
        raise ValueError('lbl_true or lbl_pred must be not None.')

    if lbl_true is not None:
        mask = lbl_true == -1
        lbl_true[mask] = 0
        if lbl_pred is not None:
            lbl_pred[mask] = 0

    vizs = []

    if single_grasp_true is not None:
        single_grasp_true = single_grasp_true[:, :, np.newaxis]
        single_grasp_true = np.repeat(single_grasp_true, 3, axis=2)
        single_grasp_true = single_grasp_true * np.array([255, 0, 0])
        single_grasp_true = single_grasp_true.astype(np.int32)
        if alpha is not None:
            single_grasp_true = single_grasp_true * (1.0 - alpha) + img * alpha

    if single_grasp_pred is not None:
        single_grasp_pred = single_grasp_pred[:, :, np.newaxis]
        single_grasp_pred = np.repeat(single_grasp_pred, 3, axis=2)
        single_grasp_pred = single_grasp_pred * np.array([255, 0, 0])
        single_grasp_pred = single_grasp_pred.astype(np.int32)
        if alpha is not None:
            single_grasp_pred = single_grasp_pred * (1.0 - alpha) + img * alpha

    if dual_grasp_true is not None:
        dual_grasp_true = dual_grasp_true[:, :, np.newaxis]
        dual_grasp_true = np.repeat(dual_grasp_true, 3, axis=2)
        dual_grasp_true = dual_grasp_true * np.array([255, 0, 0])
        dual_grasp_true = dual_grasp_true.astype(np.int32)
        if alpha is not None:
            dual_grasp_true = dual_grasp_true * (1.0 - alpha) + img * alpha

    if dual_grasp_pred is not None:
        dual_grasp_pred = dual_grasp_pred[:, :, np.newaxis]
        dual_grasp_pred = np.repeat(dual_grasp_pred, 3, axis=2)
        dual_grasp_pred = dual_grasp_pred * np.array([255, 0, 0])
        dual_grasp_pred = dual_grasp_pred.astype(np.int32)
        if alpha is not None:
            dual_grasp_pred = dual_grasp_pred * (1.0 - alpha) + img * alpha

    if lbl_true is not None:
        if alpha is not None:
            lbl_viz = fcn.utils.label2rgb(
                lbl_true, img, label_names=label_names, n_labels=n_class,
                alpha=alpha)
        else:
            lbl_viz = fcn.utils.label2rgb(
                lbl_true, label_names=label_names, n_labels=n_class)
        if dual_grasp_true is None:
            viz_trues = [
                img,
                lbl_viz,
                single_grasp_true
            ]
            vizs.append(fcn.utils.get_tile_image(viz_trues, (1, 3)))
        else:
            viz_trues = [
                img,
                lbl_viz,
                single_grasp_true,
                dual_grasp_true
            ]
            vizs.append(fcn.utils.get_tile_image(viz_trues, (1, 4)))

    if lbl_pred is not None:
        if alpha is not None:
            lbl_viz = fcn.utils.label2rgb(
                lbl_pred, img, label_names=label_names, n_labels=n_class,
                alpha=alpha)
        else:
            lbl_viz = fcn.utils.label2rgb(
                lbl_pred, label_names=label_names, n_labels=n_class)
        if dual_grasp_pred is None:
            viz_preds = [
                img,
                lbl_viz,
                single_grasp_pred
            ]
            vizs.append(fcn.utils.get_tile_image(viz_preds, (1, 3)))
        else:
            viz_preds = [
                img,
                lbl_viz,
                single_grasp_pred,
                dual_grasp_pred
            ]
            vizs.append(fcn.utils.get_tile_image(viz_preds, (1, 4)))

    if len(vizs) == 1:
        return vizs[0]
    elif len(vizs) == 2:
        return fcn.utils.get_tile_image(vizs, (2, 1))
    else:
        raise RuntimeError
