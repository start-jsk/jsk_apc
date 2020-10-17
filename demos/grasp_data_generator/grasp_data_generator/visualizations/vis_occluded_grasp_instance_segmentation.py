from __future__ import division

import numpy as np

from chainercv.visualizations.colormap import voc_colormap
from chainercv.visualizations import vis_image

from grasp_data_generator.models.occluded_grasp_mask_rcnn.utils \
    import rot_lbl_to_rot


def vis_occluded_grasp_instance_segmentation(
        img, ins_label, label=None, bbox=None, score=None,
        sg_label=None, dg_label=None, label_names=None, rotate_angle=None,
        instance_colors=None, alpha=0.7, linewidth=1., fontsize=8, prefix=None,
        axes=None,
):
    from matplotlib import pyplot as plt

    if bbox is not None and len(bbox) != len(ins_label):
        raise ValueError('The length of mask must be same as that of bbox')
    if label is not None and len(bbox) != len(label):
        raise ValueError('The length of label must be same as that of bbox')
    if score is not None and len(bbox) != len(score):
        raise ValueError('The length of score must be same as that of bbox')

    n_inst = len(bbox)
    if instance_colors is None:
        instance_colors = voc_colormap(list(range(1, n_inst + 1)))
    instance_colors = np.array(instance_colors)

    if axes is None:
        f, axes = plt.subplots(1, 5, sharey=True)
    else:
        f = None

    ins_names = ['background', 'visible', 'occluded']
    for ins_id, ax in enumerate(axes[:3]):
        if prefix is None:
            ax.set_title(ins_names[ins_id])
        else:
            ax.set_title('{0} : {1}'.format(prefix, ins_names[ins_id]))
        ax = vis_image(img, ax=ax)
        _, H, W = img.shape
        canvas_img = np.zeros((H, W, 4), dtype=np.uint8)
        for i, (bb, ins_lbl) in enumerate(zip(bbox, ins_label)):
            # The length of `colors` can be smaller than the number of
            # instances if a non-default `colors` is used.
            color = instance_colors[i % len(instance_colors)]
            rgba = np.append(color, alpha * 255)
            bb = np.round(bb).astype(np.int32)
            y_min, x_min, y_max, x_max = bb
            if y_max > y_min and x_max > x_min:
                ins_mask = ins_lbl[y_min:y_max, x_min:x_max] == ins_id
                canvas_img[y_min:y_max, x_min:x_max][ins_mask] = rgba

            xy = (bb[1], bb[0])
            height = bb[2] - bb[0]
            width = bb[3] - bb[1]
            ax.add_patch(plt.Rectangle(
                xy, width, height, fill=False,
                edgecolor=color / 255, linewidth=linewidth, alpha=alpha))

            caption = []
            if label is not None and label_names is not None:
                lb = label[i]
                if not (0 <= lb < len(label_names)):
                    raise ValueError('No corresponding name is given')
                caption.append(label_names[lb])
            if score is not None:
                sc = score[i]
                caption.append('{:.2f}'.format(sc))

            if len(caption) > 0:
                ax.text((x_max + x_min) / 2, y_min,
                        ': '.join(caption),
                        style='italic',
                        bbox={'facecolor': color / 255, 'alpha': alpha},
                        fontsize=fontsize, color='white')

        ax.imshow(canvas_img)

    ax3, ax4 = axes[3:5]
    if prefix is None:
        ax3.set_title('single grasp')
    else:
        ax3.set_title('{0} : single grasp'.format(prefix))
    ax3 = vis_image(img, ax=ax3)
    _, H, W = img.shape
    canvas_img = np.zeros((H, W, 4), dtype=np.uint8)
    for i, (bb, sg_lbl) in enumerate(zip(bbox, sg_label)):
        count = np.bincount(sg_lbl.flatten(), minlength=1)
        # no grasp mask
        if len(count) == 1:
            continue
        rot_id = np.argmax(count[1:]) + 1

        # The length of `colors` can be smaller than the number of
        # instances if a non-default `colors` is used.
        color = instance_colors[i % len(instance_colors)]
        rgba = np.append(color, alpha * 255)
        bb = np.round(bb).astype(np.int32)
        y_min, x_min, y_max, x_max = bb
        if y_max > y_min and x_max > x_min:
            canvas_img[sg_lbl == rot_id] = rgba

        xy = (bb[1], bb[0])
        height = bb[2] - bb[0]
        width = bb[3] - bb[1]
        ax3.add_patch(plt.Rectangle(
            xy, width, height, fill=False,
            edgecolor=color / 255, linewidth=linewidth, alpha=alpha))

        caption = []
        if label is not None and label_names is not None:
            lb = label[i]
            if not (0 <= lb < len(label_names)):
                raise ValueError('No corresponding name is given')
            caption.append(label_names[lb])
        if score is not None:
            sc = score[i]
            caption.append('{:.2f}'.format(sc))
        if rotate_angle is not None:
            rot = rot_lbl_to_rot(rot_id, rotate_angle)
            caption.append('{} degree'.format(rot))

        if len(caption) > 0:
            ax3.text((x_max + x_min) / 2, y_min,
                     ': '.join(caption),
                     style='italic',
                     bbox={'facecolor': color / 255, 'alpha': alpha},
                     fontsize=fontsize, color='white')
    ax3.imshow(canvas_img)

    if prefix is None:
        ax4.set_title('dual grasp')
    else:
        ax4.set_title('{0} : dual grasp'.format(prefix))
    ax4 = vis_image(img, ax=ax4)
    _, H, W = img.shape
    canvas_img = np.zeros((H, W, 4), dtype=np.uint8)
    for i, (bb, dg_lbl) in enumerate(zip(bbox, dg_label)):
        count = np.bincount(dg_lbl.flatten(), minlength=1)
        # no grasp mask
        if len(count) == 1:
            continue
        rot_id = np.argmax(count[1:]) + 1

        # The length of `colors` can be smaller than the number of
        # instances if a non-default `colors` is used.
        color = instance_colors[i % len(instance_colors)]
        rgba = np.append(color, alpha * 255)
        bb = np.round(bb).astype(np.int32)
        y_min, x_min, y_max, x_max = bb
        if y_max > y_min and x_max > x_min:
            canvas_img[dg_lbl == rot_id] = rgba

        xy = (bb[1], bb[0])
        height = bb[2] - bb[0]
        width = bb[3] - bb[1]
        ax4.add_patch(plt.Rectangle(
            xy, width, height, fill=False,
            edgecolor=color / 255, linewidth=linewidth, alpha=alpha))

        caption = []
        if label is not None and label_names is not None:
            lb = label[i]
            if not (0 <= lb < len(label_names)):
                raise ValueError('No corresponding name is given')
            caption.append(label_names[lb])
        if score is not None:
            sc = score[i]
            caption.append('{:.2f}'.format(sc))
        if rotate_angle is not None and dg_lbl.max() > 0:
            rot = rot_lbl_to_rot(rot_id, rotate_angle)
            caption.append('{} degree'.format(rot))

        if len(caption) > 0:
            ax4.text((x_max + x_min) / 2, y_min,
                     ': '.join(caption),
                     style='italic',
                     bbox={'facecolor': color / 255, 'alpha': alpha},
                     fontsize=fontsize, color='white')
    ax4.imshow(canvas_img)

    return f, axes
