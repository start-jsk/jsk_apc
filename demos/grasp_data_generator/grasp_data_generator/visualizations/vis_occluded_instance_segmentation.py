from __future__ import division

import numpy as np

from chainercv.visualizations.colormap import voc_colormap
from chainercv.visualizations import vis_image


def vis_occluded_instance_segmentation(
        img, ins_label, label=None, bbox=None, score=None, label_names=None,
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
        f, axes = plt.subplots(1, 3, sharey=True)
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
    return f, axes
