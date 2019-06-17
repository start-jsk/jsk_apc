from __future__ import print_function

import numpy as np


def find_obstacles(target_id, bboxes, labels, masks, label_names=None):
    if label_names is None:
        label_names = np.array(['label_%d' % l for l in labels])

    obstacles = []

    mask_occluded = masks[target_id] == 2
    mask_whole = np.isin(masks[target_id], [1, 2])
    occluded_ratio = 1. * mask_occluded.sum() / mask_whole.sum()
    print(
        'occluded_ratio: {}:{}: {:%}'.format(
            labels[target_id],
            label_names[labels[target_id]],
            occluded_ratio,
        ),
    )
    if occluded_ratio <= 0.1:
        return obstacles

    for ins_id, (label, mask) in enumerate(zip(labels, masks)):
        if ins_id == target_id:
            continue
        mask_occluded_by_this = np.bitwise_and(mask == 1, mask_occluded)
        try:
            ratio_occluded_by_this = (
                1. * mask_occluded_by_this.sum() / mask_occluded.sum()
            )
        except ZeroDivisionError:
            ratio_occluded_by_this = 0
        if ratio_occluded_by_this > 0.1:
            print(
                'Target: {}:{} is occluded by {}:{}: {:%}'.format(
                    labels[target_id],
                    label_names[labels[target_id]],
                    label,
                    label_names[label],
                    ratio_occluded_by_this,
                ),
            )
            obstacles.append(ins_id)
    return obstacles
