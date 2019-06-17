import numpy as np
import skimage.io

import instance_occlsegm_lib.datasets.apc.arc2017
import instance_occlsegm_lib.image


def load_item_data(item_data_dir, load_pred=True, skip_known=False):
    obj_names, obj_data = instance_occlsegm_lib.datasets.apc.\
        arc2017.load_item_data(item_data_dir, skip_known=skip_known)
    if not load_pred:
        return obj_names, obj_data

    for obj_datum in obj_data:
        mask_file = obj_datum['img_file'] + '.mask_pred.jpg'
        mask = skimage.io.imread(mask_file) > 127
        if mask.ndim == 3:
            mask = mask[:, :, 0]

        H, W = obj_datum['img'].shape[:2]
        mask = mask.astype(np.float32)  # bool -> float
        mask = instance_occlsegm_lib.image.resize(mask, height=H, width=W)
        mask = mask > 0.5  # float -> bool

        obj_datum['mask'] = mask
        if 'lbl' in obj_datum:
            obj_datum.pop('lbl')
    return obj_names, obj_data
