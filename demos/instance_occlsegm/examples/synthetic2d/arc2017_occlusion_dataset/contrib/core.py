import json
import logging

import labelme
import numpy as np


labelme.logger.logger.setLevel(logging.ERROR)  # ignore WARNING


def get_class_names():
    from instance_occlsegm_lib.datasets.apc.arc2017 import class_names_arc2017
    class_names = list(class_names_arc2017[:-1])
    class_names[0] = '__background__'
    class_names = np.asarray(class_names)
    class_names.setflags(write=0)
    return class_names


def load_json_file(json_file):
    data = json.load(open(json_file))
    img = labelme.utils.img_b64_to_arr(data['imageData'])
    img_shape = img.shape[:2]

    shapes_fg, shapes_bg = [], []
    for shape in data['shapes']:
        if shape['label'] == '0':
            shapes_bg.append(shape)
        else:
            shapes_fg.append(shape)

    lbl, label_names = labelme.utils.labelme_shapes_to_label(
        img_shape, shapes_bg)
    mask_bg = lbl != 0
    lbl, label_names = labelme.utils.labelme_shapes_to_label(
        img_shape, shapes_fg)

    lbl_new = - np.ones(lbl.shape, dtype=np.int32)
    lbl_new[mask_bg] = 0
    for label_id, label_name in enumerate(label_names):
        mask = lbl == label_id
        try:
            lbl_new[mask] = int(label_name)
        except Exception:
            assert label_name == '_background_'
    lbl_cls = lbl_new.copy()
    lbl_ins = lbl_cls.copy()
    lbl_ins[lbl_cls == 0] = -1
    lbl_ins[lbl_cls == -1] = -1
    img[lbl_cls == -1] = 0

    return img, lbl_ins, lbl_cls
