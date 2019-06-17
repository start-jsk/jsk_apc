import os.path as osp

import instance_occlsegm_lib


here = osp.dirname(osp.abspath(__file__))


def voc_image():
    voc_id = '2007_000032'
    data_dir = osp.join(here, '_data/voc', voc_id)

    img_file = osp.join(data_dir, 'image.jpg')
    img = instance_occlsegm_lib.io.imread(img_file)
    cls_file = osp.join(data_dir, 'lbl_class.png')
    cls = instance_occlsegm_lib.io.lbread(cls_file)
    ins_file = osp.join(data_dir, 'lbl_instance.png')
    ins = instance_occlsegm_lib.io.lbread(ins_file)

    return img, cls, ins
