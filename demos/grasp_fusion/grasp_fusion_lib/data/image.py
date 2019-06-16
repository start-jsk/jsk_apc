import os.path as osp

import grasp_fusion_lib


here = osp.dirname(osp.abspath(__file__))


def voc_image():
    voc_id = '2007_000032'
    data_dir = osp.join(here, '_data/voc', voc_id)

    img_file = osp.join(data_dir, 'image.jpg')
    img = grasp_fusion_lib.io.imread(img_file)
    cls_file = osp.join(data_dir, 'lbl_class.png')
    cls = grasp_fusion_lib.io.lbread(cls_file)
    ins_file = osp.join(data_dir, 'lbl_instance.png')
    ins = grasp_fusion_lib.io.lbread(ins_file)

    return img, cls, ins
