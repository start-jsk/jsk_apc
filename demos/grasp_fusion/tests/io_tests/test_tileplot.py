import skimage.data

import grasp_fusion_lib


def test_tileplot():
    args_lst = []
    for img in ['astronaut', 'camera', 'coffee', 'horse']:
        img = getattr(skimage.data, img)()
        args_lst.append((img,))

    grasp_fusion_lib.io.tileplot(
        grasp_fusion_lib.io.imgplot, args_lst, shape=(2, 2))
