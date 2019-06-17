import skimage.data

import instance_occlsegm_lib


def test_tileplot():
    args_lst = []
    for img in ['astronaut', 'camera', 'coffee', 'horse']:
        img = getattr(skimage.data, img)()
        args_lst.append((img,))

    instance_occlsegm_lib.io.tileplot(
        instance_occlsegm_lib.io.imgplot, args_lst, shape=(2, 2))
