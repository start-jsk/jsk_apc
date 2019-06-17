import copy

import chainer
import numpy as np

import instance_occlsegm_lib


def random_crop(img, size, random_state=None):
    H, W = img.shape[:2]
    assert isinstance(size[0], int)
    assert isinstance(size[1], int)
    assert len(size) == 2
    assert H >= size[0]
    assert W >= size[1]
    if H == size[0]:
        ymin = 0
    else:
        ymin = random_state.randint(0, max(0, H - size[0]))
    ymax = ymin + size[0]
    if W == size[1]:
        xmin = 0
    else:
        xmin = random_state.randint(0, max(0, W - size[1]))
    xmax = xmin + size[1]
    img_cropped = img[ymin:ymax, xmin:xmax]
    return img_cropped


class ClassSegRandomCropDataset(chainer.dataset.DatasetMixin):

    def __init__(self, dataset, size):
        self.class_names = dataset.class_names
        self.dataset = dataset
        assert isinstance(size, int)
        self.size = size

    def __len__(self):
        return len(self.dataset)

    def get_example(self, i):
        img, lbl = self.dataset[i]

        random_state = np.random.RandomState()
        obj_datum = next(instance_occlsegm_lib.aug.augment_object_data(
            object_data=[dict(img=img, lbl=lbl)],
            random_state=random_state,
            fit_output=True,
            aug_color=False,
            aug_geo=True))
        img = obj_datum['img']
        lbl = obj_datum['lbl']

        mask_labeled = lbl != -1
        img[~mask_labeled] = np.random.normal(0, 255, size=img.shape)\
            .astype(img.dtype)[~mask_labeled]
        y1, x1, y2, x2 = instance_occlsegm_lib.image.masks_to_bboxes(
            [mask_labeled])[0]
        img = img[y1:y2, x1:x2]
        lbl = lbl[y1:y2, x1:x2]

        H, W = img.shape[:2]
        assert lbl.shape == (H, W)

        size = self.size
        if min(H, W) < size:
            if H < W:
                img = instance_occlsegm_lib.image.resize(img, height=size)
                lbl = instance_occlsegm_lib.image.resize(
                    lbl, height=size, interpolation=0)
            else:
                img = instance_occlsegm_lib.image.resize(img, width=size)
                lbl = instance_occlsegm_lib.image.resize(
                    lbl, width=size, interpolation=0)

        img = random_crop(img, size=(size, size),
                          random_state=copy.copy(random_state))
        lbl = random_crop(lbl, size=(size, size),
                          random_state=copy.copy(random_state))
        return img, lbl
