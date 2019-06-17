import warnings

import imgaug.augmenters as iaa
import imgaug.imgaug as ia
from imgaug.parameters import Deterministic
import numpy as np
import skimage.measure
import skimage.transform

import instance_occlsegm_lib.image


def seg_dataset_to_object_data(seg_dataset, random=True, repeat=True,
                               random_state=None, ignore_labels=None,
                               one2one=True):
    if random and not isinstance(random_state, np.random.RandomState):
        random_state = np.random.RandomState(random_state)

    while True:
        if random:
            indices = random_state.randint(
                0, len(seg_dataset), len(seg_dataset))
        else:
            indices = np.arange(0, len(seg_dataset))

        for index in indices:
            img, lbl = seg_dataset[index]

            lbl += 1  # because regionprops ignores label 0
            regionprops = skimage.measure.regionprops(lbl)
            if random:
                random_state.shuffle(regionprops)
            for rp in regionprops:
                rp_label = rp.label - 1
                if ignore_labels and rp_label in ignore_labels:
                    continue

                y1, x1, y2, x2 = rp.bbox
                img_ins = img[y1:y2, x1:x2]
                mask_ins = rp.filled_image

                yield {'label': rp_label, 'img': img_ins, 'mask': mask_ins}

                # an object from an image
                if one2one:
                    # single object per single img, lbl pair
                    # to avoid variation of object data
                    break

        if not repeat:
            break


def augment_object_data(object_data, random_state=None, fit_output=True,
                        aug_color=True, aug_geo=True, augmentations=None,
                        random_order=False, scale=(0.5, 1.0)):
    try:
        iaa.Affine(fit_output=True)
    except TypeError:
        warnings.warn('Your imgaug does not support fit_output kwarg for'
                      'imgaug.augmenters.Affine. Please install via'
                      '\n\n\tpip install -e git+https://github.com/wkentaro/imgaug@affine_resize\n\n'  # NOQA
                      'to enable it.')
        fit_output = False

    if random_state is None:
        random_state = np.random.RandomState()
    if augmentations is None:
        st = lambda x: iaa.Sometimes(0.3, x)  # NOQA
        kwargs_affine = dict(
            order=1,  # order=0 for mask
            cval=0,
            scale=scale,
            translate_px=(-16, 16),
            rotate=(-180, 180),
            shear=(-16, 16),
            mode='constant',
        )
        if fit_output:
            kwargs_affine['fit_output'] = fit_output
        augmentations = [
            st(iaa.WithChannels([0, 1], iaa.Multiply([1, 1.5]))
               if aug_color else iaa.Noop()),
            st(iaa.WithColorspace(
                'HSV',
                children=iaa.WithChannels([1, 2], iaa.Multiply([0.5, 2])))
                if aug_color else iaa.Noop()),
            st(iaa.GaussianBlur(sigma=[0.0, 1.0])
               if aug_color else iaa.Noop()),
            iaa.Sometimes(0.8, iaa.Affine(**kwargs_affine)
                          if aug_geo else iaa.Noop()),
        ]
    aug = iaa.Sequential(
        augmentations,
        random_order=random_order,
        random_state=ia.copy_random_state(random_state),
    )

    def activator_imgs(images, augmenter, parents, default):
        if isinstance(augmenter, iaa.Affine):
            augmenter.order = Deterministic(1)
            augmenter.cval = Deterministic(0)
        return True

    def activator_masks(images, augmenter, parents, default):
        white_lists = (
            iaa.Affine, iaa.PerspectiveTransform, iaa.Sequential, iaa.Sometimes
        )
        if not isinstance(augmenter, white_lists):
            return False
        if isinstance(augmenter, iaa.Affine):
            augmenter.order = Deterministic(0)
            augmenter.cval = Deterministic(0)
        return True

    def activator_lbls(images, augmenter, parents, default):
        white_lists = (
            iaa.Affine, iaa.PerspectiveTransform, iaa.Sequential, iaa.Sometimes
        )
        if not isinstance(augmenter, white_lists):
            return False
        if isinstance(augmenter, iaa.Affine):
            augmenter.order = Deterministic(0)
            augmenter.cval = Deterministic(-1)
        return True

    for objd in object_data:
        aug = aug.to_deterministic()
        objd['img'] = aug.augment_image(
            objd['img'], hooks=ia.HooksImages(activator=activator_imgs))
        if 'mask' in objd:
            objd['mask'] = aug.augment_image(
                objd['mask'], hooks=ia.HooksImages(activator=activator_masks))
        if 'lbl' in objd:
            objd['lbl'] = aug.augment_image(
                objd['lbl'], hooks=ia.HooksImages(activator=activator_lbls))
        if 'lbl_suc' in objd:
            objd['lbl_suc'] = aug.augment_image(
                objd['lbl_suc'],
                hooks=ia.HooksImages(activator=activator_lbls))
        if 'masks' in objd:
            masks = []
            for mask in objd['masks']:
                mask = aug.augment_image(
                    mask,
                    hooks=ia.HooksImages(activator=activator_masks),
                )
                masks.append(mask)
            masks = np.asarray(masks)
            objd['masks'] = masks
            del masks
        if 'lbls' in objd:
            lbls = []
            for lbl in objd['lbls']:
                lbl = aug.augment_image(
                    lbl,
                    hooks=ia.HooksImages(activator=activator_lbls),
                )
                lbls.append(lbl)
            lbls = np.asarray(lbls)
            objd['lbls'] = lbls
            del lbls
        yield objd


def stack_objects(img, lbl, object_data, region_label,
                  random_state=None, stack_ratio=(0.2, 0.99),
                  n_objects=(None, None),
                  return_instances=False):
    # initialize outputs
    img, lbl = img.copy(), lbl.copy()
    lbl_suc = np.zeros(img.shape[:2], dtype=np.int32)
    lbl_suc.fill(-1)

    if random_state is None:
        random_state = np.random.RandomState()

    if isinstance(stack_ratio, tuple) and len(stack_ratio) == 2:
        stack_ratio = random_state.uniform(*stack_ratio)
    assert isinstance(stack_ratio, float)

    bboxes = []
    labels = []
    masks = []
    for i, objd in enumerate(object_data):
        img_h, img_w = img.shape[:2]
        ins_h, ins_w = objd['img'].shape[:2]

        mask_rg = lbl == region_label
        mask_labeled = lbl != -1

        # choose where to put
        Y_r, X_r = np.where(mask_rg)
        x_r = random_state.choice(X_r)
        y_r = random_state.choice(Y_r)
        x1 = max(x_r - ins_w / 2., 0)
        x2 = min(x1 + ins_w, img_w)
        y1 = max(y_r - ins_h / 2., 0)
        y2 = min(y1 + ins_h, img_h)
        x1, x2, y1, y2 = map(int, [x1, x2, y1, y2])
        ins_h = y2 - y1
        ins_w = x2 - x1

        img_ins = objd['img'][:ins_h, :ins_w]
        mask_ins = objd['mask'][:ins_h, :ins_w]
        if 'lbl_suc' in objd:
            lbl_suc_ins = objd['lbl_suc'][:ins_h, :ins_w]

        # put object on current objects
        mask_ins = mask_ins & mask_labeled[y1:y2, x1:x2]
        if mask_ins.sum() == 0:
            continue
        img[y1:y2, x1:x2][mask_ins] = img_ins[mask_ins]
        lbl[y1:y2, x1:x2][mask_ins] = objd['label']
        if 'lbl_suc' in objd:
            lbl_suc[y1:y2, x1:x2][mask_ins] = lbl_suc_ins[mask_ins]

        if return_instances:
            labels.append(objd['label'])
            mask = np.zeros((img_h, img_w), dtype=bool)
            mask[y1:y2, x1:x2][mask_ins] = True
            masks.append(mask)
            bbox = instance_occlsegm_lib.image.masks_to_bboxes([mask])[0]
            assert 0 <= bbox[0] <= img_h  # y1
            assert 0 <= bbox[1] <= img_w  # x1
            assert 0 <= bbox[2] <= img_h  # y2
            assert 0 <= bbox[3] <= img_w  # x2
            bboxes.append(bbox)

        mask_labeled = (lbl != -1)
        mask_objects = mask_labeled & (lbl != region_label)
        stack_ratio_t = 1. * mask_objects.sum() / mask_labeled.sum()
        if stack_ratio_t > stack_ratio:
            break

        if (all(isinstance(x, int) for x in n_objects) and
                not (n_objects[0] <= i <= n_objects[1])):
            break

    if return_instances:
        bboxes = np.asarray(bboxes, dtype=np.int32)
        labels = np.asarray(labels, dtype=np.int32)
        masks = np.asarray(masks, dtype=bool)

    return {
        'img': img,
        'lbl': lbl,
        'lbl_suc': lbl_suc,
        'bboxes': bboxes,
        'labels': labels,
        'masks': masks,
    }
