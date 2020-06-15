import functools
import glob
import json
import math
import os
import os.path as osp
import re

import concurrent.futures
import cv2
import labelme
import numpy as np
import six
import skimage.io

from grasp_fusion_lib.datasets.apc.arc2017 import class_names_arc2017
import grasp_fusion_lib.func
import grasp_fusion_lib.image

import grasp_fusion_lib.data


class ItemData(object):

    def __init__(self, root_dir, img_regex=r'.*\.png$', img_size=180 * 180):
        self.root_dir = root_dir

        self.object_names, self.object_dirs = self._scan()

        self._img_regex = img_regex
        self._img_size = img_size
        self._data = self._load()

    def __len__(self):
        return len(self._data)

    def __getitem__(self, i):
        return self._data[i]

    def __repr__(self):
        params = [
            ('root_dir', self.root_dir),
            ('n_objects', len(self.object_names)),
            ('object_names', self.object_names),
            ('data_size', len(self)),
        ]
        params = ''.join('  {}={},\n'.format(k, repr(v)) for k, v in params)
        return '{}(\n{})'.format(
            self.__class__.__name__,
            params,
        )

    def _scan(self):
        object_names = []
        object_dirs = []
        for obj_dir in sorted(os.listdir(self.root_dir)):
            obj_name = obj_dir.lower()
            obj_dir = osp.join(osp.abspath(self.root_dir), obj_dir)
            info_json = osp.join(obj_dir, '%s.json' % obj_name)
            assert osp.isdir(obj_dir), obj_dir
            assert osp.exists(info_json), info_json

            try:
                data = json.load(open(info_json))
            except Exception as e:
                print('Error on file: %s' % info_json)
                raise Exception(e)
            assert data['name'] == obj_name

            object_names.append(obj_name)
            object_dirs.append(obj_dir)

        return object_names, object_dirs

    @staticmethod
    def load_img(img_file):
        img = skimage.io.imread(img_file)
        if img.shape[2] == 4:  # RGBA
            img[img[:, :, 3] == 0] = 0
            img = img[:, :, :3]  # RGBA -> RGB
        return img

    @staticmethod
    def load_mask(img_file, img):
        labelme_file = osp.splitext(img_file)[0] + '.json'
        if osp.isfile(labelme_file):
            # labelme_file -> mask
            with open(labelme_file) as f:
                data = json.load(f)
            lbl, _ = _labelme_shapes_to_label(
                img.shape, data['shapes'],
            )
            mask = lbl == 1
        else:
            # black region -> mask
            mask = ~(img == 0).all(axis=2)
        return mask

    @staticmethod
    def load_lbl_suc(img_file, img, mask):
        # lbl_suc: -1: unlabeled, 0: no_suction, 1: suction
        suction_file = osp.splitext(img_file)[0] + '.suction.json'
        lbl_suc = np.empty(img.shape[:2], dtype=np.int32)
        lbl_suc.fill(-1)
        lbl_suc[mask] = 1
        if osp.isfile(suction_file):
            data = json.load(open(suction_file))
            lbl_suc, _ = _labelme_shapes_to_label(
                img.shape, data['shapes'])
            lbl_suc[~mask] = -1
            assert np.all(np.isin(np.unique(lbl_suc), [-1, 0, 1]))
        return lbl_suc

    def _load(self):
        n_objects = len(self.object_names)
        object_ids = list(range(1, n_objects + 1))

        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = []
            for object_id, object_name, object_dir in zip(
                    object_ids, self.object_names, self.object_dirs
            ):
                future = executor.submit(
                    self._load_from_object_dir,
                    object_id=object_id,
                    object_name=object_name,
                    object_dir=object_dir,
                )
                futures.append(future)

        data = []
        for future in futures:
            data.extend(future.result())
        return data

    def _load_from_object_dir(self, object_id, object_name, object_dir):
        object_data = []
        for fname in os.listdir(object_dir):
            if not re.match(self._img_regex, fname):
                continue
            img_file = osp.join(object_dir, fname)

            object_data_i = {
                'img_file': img_file,
                'label': object_id,
                'class_name': object_name,
            }
            for k, v in self.load_from_img_file(img_file).items():
                if k in object_data_i:
                    raise ValueError("Key '%s' already exists" % k)
                object_data_i[k] = v
            object_data.append(object_data_i)
        return object_data

    def load_from_img_file(self, img_file):
        """Load object data from image file path.

        Overwrite this method for customize the loading.
        """
        # load
        img = self.load_img(img_file)
        mask = self.load_mask(img_file, img)
        lbl_suc = self.load_lbl_suc(img_file, img, mask)

        # crop
        y1, x1, y2, x2 = grasp_fusion_lib.image.masks_to_bboxes([mask])[0]
        img = img[y1:y2, x1:x2]
        mask = mask[y1:y2, x1:x2]
        lbl_suc = lbl_suc[y1:y2, x1:x2]

        # resize
        img = grasp_fusion_lib.image.resize(img, size=self._img_size)
        mask = grasp_fusion_lib.image.resize(
            mask.astype(np.int32),
            size=self._img_size,
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        lbl_suc = grasp_fusion_lib.image.resize(
            lbl_suc, size=self._img_size, interpolation=cv2.INTER_NEAREST,
        )

        return {'img': img, 'mask': mask, 'lbl_suc': lbl_suc}


class ItemDataARC2017(ItemData):

    def __init__(self):
        root_dir = osp.expanduser('~/data/arc2017/datasets/ItemDataARC2017')
        super(ItemDataARC2017, self).__init__(root_dir=root_dir)

    def download(self):
        path = self.root_dir + '.zip'
        postprocess = functools.partial(
            grasp_fusion_lib.data.extract, path, to=osp.dirname(path),
        )
        grasp_fusion_lib.data.download(
            url='https://drive.google.com/uc?id=1hJe4JZvqc2Ni1sjuKwXuBxgddHH2zNFa',  # NOQA
            md5='c8ad2268b7f2d16accd716c0269d4e5f',
            path=path,
            postprocess=postprocess,
        )


if six.PY2:
    import copy_reg

    def _reduce_method(m):
        if m.im_self is None:
            return getattr, (m.im_class, m.im_func.func_name)
        else:
            return getattr, (m.im_self, m.im_func.func_name)

    copy_reg.pickle(type(ItemData._load_from_object_dir), _reduce_method)


# -----------------------------------------------------------------------------


def _directory_to_hash(data_dir, *args, **kwargs):
    return grasp_fusion_lib.data.directory_to_hash(data_dir)


def _load_item_data(args):
    obj_id, obj_name, obj_dir, skip_known, target_size = args

    obj_data = []

    if skip_known and obj_name in class_names_arc2017 and \
            not glob.glob(osp.join(obj_dir, '*.suction.json')):
        # known object and no grasp region annotation
        return obj_data

    for img_file in glob.glob(osp.join(obj_dir, '*.png')):
        # img
        img = skimage.io.imread(img_file)
        if img.shape[2] == 4:  # RGBA
            img[img[:, :, 3] == 0] = 0
            img = img[:, :, :3]  # RGBA -> RGB

        # mask
        labelme_file = osp.splitext(img_file)[0] + '.json'
        if osp.isfile(labelme_file):
            if labelme is None:
                raise ImportError('Please install labelme\n.')
            # labelme_file -> mask
            data = json.load(open(labelme_file))
            lbl, _ = _labelme_shapes_to_label(
                img.shape, data['shapes'])
            mask = lbl == 1
        else:
            # black region -> mask
            mask = ~(img == 0).all(axis=2)

        # lbl_suc: -1: unlabeled, 0: no_suction, 1: suction
        suction_file = osp.splitext(img_file)[0] + '.suction.json'
        lbl_suc = np.empty(img.shape[:2], dtype=np.int32)
        lbl_suc.fill(-1)
        lbl_suc[mask] = 1
        if osp.isfile(suction_file):
            data = json.load(open(suction_file))
            lbl_suc, _ = _labelme_shapes_to_label(
                img.shape, data['shapes'])
            lbl_suc[~mask] = -1
            assert np.all(np.isin(np.unique(lbl_suc), [-1, 0, 1]))

        # crop
        y1, x1, y2, x2 = grasp_fusion_lib.image.masks_to_bboxes([mask])[0]
        img = img[y1:y2, x1:x2]
        mask = mask[y1:y2, x1:x2]
        lbl_suc = lbl_suc[y1:y2, x1:x2]

        # resize
        scale = math.sqrt((1. * target_size * target_size) /
                          (img.shape[0] * img.shape[1]))
        img = cv2.resize(img, None, None, fx=scale, fy=scale)
        mask = cv2.resize(
            mask.astype(np.int32), None, None, fx=scale, fy=scale,
            interpolation=cv2.INTER_NEAREST).astype(bool)
        lbl_suc = cv2.resize(
            lbl_suc, None, None, fx=scale, fy=scale,
            interpolation=cv2.INTER_NEAREST)

        obj_data.append({
            'img_file': img_file,
            'label': obj_id,
            'img': img,
            'mask': mask,
            'lbl_suc': lbl_suc,
            # XXX: backward compatibility
            'lbl': grasp_fusion_lib.image.mask_to_lbl(mask, obj_id),
        })

    return obj_data


@grasp_fusion_lib.func.cache(key=_directory_to_hash)
def load_item_data(item_data_dir, skip_known=True, target_size=180):
    item_data = ItemData(item_data_dir)
    object_names = item_data.object_names
    object_dirs = item_data.object_dirs
    object_ids = [i + 1 for i in range(len(object_names))]

    with concurrent.futures.ProcessPoolExecutor() as executor:
        params = [(obj_id, obj_name, obj_dir, skip_known, target_size)
                  for obj_id, obj_name, obj_dir
                  in zip(object_ids, object_names, object_dirs)]
        object_data_each = executor.map(
            _load_item_data, params)

    object_data = []
    for obj_data in object_data_each:
        object_data.extend(obj_data)

    return object_names, object_data


def _labelme_shapes_to_label(img_shape, shapes):
    label_name_to_value = {'_background_': 0}
    for shape in shapes:
        label_name = shape['label']
        if label_name in label_name_to_value:
            label_value = label_name_to_value[label_name]
        else:
            label_value = len(label_name_to_value)
            label_name_to_value[label_name] = label_value

    lbl, _ = labelme.utils.shapes_to_label(img_shape, shapes, label_name_to_value)
    return lbl, label_name_to_value
