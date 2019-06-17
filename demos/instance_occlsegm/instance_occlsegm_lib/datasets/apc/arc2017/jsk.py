import copy
import math
import os
import os.path as osp

import chainer
import cv2
import imgaug.augmenters as iaa
import numpy as np
import skimage.io
try:
    from sklearn.model_selection import train_test_split
except ImportError:
    from sklearn.cross_validation import train_test_split

import instance_occlsegm_lib.aug
import instance_occlsegm_lib.data
from instance_occlsegm_lib.datasets.apc.arc2017.base import class_names_arc2017
from instance_occlsegm_lib.datasets.apc.arc2017.base import DATASETS_DIR
from instance_occlsegm_lib.datasets.apc.arc2017.base import \
    get_class_id_map_from_2016_to_2017
from instance_occlsegm_lib.datasets.apc.arc2017.item_data import load_item_data


class CollectedOnKivaPod(chainer.dataset.DatasetMixin):

    """ARC2017 dataset collected for/on Kiva Pod at JSK Lab."""

    class_names = class_names_arc2017

    def __init__(self, split='train', aug='standard'):
        assert split in ['all', 'train', 'valid']
        self.split = split

        assert aug in ['none', 'standard', 'stack']
        self.aug_method = aug

        ids = []
        dataset_dir = osp.join(DATASETS_DIR, 'dataset_jsk_v1_20170325_kivapod_annotated')  # NOQA
        if not osp.exists(dataset_dir):
            self.download()
        for scene_dir in os.listdir(dataset_dir):
            scene_dir = osp.join(dataset_dir, scene_dir)
            if osp.exists(osp.join(scene_dir, 'label.npz')):
                ids.append(scene_dir)
        dataset_dir = osp.join(DATASETS_DIR, 'JSKARC2017From16')
        if not osp.exists(dataset_dir):
            self._convert_dataset_16_to_17()
        for scene_dir in os.listdir(dataset_dir):
            if scene_dir == '1466804951244465112':
                # Wrong annotation
                continue
            scene_dir = osp.join(dataset_dir, scene_dir)
            ids.append(scene_dir)
        # split ids to train/val
        ids_train, ids_valid = train_test_split(
            ids, test_size=0.25, random_state=1)
        self._ids = {'all': ids, 'train': ids_train, 'valid': ids_valid}

    def _convert_dataset_16_to_17(self):
        cls_id_16_to_17 = get_class_id_map_from_2016_to_2017()

        out_dir = osp.join(DATASETS_DIR, 'JSKARC2017From16')

        dataset_apc2016 = instance_occlsegm_lib.datasets.apc.JskAPC2016Dataset(
            split='all')
        for index in range(len(dataset_apc2016)):
            img, lbl = dataset_apc2016[index]

            # conversion
            for i16 in np.unique(lbl):
                lbl[lbl == i16] = cls_id_16_to_17.get(i16, -1)

            # visualization
            lbl_viz = lbl.copy()
            lbl_viz[lbl == -1] = 0
            lbl_viz = instance_occlsegm_lib.image.label2rgb(
                lbl_viz, img, label_names=class_names_arc2017)
            lbl_viz[lbl == -1] = (0, 0, 0)

            # save
            out_sub_dir = osp.join(out_dir, '%08d' % index)
            if not osp.exists(out_sub_dir):
                os.makedirs(out_sub_dir)
            out_img_file = osp.join(out_sub_dir, 'image.jpg')
            skimage.io.imsave(out_img_file, img)
            out_lbl_file = osp.join(out_sub_dir, 'label.npz')
            np.savez_compressed(out_lbl_file, lbl)
            out_lbl_viz_file = osp.join(out_sub_dir, 'label_viz.jpg')
            skimage.io.imsave(out_lbl_viz_file, lbl_viz)
            print('saved to: %s' % out_sub_dir)

    def download(self):
        instance_occlsegm_lib.data.download(
            url='https://drive.google.com/uc?id=1_E2Ao9DbgRm70EyWMxwf2jUWY7LK0lis',  # NOQA
            path=osp.join(DATASETS_DIR, 'dataset_jsk_v1_20170325_kivapod_annotated.tgz'),  # NOQA
            md5='352dc0dace3ef06af4166a7b9497f8ed',
            postprocess=instance_occlsegm_lib.data.extractall,
        )

    def __len__(self):
        return len(self._ids[self.split])

    def get_example(self, i):
        if self.split == 'valid':
            random_state = np.random.RandomState(i)
        else:
            random_state = np.random.RandomState(np.random.randint(0, 10 ** 7))

        scene_dir = self._ids[self.split][i]
        img_file = osp.join(scene_dir, 'image.jpg')
        img = skimage.io.imread(img_file)
        lbl_file = osp.join(scene_dir, 'label.npz')
        lbl = np.load(lbl_file)['arr_0']
        if self.aug_method == 'stack':
            object_data = instance_occlsegm_lib.aug.seg_dataset_to_object_data(
                seg_dataset=CollectedOnKivaPod(split=self.split, aug='none'),
                random_state=random_state, ignore_labels=[-1, 0, 41])
            obj_datum = instance_occlsegm_lib.aug.stack_objects(
                img, lbl, object_data, region_label=41,
                random_state=random_state)
            obj_datum = next(instance_occlsegm_lib.aug.augment_object_data(
                [obj_datum], random_state))
            img = obj_datum['img']
            lbl = obj_datum['lbl']
        if self.aug_method != 'none':
            obj_datum = {'img': img, 'lbl': lbl}
            obj_datum = next(instance_occlsegm_lib.aug.augment_object_data(
                [obj_datum], random_state, fit_output=False))
            img = obj_datum['img']
            lbl = obj_datum['lbl']
        return img, lbl


class JskARC2017DatasetV1(chainer.dataset.DatasetMixin):

    class_names = class_names_arc2017

    def __init__(self, split='train', aug='standard'):
        self.datasets = [
            CollectedOnKivaPod(split, aug),
        ]

    def __len__(self):
        return sum(len(d) for d in self.datasets)

    @property
    def split(self):
        split = self.datasets[0].split
        assert all(d.split == split for d in self.datasets)
        return split

    @split.setter
    def split(self, value):
        for d in self.datasets:
            d.split = value

    def get_example(self, index):
        skipped = 0
        for dataset in self.datasets:
            current_index = index - skipped
            if current_index < len(dataset):
                return dataset[current_index]
            skipped += len(dataset)


def get_shelf_data():

    def load(img_file, lbl_file):
        img = skimage.io.imread(img_file)
        img = skimage.transform.rescale(
            img, order=1, cval=0, scale=0.5,
            mode='constant', preserve_range=True, multichannel=True,
            anti_aliasing=False)
        img = img.astype(np.uint8)
        lbl = np.load(lbl_file)['arr_0']
        lbl = skimage.transform.rescale(
            lbl, order=0, cval=-1, scale=0.5,
            mode='constant', preserve_range=True, multichannel=False,
            anti_aliasing=False)
        lbl = lbl.astype(np.int32)
        return img, lbl

    if not osp.exists(osp.join(DATASETS_DIR, 'shelfv2')):
        instance_occlsegm_lib.data.download(
            url='https://drive.google.com/uc?id=1aO4hbDDLsiymsddLgXsKDB6xRCkbeerp',  # NOQA
            path=osp.join(DATASETS_DIR, 'shelfv2.zip'),
            md5='e60d78b89f1bf2dc7219491c61269bed',
            postprocess=instance_occlsegm_lib.data.extractall,
        )

    imgs = []
    lbls = []
    # # shelfv1
    # img_file = osp.join(DATASETS_DIR, 'shelfv1/1495830136833987/image.jpg')
    # lbl_file = osp.join(DATASETS_DIR, 'shelfv1/1495830136833987/label.npz')
    # shelfv2
    img_file = osp.join(DATASETS_DIR, 'shelfv2/1495830136833987/image.jpg')
    lbl_file = osp.join(DATASETS_DIR, 'shelfv2/1495830136833987/label.npz')
    img, lbl = load(img_file, lbl_file)
    imgs.append(img)
    lbls.append(lbl)

    if not osp.exists(osp.join(DATASETS_DIR, 'tote')):
        instance_occlsegm_lib.data.download(
            url='https://drive.google.com/uc?id=11KpC4r-NYT0uBN2-YIH9opAhKBk4AytV',  # NOQA
            path=osp.join(DATASETS_DIR, 'tote.zip'),
            md5='4a416787e76d4509bf1464d24041677d',
            postprocess=instance_occlsegm_lib.data.extractall,
        )

    # tote
    img_file = osp.join(DATASETS_DIR, 'tote/1497260472135926/image.jpg')
    lbl_file = osp.join(DATASETS_DIR, 'tote/1497260472135926/label.npz')
    img, lbl = load(img_file, lbl_file)
    imgs.append(img)
    lbls.append(lbl)
    return imgs, lbls


def augment_with_zoom(img, lbl):
    random_state = np.random.randint(0, 9999)
    h, w = img.shape[:2]
    trans_x, trans_y = int(w * 0.3), int(h * 0.3)
    affine_params = dict(
        scale=(0.8, 1.5),
        translate_px={'x': (-trans_x, trans_x), 'y': (-trans_y, trans_y)},
        random_state=random_state,
    )
    aug = iaa.Affine(order=1, cval=0, **affine_params)
    img = aug.augment_image(img)
    aug = iaa.Affine(order=0, cval=-1, **affine_params)
    lbl = aug.augment_image(lbl)
    return img, lbl


class OnStorageSystem(CollectedOnKivaPod):

    class_names = class_names_arc2017

    def __init__(self, split='train', aug='stack'):
        super(OnStorageSystem, self).__init__(split=split, aug=aug)
        self.shelf_img, self.shelf_lbl = get_shelf_data()

    def get_example(self, i):
        if self.split == 'valid':
            random_state = np.random.RandomState(i)
        else:
            random_state = np.random.RandomState(np.random.randint(0, 10 ** 7))

        scene_dir = self._ids[self.split][i]
        img_file = osp.join(scene_dir, 'image.jpg')
        img = skimage.io.imread(img_file)
        lbl_file = osp.join(scene_dir, 'label.npz')
        lbl = np.load(lbl_file)['arr_0']
        if self.aug_method == 'stack':
            index_shelf = random_state.randint(0, len(self.shelf_img))
            img = self.shelf_img[index_shelf]
            lbl = self.shelf_lbl[index_shelf]

            object_data = instance_occlsegm_lib.aug.seg_dataset_to_object_data(
                seg_dataset=OnStorageSystem(split=self.split, aug='none'),
                random_state=random_state, ignore_labels=[-1, 0, 41])
            obj_datum = instance_occlsegm_lib.aug.stack_objects(
                img, lbl, object_data, region_label=41,
                random_state=random_state)
            img = obj_datum['img']
            lbl = obj_datum['lbl']
        if self.aug_method != 'none':
            obj_datum = {'img': img, 'lbl': lbl}
            obj_datum = next(instance_occlsegm_lib.aug.augment_object_data(
                [obj_datum], random_state, fit_output=False))
            img = obj_datum['img']
            lbl = obj_datum['lbl']
        return img, lbl


# class CollectedOnHand(chainer.dataset.DatasetMixin):
#
#     """Dataset collected on Hand at JSK Lab."""
#
#     class_names = class_names_arc2017
#
#     def __init__(self, split='train', size=100):
#         self.split = split
#         self.size = size
#         dataset_dir = osp.join(DATASETS_DIR, 'dataset_jsk_v2_20170527_onhand_annotated')  # NOQA
#         self._ids = []
#         for cls_id, cls_name in enumerate(self.class_names):
#             cls_dir = osp.join(dataset_dir, '%02d' % cls_id)
#             if not osp.exists(cls_dir):
#                 continue
#             for scene in os.listdir(cls_dir):
#                 scene_dir = osp.join(cls_dir, scene)
#                 self._ids.append((cls_id, scene_dir))
#         self.shelf_img, self.shelf_lbl = get_shelf_data()
#
#     def __len__(self):
#         if self.split == 'train':
#             return self.size
#         else:
#             return 0
#
#     def get_example(self, i, aug=True):
#         if self.split == 'valid':
#             random_state = np.random.RandomState(i)
#         else:
#             random_state = np.random.RandomState(
#                 np.random.randint(0, 10 ** 7))
#
#         idx = random_state.randint(0, len(self._ids))
#         cls_id, scene_dir = self._ids[idx]
#
#         img_file = osp.join(scene_dir, 'image.jpg')
#         img = skimage.io.imread(img_file)
#         mask_file = osp.join(scene_dir, 'mask.jpg')
#         mask = skimage.io.imread(mask_file, as_gray=True) >= 0.5
#         lbl = np.zeros(img.shape[:2], dtype=np.int32)
#         lbl[mask] = cls_id
#
#         if aug:
#             dataset = copy.deepcopy(self)
#             indices = random_state.randint(0, len(self), 1000)
#
#             def image_generator(indices):
#                 for j in indices:
#                     img, lbl = dataset.get_example(j, aug=False)
#                     img = skimage.transform.rescale(
#                         img, 0.25, order=1, cval=0,
#                         mode='constant', preserve_range=True,
#                         multichannel=True, anti_aliasing=False)
#                     lbl = skimage.transform.rescale(
#                         lbl, 0.25, order=0, cval=-1,
#                         mode='constant', preserve_range=True,
#                         multichannel=False, anti_aliasing=False)
#                     img = img.astype(np.uint8)
#                     lbl = lbl.astype(np.int32)
#                     yield {'img': img, 'lbl': lbl}
#
#             obj_data = image_generator(indices)
#
#             index_shelf = random_state.randint(0, len(self.shelf_img))
#             img = self.shelf_img[index_shelf].copy()
#             lbl = self.shelf_lbl[index_shelf].copy()
#
#             obj_datum = instance_occlsegm_lib.aug.stack_objects(
#                 img, lbl, obj_data)
#             img = obj_datum['img']
#             lbl = obj_datum['lbl']
#             img, lbl = augment_with_zoom(img, lbl)
#
#         return img, lbl


class JskARC2017DatasetV2(JskARC2017DatasetV1):

    class_names = class_names_arc2017

    def __init__(self, split='train'):
        if split == 'train':
            self.datasets = [OnStorageSystem('train', aug='stack')]
        elif split == 'valid':
            self.datasets = [OnStorageSystem('valid', aug='stack')]
        else:
            raise ValueError('Unsupported split: %s' % split)


class JskARC2017DatasetV3(chainer.dataset.DatasetMixin):

    class_names = class_names_arc2017

    def __init__(self, split='train', aug='stack'):
        assert split in ['all', 'train', 'valid']
        self.split = split
        self._init_ids()

        assert aug in ['none', 'standard', 'stack']
        self.aug_method = aug

    def _init_ids(self):
        sub_datasets = [
            'dataset_jsk_v3_20170614_shelfv1_tote_annotated',
            'dataset_jsk_v3_20170715_shelfv2_annotated',
            'dataset_jsk_v3_20170723_tote_annotated',
        ]
        ids = []
        for sub_dataset in sub_datasets:
            sub_dataset_dir = osp.join(DATASETS_DIR, sub_dataset)
            if not osp.exists(sub_dataset_dir):
                self.download()
            for scene_dir in os.listdir(sub_dataset_dir):
                scene_dir = osp.join(sub_dataset_dir, scene_dir)
                if osp.exists(osp.join(scene_dir, 'label.npz')):
                    ids.append(scene_dir)
        ids_train, ids_valid = train_test_split(
            ids, test_size=0.25, random_state=1)
        self._ids = {'all': ids, 'train': ids_train, 'valid': ids_valid}

    def __len__(self):
        return len(self._ids[self.split])

    def get_example(self, i):
        if self.split == 'valid':
            random_state = np.random.RandomState(i)
        else:
            random_state = np.random.RandomState(np.random.randint(0, 10 ** 7))

        scene_dir = self._ids[self.split][i]
        img_file = osp.join(scene_dir, 'image.jpg')
        img = skimage.io.imread(img_file)
        lbl_file = osp.join(scene_dir, 'label.npz')
        lbl = np.load(lbl_file)['arr_0']
        if self.aug_method == 'stack':
            object_data = instance_occlsegm_lib.aug.seg_dataset_to_object_data(
                seg_dataset=JskARC2017DatasetV3(split=self.split, aug='none'),
                random_state=random_state, ignore_labels=[-1, 0, 41])
            obj_datum = instance_occlsegm_lib.aug.stack_objects(
                img, lbl, object_data, region_label=41,
                random_state=random_state)
            img = obj_datum['img']
            lbl = obj_datum['lbl']
        if self.aug_method != 'none':
            obj_datum = {'img': img, 'lbl': lbl}
            obj_datum = next(instance_occlsegm_lib.aug.augment_object_data(
                [obj_datum], random_state, fit_output=False,
                scale=(0.5, 2.0)))
            img = obj_datum['img']
            lbl = obj_datum['lbl']
        return img, lbl

    def download(self):
        instance_occlsegm_lib.data.download(
            url='https://drive.google.com/uc?id=1w-awS2auCKi4dJW-aWmPGQ4l4ak-bldK',  # NOQA
            path=osp.join(DATASETS_DIR, 'dataset_jsk_v3_20170614_shelfv1_tote_annotated.tgz'),  # NOQA
            md5='0aae1fb86ada6ee91ad95fdeaff935c2',
            postprocess=instance_occlsegm_lib.data.extractall,
        )
        instance_occlsegm_lib.data.download(
            url='https://drive.google.com/uc?id=1PzZ3jPfn-qX_OlzpoWKFaC4m0zlOE7AN',  # NOQA
            path=osp.join(DATASETS_DIR, 'dataset_jsk_v3_20170715_shelfv2_annotated.tgz'),  # NOQA
            md5='dc919229357928a6bb2e8db2352e6945',
            postprocess=instance_occlsegm_lib.data.extractall,
        )
        instance_occlsegm_lib.data.download(
            url='https://drive.google.com/uc?id=1NGoTRjpghOqkhEmSPd5aDU1ccmRgRLss',  # NOQA
            path=osp.join(DATASETS_DIR, 'dataset_jsk_v3_20170723_tote_annotated.tgz'),  # NOQA
            md5='1436019ae9f117db50d1a490a4e387d2',
            postprocess=instance_occlsegm_lib.data.extractall,
        )


class ItemDataDataset(chainer.dataset.DatasetMixin):

    class_names = None

    def __init__(self, split, item_data_dir=None, ret_load_item_data=None,
                 from_scratch=False, do_aug=True, aug_level='all',
                 n_item=None, skip_known=True, verbose=True):
        self.split = split
        self._do_aug = do_aug
        assert aug_level in ['all', 'object', 'image']
        self._aug_level = aug_level
        # item_data
        if item_data_dir is not None:
            object_names, self.item_data = load_item_data(
                item_data_dir, skip_known=skip_known)
        elif ret_load_item_data is not None:
            object_names, self.item_data = ret_load_item_data
        else:
            raise ValueError
        if n_item:
            n_obj = len(object_names)
            keep_labels = np.random.permutation(n_obj)[:n_item] + 1
            object_names = [object_names[l - 1] for l in keep_labels]
            item_data = []
            for d in self.item_data:
                if d['label'] in keep_labels:
                    new_label = np.where(keep_labels == d['label'])[0][0] + 1
                    d['label'] = new_label
                    item_data.append(d)
            self.item_data = item_data
        # class_names
        self.class_names = ['__background__'] + object_names[:]
        # label conversion for self._dataset
        self.class_id_map = {}
        for cls_id_from, cls_name in enumerate(class_names_arc2017):
            if cls_id_from == 41:
                cls_id_to = 0
            else:
                try:
                    cls_id_to = self.class_names.index(cls_name)
                except ValueError:
                    cls_id_to = -1
            self.class_id_map[cls_id_from] = cls_id_to
            if verbose:
                print('{:02d} -> {:02d}'.format(cls_id_from, cls_id_to))
        if verbose:
            for cls_id, cls_name in enumerate(self.class_names):
                print('{:02d}: {}'.format(cls_id, cls_name))
        # shelf templates
        self._dataset = None
        if not from_scratch:
            self._dataset = JskARC2017DatasetV3(split=split)
        self.shelf_img, self.shelf_lbl = get_shelf_data()

    def __len__(self):
        return 100  # fixed size

    def get_example(self, i):
        if self.split == 'valid':
            random_state = np.random.RandomState(i)
        else:
            random_state = np.random.RandomState()

        if self._dataset is not None and random_state.randint(0, 10) < 7:
            index_shelf = random_state.randint(0, len(self._dataset))
            img_shelf, lbl_shelf = self._dataset[index_shelf]
        else:
            index_shelf = np.random.randint(0, len(self.shelf_img))
            img_shelf = self.shelf_img[index_shelf]
            lbl_shelf = self.shelf_lbl[index_shelf]

        mask_labeled = lbl_shelf != -1
        y1, x1, y2, x2 = instance_occlsegm_lib.image.masks_to_bboxes(
            [mask_labeled])[0]
        img_shelf = img_shelf[y1:y2, x1:x2]
        lbl_shelf = lbl_shelf[y1:y2, x1:x2]

        scale = (500. * 500.) / (img_shelf.shape[0] * img_shelf.shape[1])
        scale = math.sqrt(scale)
        img_shelf = cv2.resize(img_shelf, None, None, fx=scale, fy=scale)
        lbl_shelf = cv2.resize(lbl_shelf, None, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_NEAREST)

        # transform class ids (label values)
        img_shelf_converted = img_shelf.copy()
        lbl_shelf_converted = lbl_shelf.copy()
        for cls_id_from, cls_id_to in self.class_id_map.items():
            mask = lbl_shelf == cls_id_from
            if cls_id_to == -1:
                img_shelf_converted[mask] = 0
            cls_id_to = 0 if cls_id_to == -1 else cls_id_to
            lbl_shelf_converted[mask] = cls_id_to
        img_shelf = img_shelf_converted
        lbl_shelf = lbl_shelf_converted
        lbl_shelf[lbl_shelf == 41] = 0  # __shelf__ -> __background__

        item_data = copy.deepcopy(self.item_data)
        random_state.shuffle(item_data)
        if self._do_aug and self._aug_level in ['all', 'object']:
            item_data = instance_occlsegm_lib.aug.augment_object_data(
                item_data, random_state)
        stacked = instance_occlsegm_lib.aug.stack_objects(
            img_shelf, lbl_shelf, item_data,
            region_label=0, random_state=random_state)
        if self._do_aug and self._aug_level in ['all', 'image']:
            stacked = next(instance_occlsegm_lib.aug.augment_object_data(
                [stacked], random_state, fit_output=False))
        return stacked['img'], stacked['lbl']


class ClassIdMappedDataset(chainer.dataset.DatasetMixin):

    class_names = None

    def __init__(self, dataset, class_names, class_id_map):
        self._dataset = dataset
        self.split = dataset.split
        self.class_names = class_names
        self.class_id_map = class_id_map

    def __len__(self):
        return len(self._dataset)

    def get_example(self, i):
        img, lbl = self._dataset.get_example(i)
        lbl_converted = lbl.copy()
        for cls_id_from, cls_id_to in self.class_id_map.items():
            lbl_converted[lbl == cls_id_from] = cls_id_to
        lbl = lbl_converted
        return img, lbl
