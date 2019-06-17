import os
import os.path as osp

import chainer
from imgaug import augmenters as iaa
import matplotlib.pyplot as plt
import numpy as np
import random
import rospkg
import scipy.misc
from sklearn.model_selection import train_test_split
import yaml

filepath = osp.dirname(osp.realpath(__file__))


class StowingDataset(chainer.dataset.DatasetMixin):
    mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
    class_yamlpath = osp.join(filepath, 'data/label_names.yaml')

    def __init__(
            self,
            data_type='all',
            random_state=1234,
            resize_rate=0.5,
            test_size=0.1,
            cross_validation=False,
            loop=None,
            img_aug=False,
            with_damage=True,
            classification=False):
        assert data_type in ('train', 'val', 'all')

        # scrape dataset
        data_ids = self._get_data_ids()

        # split train/val data
        if data_type != 'all':
            if cross_validation:
                random.seed(random_state)
                random.shuffle(data_ids)
                test_num = int(len(data_ids) * test_size)
                start = loop * test_num
                end = (loop + 1) * test_num
                if len(data_ids) - end > test_num:
                    ids_train = data_ids[:start] + data_ids[end:]
                    ids_val = data_ids[start:end]
                else:
                    ids_train = data_ids[:start]
                    ids_val = data_ids[start:]
            else:
                ids_train, ids_val = train_test_split(
                    data_ids, test_size=test_size, random_state=random_state)
        if data_type == 'train':
            self.data_ids = ids_train
        elif data_type == 'val':
            self.data_ids = ids_val
        else:
            self.data_ids = data_ids
        self.resize_rate = resize_rate
        if img_aug:
            self.aug = iaa.Sequential([
                iaa.Sometimes(
                    0.3,
                    iaa.InColorspace(
                        'HSV',
                        children=iaa.WithChannels([1, 2],
                                                  iaa.Multiply([0.5, 2])))),
                iaa.Fliplr(0.5)])
        else:
            self.aug = False

        if with_damage:
            self.failure_label = np.array([
                'singlearm_drop',
                'singlearm_protrude',
                'singlearm_damage',
                'dualarm_drop',
                'dualarm_protrude',
                'dualarm_damage',
            ])
        else:
            self.failure_label = np.array([
                'singlearm_drop',
                'singlearm_protrude',
                'dualarm_drop',
                'dualarm_protrude',
            ])

        if classification:
            with open(self.class_yamlpath, 'r') as f:
                self.class_label = yaml.load(f)
        else:
            self.class_label = None

    def __len__(self):
        return len(self.data_ids)

    def img_to_datum(self, img):
        datum = img.astype(np.float32)
        datum = datum[:, :, ::-1]  # RGB -> BGR
        datum -= self.mean_bgr
        datum = datum.transpose((2, 0, 1))
        return datum

    def datum_to_img(self, blob):
        bgr = blob.transpose((1, 2, 0))
        bgr += self.mean_bgr
        rgb = bgr[:, :, ::-1]  # BGR -> RGB
        rgb = rgb.astype(np.uint8)
        return rgb

    def get_t(self, data_dir):
        label_file = osp.join(data_dir, 'label.txt')
        failure_labels = open(label_file, 'r').read().strip().split('\n')
        t = np.zeros(len(self.failure_label), dtype=np.int32)
        tmp_cond = None
        for label_name in failure_labels:
            cond, state = label_name.split('_')
            if tmp_cond is not None:
                assert cond == tmp_cond
            if state != 'success':
                if state not in ['drop', 'protrude', 'damage']:
                    print('Unknown label {0} in {1}'
                          .format(label_name, label_file))
                t[np.where(self.failure_label == label_name)] = 1.0
            tmp_cond = cond
        half_length = len(self.failure_label) / 2
        if cond == 'singlearm':
            t[half_length:] = -1
        else:
            t[:half_length] = -1
        # return t, cond
        return t

    def get_average_t(self):
        singlearm_t = []
        dualarm_t = []
        for data_id in self.data_ids:
            data_dir = osp.join(self.dataset_dir, data_id)
            t = self.get_t(data_dir).astype(np.float32)
            half_length = len(self.failure_label) / 2
            if t[0] == -1:
                dualarm_t.append(t[half_length:])
            else:
                singlearm_t.append(t[:half_length])
        singlearm_average_t = sum(singlearm_t) / len(singlearm_t)
        dualarm_average_t = sum(dualarm_t) / len(dualarm_t)
        return singlearm_average_t, dualarm_average_t

    def get_baseline_acc(self):
        acc = []
        singlearm_average_t, dualarm_average_t = self.get_average_t()
        for data_id in self.data_ids:
            data_dir = osp.join(self.dataset_dir, data_id)
            t = self.get_t(data_dir)
            half_length = len(self.failure_label) / 2
            if t[0] == -1:
                t = t[half_length:]
                average_t = dualarm_average_t
            else:
                t = t[:half_length]
                average_t = singlearm_average_t
            if all((t > 0.5) == (average_t > 0.5)):
                acc.append(1)
            else:
                acc.append(0)
        return sum(acc) / float(len(acc))

    def get_example(self, i, masked=True):
        data_id = self.data_ids[i]

        data_dir = osp.join(self.dataset_dir, data_id)
        img_file = osp.join(data_dir, 'image_rect_color.png')
        mask_file = osp.join(data_dir, 'clipped_mask_rgb.png')
        img = scipy.misc.imread(img_file, mode='RGB')
        mask = scipy.misc.imread(mask_file, mode='L')

        # apply mask
        # img[mask < 128] = (0, 0, 0)
        if self.aug:
            img = self.aug.augment_image(img)
        if masked:
            img[mask < 128] = self.mean_bgr[::-1]
        if self.resize_rate < 1:
            img = scipy.misc.imresize(img, self.resize_rate)
        datum = self.img_to_datum(img)
        t = self.get_t(data_dir)
        if self.class_label is None:
            return datum, t
        else:
            with open(osp.join(data_dir, 'target.txt'), 'r') as f:
                target = f.read().split('\n')[0]
            label_id = np.int32(self.class_label.index(target))
            return datum, t, label_id

    def visualize_example(self, i, masked=True):
        if self.class_label is None:
            datum, t = self.get_example(i, masked)
        else:
            datum, t, _ = self.get_example(i, masked)
        img = self.datum_to_img(datum)
        plt.imshow(img)
        plt.axis('off')
        plt.title(str(t.tolist()))
        plt.show()

    def _get_data_ids(self):
        data_ids = []
        for data_id in os.listdir(self.dataset_dir):
            data_id = osp.join(self.dataset_dir, data_id)
            if not osp.isdir(data_id):
                continue
            data_ids.append(data_id)
        return data_ids


class SinglearmFailureDataset(StowingDataset):
    def _get_data_ids(self):
        data_ids = []
        for data_id in self.data_ids:
            data_dir = osp.join(self.dataset_dir, data_id)
            t = self.get_t(data_dir).astype(np.float32)
            half_length = len(self.failure_label) / 2
            singlearm_t = t[:half_length]
            if any(x == 1 for x in singlearm_t):
                data_ids.append(data_id)
        return data_ids


class DualarmDatasetV1(StowingDataset):
    rospack = rospkg.RosPack()
    dataset_dir = osp.join(
        rospack.get_path('selective_dualarm_stowing'),
        'dataset/v1')


class DualarmDatasetV2(StowingDataset):
    rospack = rospkg.RosPack()
    dataset_dir = osp.join(
        rospack.get_path('selective_dualarm_stowing'),
        'dataset/v2')


class DualarmDatasetV3(StowingDataset):
    rospack = rospkg.RosPack()
    dataset_dir = osp.join(
        rospack.get_path('selective_dualarm_stowing'),
        'dataset/v3')


class DualarmDatasetV4(StowingDataset):
    rospack = rospkg.RosPack()
    dataset_dir = osp.join(
        rospack.get_path('selective_dualarm_stowing'),
        'dataset/v4')


class DualarmDatasetV5(StowingDataset):
    rospack = rospkg.RosPack()
    dataset_dir = osp.join(
        rospack.get_path('selective_dualarm_stowing'),
        'dataset/v5')
    class_yamlpath = osp.join(filepath, 'data/label_names_v5.yaml')


class SinglearmFailureDatasetV4(SinglearmFailureDataset):
    rospack = rospkg.RosPack()
    dataset_dir = osp.join(
        rospack.get_path('selective_dualarm_stowing'),
        'dataset/v4')
