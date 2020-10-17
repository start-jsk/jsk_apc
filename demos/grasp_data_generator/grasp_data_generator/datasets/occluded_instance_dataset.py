import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import os.path as osp
import yaml

import chainer
from chainercv.utils.mask.mask_to_bbox import mask_to_bbox
import imgaug.augmenters as iaa
from imgaug.parameters import Deterministic
import scipy.misc
from sklearn.model_selection import train_test_split

from grasp_data_generator.visualizations \
    import vis_occluded_instance_segmentation


filepath = osp.dirname(osp.realpath(__file__))


class OIRealAnnotatedDataset(chainer.dataset.DatasetMixin):

    def __init__(
            self, split='train', random_state=1234,
            imgaug=True, clip=False, test_size=0.1,
    ):
        self.split = split
        ids = sorted(os.listdir(self.data_dir))
        ids = [d for d in ids if osp.isdir(osp.join(self.data_dir, d))]
        ids_train, ids_val = train_test_split(
            ids, test_size=test_size, random_state=random_state)
        ids_train = sorted(ids_train)
        ids_val = sorted(ids_val)
        self._ids = {'all': ids, 'train': ids_train, 'val': ids_val}
        self.clip = clip

        with open(osp.join(self.data_dir, 'label_names.yaml')) as f:
            self.label_names = yaml.load(f)

        st = lambda x: iaa.Sometimes(0.3, x)  # NOQA
        if imgaug:
            self.color_aug = iaa.Sequential(
                [
                    st(iaa.InColorspace(
                        'HSV',
                        children=iaa.WithChannels([1, 2],
                                                  iaa.Multiply([0.5, 2])))),
                    iaa.WithChannels([0, 1], iaa.Multiply([1, 1.5])),
                ],
                random_order=False,
                random_state=random_state)
            self.aug = iaa.Sequential(
                [
                    iaa.Affine(
                        cval=0,
                        order=0,
                        rotate=0,
                        mode='constant'),
                    iaa.Fliplr(0)
                ],
                random_order=False,
                random_state=random_state)

        else:
            self.aug = None

    def __len__(self):
        return len(self._ids[self.split])

    def get_example(self, i):
        data_id = self._ids[self.split][i]
        datum_dir = osp.join(self.data_dir, data_id)
        img = scipy.misc.imread(osp.join(datum_dir, 'rgb.png'))
        ins_label = np.load(osp.join(datum_dir, 'ins_imgs.npz'))['ins_imgs']
        ins_label = ins_label.transpose((1, 2, 0))  # CHW -> HWC
        label = np.array(
            yaml.load(open(osp.join(datum_dir, 'labels.yaml'))),
            dtype=np.int32)
        if self.aug:
            aug_rotate_angle = np.random.uniform(-180, 180)
            fliplr = np.random.uniform() > 0.5
            img = self.color_aug.augment_image(img)
            aug = self.aug.to_deterministic()
            aug[0].order = Deterministic(1)
            aug[0].rotate = Deterministic(-1 * aug_rotate_angle)
            aug[1].p = Deterministic(1 if fliplr else 0)
            img = aug.augment_image(img)
            aug[0].order = Deterministic(0)
            ins_label = aug.augment_image(ins_label)

        if self.clip:
            fg_indices = np.where(~np.all(img == 0, axis=2))
            y_min, y_max = fg_indices[0].min(), fg_indices[0].max()
            x_min, x_max = fg_indices[1].min(), fg_indices[1].max()
            img = img[y_min:y_max, x_min:x_max]
            ins_label = ins_label[y_min:y_max, x_min:x_max]

        # HWC -> CHW
        img = img.transpose((2, 0, 1)).astype(np.float32)
        ins_label = ins_label.transpose((2, 0, 1))
        return img, ins_label, label

    def visualize(self, i):
        img, ins_label, label = self.get_example(i)
        bbox = mask_to_bbox(ins_label > 0)
        vis_occluded_instance_segmentation(
            img, ins_label, label, bbox, None, self.label_names)
        plt.show()


class OIRealAnnotatedDatasetV1(OIRealAnnotatedDataset):
    data_dir = osp.join(filepath, '../../data/evaluation_data/20181231_194442')


class OIRealAnnotatedDatasetV2(OIRealAnnotatedDataset):
    data_dir = osp.join(filepath, '../../data/evaluation_data/20190107_142843')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--aug', '-a', action='store_true')
    parser.add_argument('--clip', '-c', action='store_true')
    parser.add_argument('--dataset', choices=['v1', 'v2'],
                        default='v1', help='Dataset version')
    args = parser.parse_args()

    if args.dataset == 'v1':
        dataset = OIRealAnnotatedDatasetV1(
            split='all', imgaug=args.aug, clip=args.clip)
    elif args.dataset == 'v2':
        dataset = OIRealAnnotatedDatasetV2(
            split='all', imgaug=args.aug, clip=args.clip)
    else:
        raise ValueError(
            'Given dataset is not supported: {}'.format(args.dataset))

    for i in range(0, len(dataset)):
        dataset.visualize(i)
