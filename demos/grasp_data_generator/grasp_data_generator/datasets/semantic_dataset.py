import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import os.path as osp
import yaml

import chainer
import imgaug.augmenters as iaa
from imgaug.parameters import Deterministic
import scipy.misc
from sklearn.model_selection import train_test_split

from chainercv.visualizations import vis_semantic_segmentation


filepath = osp.dirname(osp.realpath(__file__))


class SemanticRealAnnotatedDataset(chainer.dataset.DatasetMixin):

    def __init__(
            self, split='train', random_state=1234,
            imgaug=True, test_size=0.1,
    ):
        self.split = split
        ids = sorted(os.listdir(self.data_dir))
        ids = [d for d in ids if osp.isdir(osp.join(self.data_dir, d))]
        ids_train, ids_val = train_test_split(
            ids, test_size=test_size, random_state=random_state)
        ids_train = sorted(ids_train)
        ids_val = sorted(ids_val)
        self._ids = {'all': ids, 'train': ids_train, 'val': ids_val}

        with open(osp.join(self.data_dir, 'label_names.yaml')) as f:
            self.label_names = yaml.load(f)
        self.label_names = ['background'] + self.label_names

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
        cls_ids = np.array(
            yaml.load(open(osp.join(datum_dir, 'labels.yaml'))),
            dtype=np.int32)
        H, W, _ = img.shape
        label = - np.ones((H, W), dtype=np.int32)
        for ins_lbl, cls_id in zip(ins_label, cls_ids):
            label[ins_lbl == 1] = cls_id

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
            label = aug.augment_image(label)
        img = img.astype(np.float32)
        label = label + 1
        return img, label

    def visualize(self, i):
        img, label = self.get_example(i)
        img = img.transpose((2, 0, 1))
        ax, legend_handles = vis_semantic_segmentation(
            img, label, label_names=self.label_names, alpha=0.8)
        ax.legend(handles=legend_handles, bbox_to_anchor=(1, 1), loc=2)
        plt.show()


class SemanticRealAnnotatedDatasetV1(SemanticRealAnnotatedDataset):
    data_dir = osp.join(filepath, '../../data/evaluation_data/20181231_194442')


class SemanticRealAnnotatedDatasetV2(SemanticRealAnnotatedDataset):
    data_dir = osp.join(filepath, '../../data/evaluation_data/20190107_142843')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--aug', '-a', action='store_true')
    parser.add_argument('--dataset', choices=['v1', 'v2'],
                        default='v1', help='Dataset version')
    args = parser.parse_args()

    if args.dataset == 'v1':
        dataset = SemanticRealAnnotatedDatasetV1(split='all', imgaug=args.aug)
    elif args.dataset == 'v2':
        dataset = SemanticRealAnnotatedDatasetV2(split='all', imgaug=args.aug)
    else:
        raise ValueError(
            'Given dataset is not supported: {}'.format(args.dataset))

    for i in range(0, len(dataset)):
        dataset.visualize(i)
