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
    import vis_occluded_grasp_instance_segmentation


filepath = osp.dirname(osp.realpath(__file__))


class OIDualarmGraspDataset(chainer.dataset.DatasetMixin):

    def __init__(
            self, split='train', random_state=1234,
            imgaug=True, threshold=0.3, test_size=0.1,
            return_rotation=False,
    ):
        self.split = split
        self.threshold = threshold
        self.return_rotation = return_rotation

        ids = sorted(os.listdir(self.data_dir))
        ids = [d for d in ids if osp.isdir(osp.join(self.data_dir, d))]
        ids_train, ids_val = train_test_split(
            ids, test_size=test_size, random_state=random_state)
        ids_train = sorted(ids_train)
        ids_val = sorted(ids_val)
        self._ids = {'all': ids, 'train': ids_train, 'val': ids_val}

        with open(osp.join(self.data_dir, 'label_names.yaml')) as f:
            self.label_names = yaml.load(f)[1:]

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
        # set background as -1
        # [0, n_fg_class + 1] -> [-1, n_fg_class]
        if self.label_yaml:
            label = np.array(
                yaml.load(open(osp.join(datum_dir, 'labels.yaml'))),
                dtype=np.int32)
        else:
            label = np.load(osp.join(datum_dir, 'labels.npz'))['labels']
        label = label - 1
        single_grasp = scipy.misc.imread(
            osp.join(datum_dir, 'single_grasp.png'))
        dual_grasp = scipy.misc.imread(osp.join(datum_dir, 'dual_grasp.png'))
        if self.aug:
            aug_rotate_angle = np.random.uniform(-180, 180)
            fliplr = np.random.uniform() > 0.5
            img = self.color_aug.augment_image(img)
            aug = self.aug.to_deterministic()
            aug[0].order = Deterministic(1)
            aug[0].rotate = Deterministic(-1 * aug_rotate_angle)
            aug[1].p = Deterministic(1 if fliplr else 0)
            img = aug.augment_image(img)
            single_grasp = aug.augment_image(single_grasp)
            dual_grasp = aug.augment_image(dual_grasp)
            aug[0].order = Deterministic(0)
            ins_label = ins_label.transpose((1, 2, 0))  # CHW -> HWC
            ins_label = aug.augment_image(ins_label)
            ins_label = ins_label.transpose((2, 0, 1))  # HWC -> CHW
        else:
            aug_rotate_angle = None
            fliplr = None

        if self.return_rotation:
            orig_rotation = np.array(
                yaml.load(open(osp.join(datum_dir, 'object_rotation.yaml'))),
                dtype=np.float32)
            if fliplr:
                orig_rotation = -1 * orig_rotation
            if aug_rotate_angle is not None:
                aug_rotation = orig_rotation + aug_rotate_angle
                rotation = []
                for rotate_angle in aug_rotation:
                    if rotate_angle >= 0:
                        rotate_angle = rotate_angle % 180
                    else:
                        rotate_angle = (rotate_angle % 180) - 180
                    if rotate_angle > 90:
                        rotate_angle = rotate_angle - 180
                    elif rotate_angle < -90:
                        rotate_angle = rotate_angle + 180
                    rotation.append(rotate_angle)
                rotation = np.array(rotation, dtype=np.float32)
            else:
                rotation = orig_rotation

        img = img.transpose((2, 0, 1)).astype(np.float32)
        single_grasp = single_grasp > (self.threshold * 255)
        dual_grasp = dual_grasp > (self.threshold * 255)
        sg_mask, dg_mask = [], []
        for ins_lbl in ins_label:
            sg_msk = single_grasp.copy()
            dg_msk = dual_grasp.copy()
            sg_msk[ins_lbl != 1] = False
            dg_msk[ins_lbl != 1] = False
            sg_mask.append(sg_msk[None])
            dg_mask.append(dg_msk[None])
        sg_mask = np.concatenate(sg_mask, axis=0).astype(np.bool)
        dg_mask = np.concatenate(dg_mask, axis=0).astype(np.bool)
        if self.return_rotation:
            return img, ins_label, label, sg_mask, dg_mask, rotation
        else:
            return img, ins_label, label, sg_mask, dg_mask

    def visualize(self, i):
        if self.return_rotation:
            img, ins_label, label, sg_mask, dg_mask, rotation = \
                self.get_example(i)
            print('rotation:')
            for ins_lbl, (rot, lbl) in enumerate(zip(rotation, label)):
                print('    {0:>15}_{1:02d}: {2}'.format(
                        self.label_names[lbl], ins_lbl, rot))
        else:
            img, ins_label, label, sg_mask, dg_mask = self.get_example(i)
        bbox = mask_to_bbox(ins_label != 0)
        vis_occluded_grasp_instance_segmentation(
            img, ins_label, label, bbox, None,
            sg_mask, dg_mask, self.label_names)
        plt.show()


class OIDualarmGraspDatasetV1(OIDualarmGraspDataset):

    data_dir = osp.join(
        filepath, '../../data/training_data/', '20181101_232144')
    label_yaml = False


class OIDualarmGraspDatasetV2(OIDualarmGraspDataset):

    data_dir = osp.join(
        filepath, '../../data/training_data/', '20181121_171617')
    label_yaml = True


class FinetuningOIDualarmGraspDatasetV1(OIDualarmGraspDataset):
    data_dir = osp.join(
        filepath, '../../data/finetuning_data/', '20181217_232456')
    label_yaml = True


class FinetuningOIDualarmGraspDatasetV2(OIDualarmGraspDataset):
    data_dir = osp.join(
        filepath, '../../data/finetuning_data/', '20181220_232224')
    label_yaml = True


class FinetuningOIDualarmGraspDatasetV3(OIDualarmGraspDataset):
    data_dir = osp.join(
        filepath, '../../data/finetuning_data/', '20181226_134846')
    label_yaml = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--aug', '-a', action='store_true')
    parser.add_argument('--dataset', choices=['v1', 'v2', 'fv1', 'fv2', 'fv3'],
                        default='v2', help='Dataset version')
    parser.add_argument('--return-rotation', '-r', action='store_true')
    args = parser.parse_args()

    if args.dataset == 'v1':
        dataset = OIDualarmGraspDatasetV1(
            split='all', imgaug=args.aug,
            return_rotation=False)
    elif args.dataset == 'v2':
        dataset = OIDualarmGraspDatasetV2(
            split='all', imgaug=args.aug,
            return_rotation=args.return_rotation)
    elif args.dataset == 'fv1':
        dataset = FinetuningOIDualarmGraspDatasetV1(
            split='all', imgaug=args.aug,
            return_rotation=args.return_rotation)
    elif args.dataset == 'fv2':
        dataset = FinetuningOIDualarmGraspDatasetV2(
            split='all', imgaug=args.aug,
            return_rotation=args.return_rotation)
    elif args.dataset == 'fv3':
        dataset = FinetuningOIDualarmGraspDatasetV3(
            split='all', imgaug=args.aug,
            return_rotation=args.return_rotation)
    else:
        raise ValueError(
            'Given dataset is not supported: {}'.format(args.dataset))

    for i in range(0, len(dataset)):
        dataset.visualize(i)
