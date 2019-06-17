import glob
import os
import os.path as osp

import chainer
import numpy as np

import instance_occlsegm_lib


ROOT_DIR = osp.expanduser('~/data/instance_occlsegm_lib/synthetic2d')


def _download():
    instance_occlsegm_lib.data.download(
        url='https://drive.google.com/uc?id=1NKhJWs6FyTmFEPZWgatdrBIhMa_sy2Tk',  # NOQA
        path=osp.join(ROOT_DIR, '20180204_splits.zip'),
        md5='50450cf47afaa83ac0ebf7a9ce1ce43f',
        postprocess=instance_occlsegm_lib.data.extractall,
        quiet=True,
    )


def _load_npz(npz_file, augmentation=False):
    data = np.load(npz_file)

    img = data['img']
    bboxes = data['bboxes']
    labels = data['labels']
    masks = data['masks']

    # crop
    mask_bin = ~(img == 0).all(axis=2)
    y1, x1, y2, x2 = instance_occlsegm_lib.image.masks_to_bboxes([mask_bin])[0]
    img = img[y1:y2, x1:x2]
    bboxes[:, 0] -= y1
    bboxes[:, 2] -= y1
    bboxes[:, 1] -= x1
    bboxes[:, 3] -= x1
    masks = masks[:, y1:y2, x1:x2]

    bboxes[:, 0] = np.clip(bboxes[:, 0], 0, img.shape[0])
    bboxes[:, 2] = np.clip(bboxes[:, 2], 0, img.shape[0])
    bboxes[:, 1] = np.clip(bboxes[:, 1], 0, img.shape[1])
    bboxes[:, 3] = np.clip(bboxes[:, 3], 0, img.shape[1])

    if augmentation:
        data = next(instance_occlsegm_lib.aug.augment_object_data(
            [dict(img=img, lbls=masks)], scale=(0.5, 2.0), fit_output=False,
        ))
        img = data['img']
        masks = data['lbls']

        masks_vo = np.isin(masks, (1, 2))  # visible + occluded
        keep = masks_vo.sum(axis=(1, 2)) > 0

        bboxes = instance_occlsegm_lib.image.masks_to_bboxes(masks_vo[keep])
        labels = labels[keep]
        masks = masks[keep]

    bboxes = bboxes.astype(np.float32)
    labels = labels - 1  # skip __background__
    masks = masks.astype(np.int32)

    return img, bboxes, labels, masks


class ARC2017OcclusionVideoDataset(chainer.dataset.DatasetMixin):

    def __init__(self, video_dir, class_names):
        self._npz_files = sorted(glob.glob(osp.join(video_dir, '*.npz')))
        self.class_names = class_names

    def __len__(self):
        return len(self._npz_files)

    def get_example(self, i):
        npz_file = self._npz_files[i]
        return _load_npz(npz_file, augmentation=False)


class ARC2017OcclusionDataset(chainer.dataset.DatasetMixin):

    def __init__(self, split, do_aug=False):
        assert split in ['train', 'test']
        _download()

        self._do_aug = do_aug

        class_names_file = osp.join(
            ROOT_DIR, '20180204_splits/class_names.txt')
        class_names = [c.strip() for c in open(class_names_file)]
        class_names = np.asarray(class_names)
        class_names.setflags(write=0)
        self.class_names = class_names[1:]  # skip __background__

        self.video_dirs = []
        self._npz_files = []
        split_dir = osp.join(ROOT_DIR, '20180204_splits', split)
        for video_dir in sorted(os.listdir(split_dir)):
            video_dir = osp.join(split_dir, video_dir)
            self.video_dirs.append(video_dir)
            self._npz_files.extend(
                sorted(glob.glob(osp.join(video_dir, '*.npz'))))

    def get_video_datasets(self):
        datasets = []
        for video_dir in self.video_dirs:
            dataset = ARC2017OcclusionVideoDataset(video_dir, self.class_names)
            datasets.append(dataset)
        return datasets

    def __len__(self):
        return len(self._npz_files)

    def get_example(self, i):
        npz_file = self._npz_files[i]
        return _load_npz(npz_file, augmentation=self._do_aug)


if __name__ == '__main__':
    import chainer_mask_rcnn as mrcnn

    dataset = ARC2017OcclusionDataset('train', do_aug=True)
    video = ARC2017OcclusionVideoDataset(
        dataset.video_dirs[0], dataset.class_names)

    def visualize_func(dataset, index):
        print('Index: %08d' % index)

        img, bboxes, labels, masks = dataset[index]

        captions = ['%d: %s' % (l, dataset.class_names[l]) for l in labels]
        viz1 = mrcnn.utils.draw_instance_bboxes(
            img, bboxes, labels + 1, n_class=41, masks=masks == 1,
            captions=captions)
        viz2 = mrcnn.utils.draw_instance_bboxes(
            img, bboxes, labels + 1, n_class=41, masks=masks == 2,
            captions=captions)

        return np.hstack([img, viz1, viz2])

    print('split: train')
    print('video_dir: %s' % dataset.video_dirs[0])
    instance_occlsegm_lib.datasets.view_dataset(video, visualize_func)
