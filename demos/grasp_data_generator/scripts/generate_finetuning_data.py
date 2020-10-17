#!/usr/bin/env python

import argparse
import datetime
import networkx as nx
import numpy as np
import os
import os.path as osp
import random
import scipy.misc
import shutil
import yaml

from generate_mask import generate_dualarm_grasp_img
from generate_mask import generate_singlearm_grasp_img
from generate_training_data import generate_train_data_step


filepath = osp.dirname(osp.realpath(__file__))


def main(
        datadir, n_data, visualize, instance=False,
        label_yaml=None, only_sampling=False
):
    time = datetime.datetime.now()
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    savedir = osp.join(datadir, 'finetuning_data', timestamp)
    if not osp.exists(savedir):
        os.makedirs(savedir)

    background_imgs = []
    for background_name in ['tote', 'cardboard', 'shelf']:
        background_path = osp.join(
            datadir, 'background', background_name, 'top.jpg')
        background_img = scipy.misc.imread(background_path)
        background_img = scipy.misc.imresize(background_img, (480, 640))
        background_imgs.append(background_img)

    original_object_dir = osp.join(datadir, 'objects')
    if label_yaml is None:
        label_names = sorted(os.listdir(original_object_dir))
        label_names = ['__background__'] + label_names
        with open(osp.join(savedir, 'label_names.yaml'), 'w') as f:
            yaml.safe_dump(label_names, f)
    else:
        shutil.copyfile(label_yaml, osp.join(savedir, 'label_names.yaml'))
        with open(label_yaml, 'r') as f:
            label_names = yaml.load(f)

    original_object_imgpath = []
    for objectname in label_names[1:]:
        data = []
        for d in os.listdir(osp.join(original_object_dir, objectname)):
            object_savedir = osp.join(original_object_dir, objectname, d)
            rgb_path = osp.join(object_savedir, 'rgb.png')
            mask_path = osp.join(object_savedir, 'mask.png')
            sg_path = osp.join(object_savedir, 'singlearm_grasp_calc.png')
            dg_path = osp.join(object_savedir, 'dualarm_grasp_calc.png')
            pc0_path = osp.join(object_savedir, 'pc0.yaml')
            data.append([
                rgb_path,
                mask_path,
                sg_path,
                dg_path,
                pc0_path,
            ])
        original_object_imgpath.append(data)

    sampling_object_dir = osp.join(datadir, 'sampling_data')
    sampling_object_imgpath = []
    for i in range(1, len(label_names)):
        sampling_object_imgpath.append([])
    for date in os.listdir(sampling_object_dir):
        for i, objectname in enumerate(label_names[1:]):
            tmp_dir = osp.join(sampling_object_dir, date, objectname)
            data = []
            for d in os.listdir(tmp_dir):
                object_savedir = osp.join(tmp_dir, d)
                rgb_path = osp.join(object_savedir, 'input_image.png')
                mask_path = osp.join(object_savedir, 'object_mask.png')
                input_mask_path = osp.join(object_savedir, 'input_mask.png')
                sampled_path = osp.join(
                    object_savedir, 'grasp_mask.png')
                with open(osp.join(object_savedir, 'result.txt')) as f:
                    result = f.read()
                with open(osp.join(object_savedir, 'grasping_way.txt')) as f:
                    grasping_way = f.read()
                data.append([
                    rgb_path,
                    mask_path,
                    input_mask_path,
                    sampled_path,
                    result,
                    grasping_way,
                ])
            sampling_object_imgpath[i].extend(data)

    for i in range(0, n_data):
        print('num: {}'.format(i))
        generate_train_data(
            background_imgs, original_object_imgpath, sampling_object_imgpath,
            label_names, savedir, visualize, instance, only_sampling,
        )


def generate_train_data(
        background_imgs, original_object_imgpath, sampling_object_imgpath,
        label_names, savedir, visualize=False, instance=False,
        only_sampling=False, sampling_ratio=0.75, random_state=1234,
):
    background_img = random.choice(background_imgs)
    rgb_img = background_img.copy()
    label_img = np.zeros(background_img.shape[:2], dtype=np.int32)
    whole_ins_img = -1 * np.ones(background_img.shape[:2], dtype=np.int32)
    ins_imgs = []
    labels = []
    object_rotation = []

    sg_img = np.zeros(background_img.shape[:2])
    dg_img = np.zeros(background_img.shape[:2])
    graph = nx.DiGraph()

    mask_sums = []
    init_mask_sums = []
    graspable_ins_label = []

    object_num = random.randint(1, min(len(label_names) - 1, 10))
    object_names = random.sample(label_names[1:], object_num)
    object_num = random.randint(1, min(len(label_names) - 1, 10))
    if instance:
        object_names = random.sample(label_names[1:], object_num)
        random.jumpahead(1)
        for i in range(0, object_num):
            object_names.append(random.choice(label_names[1:]))
    else:
        object_names = random.sample(label_names[1:], object_num)
    random.shuffle(object_names)
    for ins_label, object_name in enumerate(object_names):
        label = label_names.index(object_name)
        labels.append(label)
        if only_sampling:
            is_sampling = True
        else:
            is_sampling = random.uniform(0, 1) < sampling_ratio

        if is_sampling:
            resize_ratio = 1.2
            object_paths = sampling_object_imgpath[label-1]
            object_path = random.choice(object_paths)
            rgb_path, mask_path, input_mask_path = object_path[:3]
            sampled_path, result, grasping_way = object_path[3:]
            object_img = scipy.misc.imread(rgb_path)
            mask_img = scipy.misc.imread(mask_path)
            input_mask_img = scipy.misc.imread(
                input_mask_path, flatten=True)
            y_indices, x_indices = np.where(input_mask_img > 0)
            y_max, y_min = y_indices.max(), y_indices.min()
            x_max, x_min = x_indices.max(), x_indices.min()
            object_img = object_img[y_min:y_max, x_min:x_max]
            mask_img = mask_img[y_min:y_max, x_min:x_max]
            if result == 'success':
                sample_grasp = scipy.misc.imread(sampled_path, flatten=True)
                sample_grasp = sample_grasp[y_min:y_max, x_min:x_max]
                sample_grasp = sample_grasp.astype(np.uint8)
                if grasping_way == 'dual':
                    single_grasp = np.zeros(mask_img.shape, dtype=np.uint8)
                    dual_grasp, pc0 = generate_dualarm_grasp_img(
                            sample_grasp, sigma=8)
                else:
                    single_grasp = generate_singlearm_grasp_img(
                            sample_grasp, sigma=8)
                    dual_grasp = np.zeros(mask_img.shape, dtype=np.uint8)
                    pc0 = np.array([0, 1])
            else:
                single_grasp = np.zeros(mask_img.shape, dtype=np.uint8)
                dual_grasp = np.zeros(mask_img.shape, dtype=np.uint8)
                pc0 = np.array([0, 1])
        else:
            resize_ratio = 0.6
            object_paths = original_object_imgpath[label-1]
            rgb_path, mask_path, sg_path, dg_path, pc0_path = \
                random.choice(object_paths)
            object_img = scipy.misc.imread(rgb_path)
            mask_img = scipy.misc.imread(mask_path)
            single_grasp = scipy.misc.imread(sg_path)
            dual_grasp = scipy.misc.imread(dg_path)
            pc0 = yaml.load(open(pc0_path, 'r'))

        generate_train_data_step(
            savedir, object_img, background_img, mask_img,
            rgb_img, whole_ins_img, label_img, sg_img, dg_img, graph,
            ins_label, label, single_grasp, dual_grasp, pc0,
            ins_imgs, labels, object_rotation, graspable_ins_label,
            mask_sums, init_mask_sums, object_name, label_names,
            visualize, resize_ratio=resize_ratio, random_state=random_state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--visualize', '-v', action='store_true')
    parser.add_argument('--num', '-n', default=100, type=int)
    parser.add_argument('--instance', action='store_true')
    parser.add_argument('--label-names', default=None, type=str)
    parser.add_argument('--only-sampling', action='store_true')
    args = parser.parse_args()

    datadir = osp.join(filepath, '../data')
    main(datadir, args.num, args.visualize, args.instance,
         args.label_names, args.only_sampling)
