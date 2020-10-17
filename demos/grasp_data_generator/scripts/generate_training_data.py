#!/usr/bin/env python

import argparse
import datetime
import imgaug.augmenters as iaa
from imgaug.parameters import Deterministic
import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_pydot import write_dot
import numpy as np
import os
import os.path as osp
import random
import scipy.misc
import shutil
import skimage.color
import yaml


filepath = osp.dirname(osp.realpath(__file__))


def main(
        datadir, n_data, visualize=False, no_human=False,
        instance=False, label_yaml=None
):
    time = datetime.datetime.now()
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    savedir = osp.join(datadir, 'training_data', timestamp)
    if not osp.exists(savedir):
        os.makedirs(savedir)

    background_imgs = []
    for background_name in ['tote', 'cardboard', 'shelf']:
        background_path = osp.join(
            datadir, 'background', background_name, 'top.jpg')
        background_img = scipy.misc.imread(background_path)
        background_img = scipy.misc.imresize(background_img, (480, 640))
        background_imgs.append(background_img)

    object_dir = osp.join(datadir, 'objects')
    if label_yaml is None:
        label_names = sorted(os.listdir(object_dir))
        label_names = ['__background__'] + label_names
        with open(osp.join(savedir, 'label_names.yaml'), 'w') as f:
            yaml.safe_dump(label_names, f)
    else:
        shutil.copyfile(label_yaml, osp.join(savedir, 'label_names.yaml'))
        with open(label_yaml, 'r') as f:
            label_names = yaml.load(f)
    object_imgpath = []
    for objectname in label_names[1:]:
        data = []
        for d in os.listdir(osp.join(object_dir, objectname)):
            object_savedir = osp.join(object_dir, objectname, d)
            rgb_path = osp.join(object_savedir, 'rgb.png')
            mask_path = osp.join(object_savedir, 'mask.png')
            pc0_path = osp.join(object_savedir, 'pc0.yaml')
            if no_human:
                sg_path = osp.join(object_savedir, 'singlearm_grasp_calc.png')
                dg_path = osp.join(object_savedir, 'dualarm_grasp_calc.png')
            else:
                sg_path = osp.join(object_savedir, 'singlearm_grasp.png')
                dg_path = osp.join(object_savedir, 'dualarm_grasp.png')
            data.append([
                rgb_path,
                mask_path,
                sg_path,
                dg_path,
                pc0_path,
            ])
        object_imgpath.append(data)

    for i in range(0, n_data):
        print('num: {}'.format(i))
        generate_train_data(
            background_imgs, object_imgpath,
            label_names, savedir, visualize,
            instance
        )


def generate_train_data(
        background_imgs, object_imgpath, label_names,
        savedir, visualize=False, instance=False, random_state=1234,
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
    if instance:
        object_names = random.sample(label_names[1:], object_num)
        random.seed(random.randrange(1, random_state))
        for i in range(0, object_num):
            object_names.append(random.choice(label_names[1:]))
    else:
        object_names = random.sample(label_names[1:], object_num)
    random.shuffle(object_names)
    for ins_label, object_name in enumerate(object_names):
        label = label_names.index(object_name)
        labels.append(label)
        object_paths = object_imgpath[label-1]
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
            visualize, resize_ratio=0.6, random_state=random_state)


def generate_train_data_step(
        savedir, object_img, background_img, mask_img,
        rgb_img, whole_ins_img, label_img, sg_img, dg_img, graph,
        ins_label, label, single_grasp, dual_grasp, pc0,
        ins_imgs, labels, object_rotation, graspable_ins_label,
        mask_sums, init_mask_sums, object_name, label_names,
        visualize, resize_ratio, random_state,

):

    st = lambda x: iaa.Sometimes(0.3, x)  # NOQA

    # rotate degree in counter clockwise
    fliplr = np.random.uniform() > 0.5
    if fliplr:
        pc0[0] = -1 * pc0[0]
    theta = np.arctan(pc0[0] / pc0[1])

    if theta >= 0:
        phi = np.pi / 2 - theta
    else:
        phi = - (np.pi / 2) - theta
    phi_angle = (phi * 180) / np.pi
    aug_rotate_angle = np.random.uniform(-180, 180)

    color_aug = iaa.Sequential(
        [
            st(iaa.InColorspace(
                'HSV',
                children=iaa.WithChannels([1, 2],
                                          iaa.Multiply([0.5, 2])))),
            iaa.WithChannels([0, 1], iaa.Multiply([1, 1.5])),
        ],
        random_order=False,
        random_state=random_state)
    # rotate in clockwise
    aug = iaa.Sequential(
        [
            iaa.Fliplr(1 if fliplr else 0),
            iaa.Affine(
                order=0,
                scale=(0.5, 1.2),
                cval=0,
                rotate=-1 * aug_rotate_angle,
                mode='constant'),
        ],
        random_order=False,
        random_state=random_state)

    # # resize
    object_img = scipy.misc.imresize(object_img, resize_ratio)
    mask_img = scipy.misc.imresize(
        mask_img, resize_ratio, interp='nearest')
    single_grasp = scipy.misc.imresize(single_grasp, resize_ratio)
    dual_grasp = scipy.misc.imresize(dual_grasp, resize_ratio)

    # imgaug
    object_img = color_aug.augment_image(object_img)
    aug_dest = aug.to_deterministic()
    aug_dest[0].order = Deterministic(1)
    object_img = aug_dest.augment_image(object_img)
    single_grasp = aug_dest.augment_image(single_grasp)
    dual_grasp = aug_dest.augment_image(dual_grasp)
    aug_dest[0].order = Deterministic(0)
    mask_img = aug_dest.augment_image(mask_img)

    # mask_img
    mask_img = mask_img > 0
    object_img[~mask_img] = 0
    mask_sums.append(mask_img.astype(np.int32).sum())
    init_mask_sums.append(mask_sums[-1])

    # translation
    width, height, _ = object_img.shape
    x = random.randint(0, rgb_img.shape[0] - width)
    y = random.randint(0, rgb_img.shape[1] - height)

    # rgb_img
    tmp_img = rgb_img[x:x+width, y:y+height]
    tmp_img[mask_img] = 0
    object_img = object_img + tmp_img
    rgb_img[x:x+width, y:y+height] = object_img

    # whole_instance_img
    tmp_whole_ins_img = whole_ins_img[x:x+width, y:y+height]
    tmp_whole_ins_img[mask_img] = ins_label
    whole_ins_img[x:x+width, y:y+height] = tmp_whole_ins_img
    whole_ins_img = whole_ins_img.astype(np.int32)

    for i, tmp_ins_img in enumerate(ins_imgs):
        tmp_ins_img[np.logical_and(tmp_ins_img, whole_ins_img != i)] = 2

    # instance imgs
    ins_img = np.zeros(background_img.shape[:2], dtype=np.int32)
    ins_img[x:x+width, y:y+height] = mask_img.astype(np.int32)
    ins_imgs.append(ins_img[None])

    # label_img
    label_img[whole_ins_img == ins_label] = label
    label_img = label_img.astype(np.int32)

    # sg_img
    tmp_single_grasp = sg_img[x:x+width, y:y+height]
    tmp_single_grasp[mask_img] = 0
    single_grasp[~mask_img] = 0
    single_grasp = single_grasp + tmp_single_grasp
    sg_img[x:x+width, y:y+height] = single_grasp

    # dg_img
    tmp_dual_grasp = dg_img[x:x+width, y:y+height]
    tmp_dual_grasp[mask_img] = 0
    dual_grasp[~mask_img] = 0
    dual_grasp = dual_grasp + tmp_dual_grasp
    dg_img[x:x+width, y:y+height] = dual_grasp

    # grasp mask
    graspable_ins_label.append(ins_label)
    tmp_graspable_ins_label = []
    graspable_mask = np.zeros(background_img.shape[:2])

    # object rotation
    rotate_angle = phi_angle + aug_rotate_angle
    if rotate_angle >= 0:
        rotate_angle = rotate_angle % 180
    else:
        rotate_angle = (rotate_angle % 180) - 180
    if rotate_angle > 90:
        rotate_angle = rotate_angle - 180
    elif rotate_angle < -90:
        rotate_angle = rotate_angle + 180
    object_rotation.append(float(rotate_angle))

    # graph viz
    graph.add_node('{0}_{1}'.format(object_name, ins_label))
    print('{0}_{1}'.format(object_name, ins_label))
    occluded_node = []
    for i, (mask_sum, init_mask_sum) \
            in enumerate(zip(mask_sums, init_mask_sums)):
        ins_mask = whole_ins_img == i
        ins_mask = ins_mask.astype(np.int32)
        if ins_mask.sum() > 0.9 * mask_sum:
            if i in graspable_ins_label:
                graspable_mask[whole_ins_img == i] = 1
                tmp_graspable_ins_label.append(i)
        # elif ins_mask.sum() < 0.3 * init_mask_sum:
        #     graph.remove_node(
        #         '{0}_{1}'.format(label_names[labels[i]], i))
        else:
            occluded_node.append(i)
            mask_sums[i] = ins_mask.sum()
    edges = []
    for i in occluded_node:
        edge = ('{0}_{1}'.format(label_names[labels[i]], i),
                '{0}_{1}'.format(object_name, ins_label))
        edges.append(edge)
    graph.add_edges_from(edges)

    graspable_ins_label = tmp_graspable_ins_label
    graspable_mask = graspable_mask.astype(np.bool)
    sg_img[~graspable_mask] = 0
    dg_img[~graspable_mask] = 0

    # visualization
    if visualize:
        plt.figure()
        nx.draw_networkx(graph)
        plt.show()

        # label viz
        label_viz = skimage.color.label2rgb(
            label_img, rgb_img, bg_label=0)
        scipy.misc.imshow(label_viz)

        # graspable viz
        graspable_mask_viz = skimage.color.label2rgb(
            graspable_mask, rgb_img, bg_label=0)
        scipy.misc.imshow(graspable_mask_viz)

        # single grasp viz
        sg_img_viz = np.repeat(
            sg_img[:, :, np.newaxis], 3, 2)
        sg_img_viz = sg_img_viz * np.array([1, 0, 0])
        sg_img_viz = 0.3 * rgb_img + 0.7 * sg_img_viz
        sg_img_viz = sg_img_viz.astype(np.int32)
        scipy.misc.imshow(sg_img_viz)

        # dual grasp viz
        dg_img_viz = np.repeat(
            dg_img[:, :, np.newaxis], 3, 2)
        dg_img_viz = dg_img_viz * np.array([1, 0, 0])
        dg_img_viz = 0.3 * rgb_img + 0.7 * dg_img_viz
        dg_img_viz = dg_img_viz.astype(np.int32)
        scipy.misc.imshow(dg_img_viz)

        # occluded viz
        occluded_label = np.zeros(
            rgb_img.shape[:2], dtype=np.int32)
        for label, ins_img in zip(labels, ins_imgs):
            occluded_label[ins_img[0] == 2] = label
        occluded_img_viz = skimage.color.label2rgb(
            occluded_label, rgb_img, bg_label=0)
        scipy.misc.imshow(occluded_img_viz)

    # save
    savedirs = os.listdir(savedir)
    savedirs = [d for d in savedirs if osp.isdir(osp.join(savedir, d))]
    savedirs = list(map(int, savedirs))
    savedirs.append(0)
    maxdir = max(savedirs)
    datadir = osp.join(savedir, '{0:05d}'.format(maxdir+1))
    if not osp.exists(datadir):
        os.makedirs(datadir)
    # path
    # datapath = osp.join(datadir, 'data.npz')
    rgbpath = osp.join(datadir, 'rgb.png')
    labelpath = osp.join(datadir, 'label_img.npz')
    inspath = osp.join(datadir, 'ins_imgs.npz')
    sgpath = osp.join(datadir, 'single_grasp.png')
    dgpath = osp.join(datadir, 'dual_grasp.png')
    graphpath = osp.join(datadir, 'graph.dot')
    labelspath = osp.join(datadir, 'labels.yaml')
    object_rotationpath = osp.join(datadir, 'object_rotation.yaml')
    # save
    scipy.misc.imsave(rgbpath, rgb_img)
    scipy.misc.imsave(sgpath, sg_img)
    scipy.misc.imsave(dgpath, dg_img)
    np.savez_compressed(labelpath, label_img=label_img)
    np.savez_compressed(
        inspath,
        ins_imgs=np.concatenate(ins_imgs, axis=0).astype(np.int32))
    write_dot(graph, graphpath)
    with open(labelspath, 'w+') as f:
        f.write(yaml.dump(labels))
    with open(object_rotationpath, 'w+') as f:
        f.write(yaml.dump(object_rotation))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--visualize', '-v', action='store_true')
    parser.add_argument('--num', '-n', default=100, type=int)
    parser.add_argument('--no-human', action='store_true')
    parser.add_argument('--instance', action='store_true')
    parser.add_argument('--label-names', default=None, type=str)
    args = parser.parse_args()

    datadir = osp.join(filepath, '../data')
    main(datadir, args.num, args.visualize, args.no_human,
         args.instance, args.label_names)
