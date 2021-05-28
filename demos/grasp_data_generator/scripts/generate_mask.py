#!/usr/bin/env python

import argparse
import cv2
import numpy as np
import os
import os.path as osp
import scipy.ndimage
from sklearn.decomposition import PCA
import yaml

import chainer
from chainer.backends import cuda
import fcn


filepath = osp.dirname(osp.realpath(__file__))
mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434],
                    dtype=np.float32)


def main(object_dir):
    parser = argparse.ArgumentParser()
    parser.add_argument('--visualize', '-v', action='store_true')
    parser.add_argument('--grasp', '-g', action='store_true')
    parser.add_argument('--no-cnn', action='store_true')
    parser.add_argument('--gpu', default=0, type=int)
    args = parser.parse_args()
    visualize = args.visualize
    grasp = args.grasp

    objectnames = os.listdir(object_dir)
    if not args.no_cnn:
        model = fcn.models.FCN8sAtOnce(n_class=2)
        modelfile = osp.join(filepath, '../data/models/model_00010000.npz')
        chainer.serializers.load_npz(modelfile, model)
        if args.gpu >= 0:
            chainer.cuda.get_device_from_id(args.gpu).use()
            model.to_gpu()

    for objectname in objectnames:
        if osp.isfile(osp.join(object_dir, objectname)):
            continue
        print('object: {}'.format(objectname))
        for d in os.listdir(osp.join(object_dir, objectname)):
            savedir = osp.join(object_dir, objectname, d)

            imgpath = osp.join(savedir, 'rgb.png')
            img = cv2.imread(imgpath)[:, :, ::-1]

            # mask
            if args.no_cnn:
                mask = generate_mask(img, visualize)
            else:
                mask = generate_mask_with_fcn(img, model, visualize)
            maskpath = osp.join(savedir, 'mask.png')
            cv2.imwrite(maskpath, mask)

            if grasp:
                # single arm grasp img
                singlearm_grasp_img = generate_singlearm_grasp_img(
                    mask, visualize, img)
                single_img_viz = img.copy()
                sg_img_viz = np.repeat(
                    singlearm_grasp_img[:, :, np.newaxis], 3, 2)
                sg_img_viz = sg_img_viz * np.array([1, 0, 0])
                single_img_viz = \
                    (0.3 * single_img_viz + 0.7 * sg_img_viz).astype(np.uint8)
                singlearm_grasp_imgpath = osp.join(
                    savedir, 'singlearm_grasp_calc.png')
                single_img_vizpath = osp.join(
                    savedir, 'singlearm_grasp_visualize.png')
                cv2.imwrite(singlearm_grasp_imgpath, singlearm_grasp_img)
                cv2.imwrite(single_img_vizpath, single_img_viz[:, :, ::-1])

                # single arm grasp img
                dualarm_grasp_img, pc0 = generate_dualarm_grasp_img(
                    mask, visualize, img)
                dual_img_viz = img.copy()
                dg_img_viz = np.repeat(
                    dualarm_grasp_img[:, :, np.newaxis], 3, 2)
                dg_img_viz = dg_img_viz * np.array([1, 0, 0])
                dual_img_viz = \
                    (0.3 * dual_img_viz + 0.7 * dg_img_viz).astype(np.uint8)
                dualarm_grasp_imgpath = osp.join(
                    savedir, 'dualarm_grasp_calc.png')
                dual_img_vizpath = osp.join(
                    savedir, 'dualarm_grasp_visualize.png')
                cv2.imwrite(dualarm_grasp_imgpath, dualarm_grasp_img)
                cv2.imwrite(dual_img_vizpath, dual_img_viz[:, :, ::-1])

                pc0_path = osp.join(savedir, 'pc0.yaml')
                with open(pc0_path, 'w') as f:
                    f.write(yaml.dump(pc0.tolist()))


def generate_mask(img, visualize=False):
    mask = img.sum(axis=2) > (2 * img.shape[2])
    print(mask.shape)
    if visualize:
        negative_mask = ~mask
        img_viz = img.copy()
        img_viz[negative_mask] = np.array([255, 0, 0])
        cv2.imshow('mask', img_viz[:, :, ::-1])
        cv2.waitKey(0)
    mask = mask.astype(np.uint8) * 255
    return mask


def prepare_for_fcn(img):
    img = img[:, :, ::-1].astype(np.float32) - mean_bgr
    img = img.transpose((2, 0, 1))
    imgs = cuda.to_gpu(img[None])
    return imgs


def generate_mask_with_fcn(img, model, visualize=False):
    imgs = prepare_for_fcn(img)
    with chainer.no_backprop_mode(), chainer.using_config('train', False):
        model(imgs)
        xp = cuda.get_array_module(imgs)
        lbl_pred = cuda.to_cpu(xp.argmax(model.score.array, axis=1)[0])
    mask_pred = lbl_pred.astype(np.uint8) * 255
    if visualize:
        negative_mask = ~(mask_pred > 0)
        img_viz = img.copy()
        img_viz[negative_mask] = np.array([255, 0, 0])
        cv2.imshow('mask', img_viz[:, :, ::-1])
        cv2.waitKey(0)
    return mask_pred


def generate_singlearm_grasp_img(mask, visualize=False, img=None, sigma=10):
    com = calculate_com(mask)
    grasp_img = np.zeros(mask.shape)
    grasp_img[com] = 255
    grasp_img = generate_grasp_img(
        mask, grasp_img, visualize, img, sigma)
    return grasp_img


def generate_dualarm_grasp_img(mask, visualize=False, img=None, sigma=10):
    gpt0, gpt1, pc0 = calculate_grasp_pair(mask)
    grasp_img = np.zeros(mask.shape)
    for gpt in [gpt0, gpt1]:
        grasp_img[gpt] = 255
    grasp_img = generate_grasp_img(
        mask, grasp_img, visualize, img, sigma)
    return grasp_img, pc0


def generate_grasp_img(mask, grasp_img, visualize=False, img=None, sigma=10):
    grasp_img = scipy.ndimage.filters.gaussian_filter(grasp_img, sigma=sigma)
    grasp_img = grasp_img / grasp_img.max() * 255
    grasp_img = grasp_img.astype(np.uint8)
    if visualize:
        img_viz = img.copy()
        grasp_img_viz = np.repeat(grasp_img[:, :, np.newaxis], 3, 2)
        grasp_img_viz = grasp_img_viz * np.array([1, 0, 0])
        img_viz = (0.3 * img_viz + 0.7 * grasp_img_viz).astype(np.uint8)
        cv2.imshow('grasp', img_viz[:, :, ::-1])
        cv2.waitKey(0)
    return grasp_img


def calculate_grasp_pair(mask):
    indices = np.column_stack(np.where(mask > 0))
    pca = PCA(n_components=2)
    pca.fit(indices)
    pc0 = pca.components_[0]
    pc0_mean = pca.mean_
    trans_indices = pca.fit_transform(indices)
    min_pc0 = trans_indices[:, 0].min()
    max_pc0 = trans_indices[:, 0].max()
    length = max_pc0 - min_pc0
    grasp_point0 = pc0_mean - 0.3 * length * pc0
    grasp_point0 = tuple(grasp_point0.astype(np.int32))
    grasp_point1 = pc0_mean + 0.3 * length * pc0
    grasp_point1 = tuple(grasp_point1.astype(np.int32))
    return grasp_point0, grasp_point1, pc0


def calculate_com(mask):
    # center of mask
    y_array, x_array = np.where(mask > 0)
    y = int(np.round(np.average(y_array)))
    x = int(np.round(np.average(x_array)))
    return y, x


if __name__ == '__main__':
    object_dir = osp.join(filepath, '../data/objects')
    main(object_dir)
