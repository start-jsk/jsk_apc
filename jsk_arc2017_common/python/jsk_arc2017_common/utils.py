import collections
import cPickle as pickle
import hashlib
import json
import math
import os.path as osp

import cv2
import numpy as np
import skimage.color
import skimage.io
import skimage.transform

import rospkg

from .data import get_object_images


PKG_DIR = rospkg.RosPack().get_path('jsk_arc2017_common')


def get_tile_shape(num, ratio_hw=None):
    if ratio_hw:
        for y_num in xrange(num):
            x_num = int(y_num / ratio_hw)
            if x_num * y_num > num:
                return y_num, x_num
    else:
        x_num = int(math.sqrt(num))
        y_num = 0
        while x_num * y_num < num:
            y_num += 1
        return y_num, x_num


def mask_to_rect(mask):
    where = np.argwhere(mask)
    (y1, x1), (y2, x2) = where.min(0), where.max(0) + 1
    return y1, x1, y2, x2


def centerize(src, shape, margin_color=None, return_mask=False):
    """Centerize image for specified image size
    Parameters
    ----------
    src: numpy.ndarray
        Image to centerize
    shape: tuple of int
        Image shape (height, width) or (height, width, channel)
    margin_color: numpy.ndarray
        Color to be filled in the blank.
    return_mask: numpy.ndarray
        Mask for centerized image.
    """
    if src.shape[:2] == shape[:2]:
        if return_mask:
            return src, np.ones(shape[:2], dtype=bool)
        else:
            return src
    if len(shape) != src.ndim:
        shape = list(shape) + [src.shape[2]]
    centerized = np.zeros(shape, dtype=src.dtype)
    if margin_color:
        centerized[:, :] = margin_color

    src_h, src_w = src.shape[:2]
    scale_h, scale_w = 1. * shape[0] / src_h, 1. * shape[1] / src_w
    scale = min(scale_h, scale_w)
    dtype = src.dtype
    src = skimage.transform.rescale(src, scale, preserve_range=True)
    src = src.astype(dtype)

    ph, pw = 0, 0
    h, w = src.shape[:2]
    dst_h, dst_w = shape[:2]
    if h < dst_h:
        ph = (dst_h - h) // 2
    if w < dst_w:
        pw = (dst_w - w) // 2

    mask = np.zeros(shape[:2], dtype=bool)
    mask[ph:ph + h, pw:pw + w] = True
    centerized[ph:ph + h, pw:pw + w] = src
    if return_mask:
        return centerized, mask
    else:
        return centerized


def _tile(imgs, shape, dst):
    """Tile images which have same size.
    Parameters
    ----------
    imgs: numpy.ndarray
        Image list which should be tiled.
    shape: tuple of int
        Tile shape.
    dst:
        Image to put the tile on.
    """
    y_num, x_num = shape
    tile_w = imgs[0].shape[1]
    tile_h = imgs[0].shape[0]
    if dst is None:
        if len(imgs[0].shape) == 3:
            dst = np.zeros((tile_h * y_num, tile_w * x_num, 3), dtype=np.uint8)
        else:
            dst = np.zeros((tile_h * y_num, tile_w * x_num), dtype=np.uint8)
    for y in range(y_num):
        for x in range(x_num):
            i = x + y * x_num
            if i < len(imgs):
                y1 = y * tile_h
                y2 = (y + 1) * tile_h
                x1 = x * tile_w
                x2 = (x + 1) * tile_w
                dst[y1:y2, x1:x2] = imgs[i]
    return dst


def tile(imgs, shape=None, dst=None, margin_color=None):
    """Tile images which have different size.
    Parameters
    ----------
    imgs:
        Image list which should be tiled.
    shape:
        The tile shape.
    dst:
        Image to put the tile on.
    margin_color: numpy.ndarray
        Color to be filled in the blank.
    """
    if shape is None:
        shape = get_tile_shape(len(imgs))

    # get max tile size to which each image should be resized
    max_h, max_w = np.inf, np.inf
    for img in imgs:
        max_h = min(max_h, img.shape[0])
        max_w = min(max_w, img.shape[1])

    # tile images
    is_color = False
    for i, img in enumerate(imgs):
        if img.ndim >= 3:
            is_color = True

        if is_color and img.ndim == 2:
            img = skimage.color.gray2rgb(img)
        if is_color and img.shape[2] == 4:
            img = img[:, :, :3]

        img = skimage.util.img_as_ubyte(img)

        img = centerize(img, (max_h, max_w, 3), margin_color)
        imgs[i] = img
    return _tile(imgs, shape, dst)


def visualize_container(container_id, contents, container_file, orders=None,
                        alpha=0.6, font_scale=5.5, thickness=4):
    if not isinstance(contents, collections.Sequence):
        raise TypeError('contents must be a sequence')
    if orders is not None and not isinstance(orders, collections.Sequence):
        raise TypeError('orders must be a sequence')

    img_container = skimage.io.imread(container_file) / 255.
    img_container = (img_container * alpha) + (1. - alpha)
    img_container = (img_container * 255).astype(np.uint8)

    if contents:
        ratio_hw = 1. * img_container.shape[0] / img_container.shape[1]
        tile_shape = get_tile_shape(len(contents), ratio_hw)

        object_imgs = get_object_images()
        if orders is not None:
            for obj_name, img_obj in object_imgs.items():
                if obj_name in orders:
                    center = img_obj.shape[1] // 2, img_obj.shape[0] // 2
                    radius = min(center)
                    cv2.circle(img_obj, center, radius, (255, 0, 0), thickness)

        imgs = [object_imgs[obj] for obj in contents]
        img_tiled = tile(imgs, shape=tile_shape)
        img_tiled = centerize(img_tiled, img_container.shape)

        masks = [np.ones(img.shape[:2], dtype=np.uint8) * 255 for img in imgs]
        mask_tiled = tile(masks, shape=tile_shape)
        mask_tiled = centerize(mask_tiled, img_container.shape[:2])
        y1, x1, y2, x2 = mask_to_rect(mask_tiled)

        assert mask_tiled.shape == img_tiled.shape[:2]
        img_tiled = img_tiled[y1:y2, x1:x2]
        mask_tiled = mask_tiled[y1:y2, x1:x2]
        assert mask_tiled.shape == img_tiled.shape[:2]

        img_tiled = centerize(img_tiled, img_container.shape)
        mask_tiled = centerize(mask_tiled, img_container.shape[:2])
        mask_tiled = mask_tiled == 255
        mask_tiled[np.all(img_tiled == 0, axis=2)] = False
        img_container[mask_tiled] = img_tiled[mask_tiled]

    font_face = cv2.FONT_HERSHEY_PLAIN
    size, baseline = cv2.getTextSize(
        container_id, font_face, font_scale, thickness)
    cv2.putText(img_container, container_id,
                (img_container.shape[1] - size[0],
                 img_container.shape[0] - size[1] + baseline),
                font_face, font_scale, color=(0, 255, 0), thickness=thickness)

    return img_container


memos = {}


def memoize(key=None):
    def _memoize(func):
        def func_wrapper(*args, **kwargs):
            if key:
                contents = pickle.dumps(
                    {'func': func.func_code.co_code,
                     'contents': key(*args, **kwargs)})
            else:
                contents = pickle.dumps(
                    {'func': func.func_code.co_code,
                     'args': args, 'kwargs': kwargs})
            sha1 = hashlib.sha1(contents).hexdigest()
            if sha1 in memos:
                return memos[sha1]
            res = func(*args, **kwargs)
            if len(memos) > 50:
                memos.popitem()
            else:
                memos[sha1] = res
            return res
        return func_wrapper
    return _memoize


@memoize(key=lambda filename, order_file:
         (json.load(open(filename)), order_file))
def visualize_item_location(filename, order_file=None):
    item_location = json.load(open(filename))

    # orders
    orders = []
    if order_file:
        data_order = json.load(open(order_file))
        for order in data_order['orders']:
            orders.extend(order['contents'])

    imgs_top = []
    # tote
    tote = item_location['tote']
    img_container = visualize_container(
        'tote', tote['contents'], orders=orders,
        container_file=osp.join(PKG_DIR, 'data/objects/tote/top.jpg'),
        font_scale=11, thickness=8)
    imgs_top.append(img_container)
    # bin
    for bin_ in sorted(item_location['bins'], reverse=True):  # C, B, A
        img_container = visualize_container(
            bin_['bin_id'], bin_['contents'], orders=orders,
            container_file=osp.join(PKG_DIR, 'data/objects/bin/top.jpg'),
            font_scale=9, thickness=7)
        imgs_top.append(img_container)
    # visualize
    img_top = tile(imgs_top, shape=(1, len(imgs_top)))

    if not item_location['boxes']:
        return img_top

    # box
    imgs_box = []
    for box in item_location['boxes']:
        img_container = visualize_container(
            box['size_id'], box['contents'], orders=orders,
            container_file=osp.join(PKG_DIR, 'data/objects/box/top.jpg'))
        imgs_box.append(img_container)
    img_box = tile(imgs_box, shape=(1, len(imgs_box)))

    scale = 1. * img_box.shape[1] / img_top.shape[1]
    img_top = cv2.resize(img_top, None, None, fx=scale, fy=scale)
    img = np.vstack([img_top, img_box])
    return img


@memoize(key=lambda filename: json.load(open(filename)))
def visualize_order(filename):
    data = json.load(open(filename))
    imgs = []
    for order in data['orders']:
        img = visualize_container(
            order['size_id'], order['contents'],
            container_file=osp.join(PKG_DIR, 'data/objects/box/top.jpg'))
        imgs.append(img)
    img = tile(imgs, shape=(1, 3))
    return img
