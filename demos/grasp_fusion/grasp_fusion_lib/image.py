import warnings

import cv2
import matplotlib
import numpy as np
import scipy
import six
import skimage.color
import skimage.transform
import skimage.util


def colorize_depth(depth, min_value=None, max_value=None, dtype=np.uint8):
    """Colorize depth image with JET colormap."""
    min_value = np.nanmin(depth) if min_value is None else min_value
    max_value = np.nanmax(depth) if max_value is None else max_value
    if np.isinf(min_value) or np.isinf(max_value):
        warnings.warn('Min or max value for depth colorization is inf.')
    if max_value == min_value:
        eps = np.finfo(depth.dtype).eps
        max_value += eps
        min_value -= eps

    colorized = depth.copy()
    nan_mask = np.isnan(colorized)
    colorized[nan_mask] = 0
    colorized = 1. * (colorized - min_value) / (max_value - min_value)
    colorized = matplotlib.cm.jet(colorized)[:, :, :3]
    if dtype == np.uint8:
        colorized = (colorized * 255).astype(dtype)
    else:
        assert np.issubdtype(dtype, np.floating)
        colorized = colorized.astype(dtype)
    colorized[nan_mask] = (0, 0, 0)
    return colorized


def colorize_heatmap(heatmap):
    """Colorize heatmap which ranges 0 to 1.

    Parameters
    ----------
    heatmap: numpy.ndarray
        Heatmap which ranges 0 to 1.
    """
    if not (0 <= heatmap.min() <= 1):
        raise ValueError('Heatmap min value must range from 0 to 1')
    if not (0 <= heatmap.max() <= 1):
        raise ValueError('Heatmap max value must range from 0 to 1')
    return colorize_depth(heatmap, min_value=0, max_value=1)


def overlay_color_on_mono(img_color, img_mono, alpha=0.5):
    """Overlay color image on mono.

    Parameters
    ----------
    img_color: numpy.ndarray, (H, W, 3)
    img_mono: numpy.ndarray, (H, W, 3) or (H, W)
    alpha: float
        Alpha value for color.

    Returns
    -------
    dst: numpy.ndarray
        Output image.
    """
    # RGB -> Gray
    if img_mono.ndim == 3:
        img_mono = skimage.color.rgb2gray(img_mono)
    img_mono = skimage.color.gray2rgb(img_mono)

    img_mono = skimage.util.img_as_float(img_mono)
    img_color = skimage.util.img_as_float(img_color)

    dst = alpha * img_color + (1 - alpha) * img_mono
    dst = (dst * 255).astype(np.uint8)
    return dst


def label_colormap(n_label=256):
    """Colormap for specified number of labels.

    Parameters
    ----------
    n_label: int
        Number of labels and colors.
    """
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    cmap = np.zeros((n_label, 3))
    for i in six.moves.range(0, n_label):
        id = i
        r, g, b = 0, 0, 0
        for j in six.moves.range(0, 8):
            r = np.bitwise_or(r, (bitget(id, 0) << 7 - j))
            g = np.bitwise_or(g, (bitget(id, 1) << 7 - j))
            b = np.bitwise_or(b, (bitget(id, 2) << 7 - j))
            id = (id >> 3)
        cmap[i, 0] = r
        cmap[i, 1] = g
        cmap[i, 2] = b
    cmap = cmap.astype(np.float32) / 255
    return cmap


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
    src = cv2.resize(src, None, None, fx=scale, fy=scale)

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


def _get_tile_shape(num):
    import math
    x_num = int(math.sqrt(num))
    y_num = 0
    while x_num * y_num < num:
        y_num += 1
    return x_num, y_num


def tile(
        imgs,
        shape=None,
        dst=None,
        margin_color=None,
        boundary=False,
        boundary_color=(255, 255, 255),
        boundary_thickness=3,
        ):
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
    imgs = imgs[:]

    if shape is None:
        shape = _get_tile_shape(len(imgs))

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
        if boundary:
            cv2.rectangle(img, (1, 1), (img.shape[1] - 1, img.shape[0] - 1),
                          boundary_color, thickness=boundary_thickness)
        imgs[i] = img
    return _tile(imgs, shape, dst)


def get_text_color(color):
    if color[0] * 0.299 + color[1] * 0.587 + color[2] * 0.114 > 170:
        return (0, 0, 0)
    return (255, 255, 255)


def label2rgb(lbl, img=None, label_names=None, n_labels=None,
              alpha=0.5, thresh_suppress=0):
    if label_names is None:
        if n_labels is None:
            n_labels = lbl.max() + 1  # +1 for bg_label 0
    else:
        if n_labels is None:
            n_labels = len(label_names)
        else:
            assert n_labels == len(label_names)
    cmap = label_colormap(n_labels)
    cmap = (cmap * 255).astype(np.uint8)

    lbl_viz = cmap[lbl]

    if img is not None:
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_gray = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
        lbl_viz = alpha * lbl_viz + (1 - alpha) * img_gray
        lbl_viz = lbl_viz.astype(np.uint8)

    np.random.seed(1234)

    mask_unlabeled = lbl == -1
    lbl_viz[mask_unlabeled] = \
        np.random.random(size=(mask_unlabeled.sum(), 3)) * 255

    if label_names is None:
        return lbl_viz

    for label in np.unique(lbl):
        if label == -1:
            continue  # unlabeled

        mask = lbl == label
        if 1. * mask.sum() / mask.size < thresh_suppress:
            continue
        mask = (mask * 255).astype(np.uint8)
        y, x = scipy.ndimage.center_of_mass(mask)
        y, x = map(int, [y, x])

        if lbl[y, x] != label:
            Y, X = np.where(mask)
            point_index = np.random.randint(0, len(Y))
            y, x = Y[point_index], X[point_index]

        text = label_names[label]
        font_face = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        text_size, baseline = cv2.getTextSize(
            text, font_face, font_scale, thickness)

        color = get_text_color(lbl_viz[y, x])
        cv2.putText(lbl_viz, text,
                    (x - text_size[0] // 2, y),
                    font_face, font_scale, color, thickness)
    return lbl_viz


def mask_to_bbox(mask):
    """Convert binary mask image to bounding box.

    Parameters
    ----------
    mask: numpy.ndarray, (H, W), bool
        Boolean mask.

    Returns
    -------
    bbox: tuple of int, (4,)
        x1, y1, x2, y2.
    """
    warnings.warn(
        'mask_to_bbox is deprecated. Use masks_to_bbox '
        'which returns array of (y1, x1, y2, x2).'
    )
    assert mask.dtype == bool
    where = np.argwhere(mask)
    (y1, x1), (y2, x2) = where.min(0), where.max(0) + 1
    return x1, y1, x2, y2


def masks_to_bboxes(masks):
    """Convert binary mask image to bounding box.

    Parameters
    ----------
    masks: numpy.ndarray, (N, H, W), bool
        Boolean masks.

    Returns
    -------
    bboxes: tuple of int, (N, 4)
        Each bbox represents (y1, x1, y2, x2).
    """
    bboxes = np.zeros((len(masks), 4), dtype=np.int32)
    for i, mask in enumerate(masks):
        assert mask.dtype == bool
        where = np.argwhere(mask)
        (y1, x1), (y2, x2) = where.min(0), where.max(0) + 1
        bboxes[i] = (y1, x1, y2, x2)
    return bboxes


def mask_to_lbl(mask, label):
    """Convert mask to label image."""
    lbl = np.empty(mask.shape, dtype=np.int32)
    lbl[mask] = label
    lbl[~mask] = -1
    return lbl


def resize(
        image,
        height=None,
        width=None,
        fy=None,
        fx=None,
        size=None,
        interpolation=cv2.INTER_LINEAR,
):
    """Resize image with cv2 resize function.

    Parameters
    ----------
    image: numpy.ndarray
        Source image.
    height, width: None or int
        Target height or width.
    fy, fx: None or float
        Target height or width scale.
    size: None or int or float
        Target image size.
    interpolation: int
        Interpolation flag. (default: cv2.INTER_LINEAR == 1)
    """
    hw_ratio = 1. * image.shape[0] / image.shape[1]  # h / w
    if height is not None or width is not None:
        if height is None:
            height = int(round(hw_ratio * width))
        elif width is None:
            width = int(round(1 / hw_ratio * height))
        assert fy is None
        assert fx is None
        assert size is None
        return cv2.resize(image, (width, height), interpolation=interpolation)
    elif fy is not None or fx is not None:
        if fy is None:
            fy = fx
        elif fx is None:
            fx = fy
        assert height is None
        assert width is None
        assert size is None
    elif size is not None:
        assert height is None
        assert width is None
        assert fy is None
        assert fx is None
        fx = fy = np.sqrt(1. * size / (image.shape[0] * image.shape[1]))
    else:
        raise ValueError
    return cv2.resize(
        image, None, None, fx=fx, fy=fy, interpolation=interpolation)


def resize_mask(mask, *args, **kwargs):
    """Resize mask in float space.

    Parameters
    ----------
    mask: numpy.ndarray
        Source mask whose size must be (H, W) and has bool dtype.

    See grasp_fusion_lib.image.resize for other parameters.
    """
    assert mask.dtype == bool
    assert mask.ndim == 2
    mask = mask.astype(float)
    mask = resize(mask, *args, **kwargs)
    mask = mask > 0.5
    return mask


def resize_lbl(lbl, *args, **kwargs):
    """Resize lbl in channel space.

    Parameters
    ----------
    lbl: numpy.ndarray
        Source mask whose size must be (H, W) and has int32 dtype.

    See grasp_fusion_lib.image.resize for other parameters.
    """
    assert lbl.dtype == np.int32
    assert lbl.ndim == 2
    # [label -> onehot] -> [resize] -> [onehot -> label]
    min_value = lbl.min()
    lbl -= min_value  # shift to make the min_value to be 0
    lbl_score = (np.arange(lbl.max() + 1) == lbl[..., None]).astype(np.float32)
    lbl_score = resize(lbl_score, *args, **kwargs)
    lbl_score = np.atleast_3d(lbl_score)
    lbl = np.argmax(lbl_score, axis=2)
    lbl = lbl.astype(np.int32)
    lbl += min_value  # restore the min_value
    return lbl
