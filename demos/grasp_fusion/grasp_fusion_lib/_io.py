import math
import os
import os.path as osp
import warnings

import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D  # NOQA
import numpy as np
import PIL.Image
import skimage.io


def load_pcd(filename):
    """Load xyz PCD file.

    Parameters
    ----------
    filename: str
        PCD filename.
    """
    points = []
    n_points = None
    with open(filename, 'r') as f:
        for line in f.readlines():
            if line.startswith('#'):
                continue

            meta_fields = [
                'VERSION',
                'FIELDS',
                'SIZE',
                'TYPE',
                'COUNT',
                'WIDTH',
                'HEIGHT',
                'VIEWPOINT',
                'POINTS',
                'DATA',
            ]
            meta = line.strip().split(' ')
            meta_header, meta_contents = meta[0], meta[1:]
            if meta_header == 'FIELDS':
                assert meta_contents == ['x', 'y', 'z']
            elif meta_header == 'POINTS':
                n_points = int(meta_contents[0])
            if meta_header in meta_fields:
                continue

            x, y, z = map(float, line.split(' '))
            points.append((x, y, z))

    points = np.array(points)

    if n_points is not None:
        assert len(points) == n_points
        assert points.shape[1] == 3

    return points


def load_off(filename):
    """Load OFF file.

    Parameters
    ----------
    filename: str
        OFF filename.
    """
    with open(filename, 'r') as f:
        assert 'OFF' in f.readline()

        verts, faces = [], []
        n_verts, n_faces = None, None
        for line in f.readlines():
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            if n_verts is None and n_faces is None:
                n_verts, n_faces, _ = map(int, line.split(' '))
            elif len(verts) < n_verts:
                verts.append([float(v) for v in line.split(' ')])
            else:
                faces.append([int(v) for v in line.split(' ')[1:]])
        verts = np.array(verts, dtype=np.float64)
        faces = np.array(faces, dtype=np.int64)

    return verts, faces


def load_vtk(filename):
    points = None
    with open(filename) as f:
        state = None
        index = 0
        for line in f:
            if line.startswith('POINTS '):
                _, n_points, dtype = line.split()
                n_points = int(n_points)
                points = np.empty((n_points, 3), dtype=dtype)
                state = 'POINTS'
                continue
            elif line.startswith('VERTICES '):
                warnings.warn('VERTICES is not currently supported.')
                _, _, n_vertices = line.split()
                n_vertices = int(n_vertices)
                # vertices = np.empty((n_vertices, 2), dtype=np.int64)
                state = 'VERTICES'
                continue

            if state == 'POINTS':
                x, y, z = line.split()
                xyz = np.array([x, y, z], dtype=np.float64)
                points[index] = xyz
                index += 1
            elif state == 'VERTICES':
                # TODO(wkentaro): support this.
                pass
    return points


def dump_off(filename, verts, faces):
    with open(filename, 'w') as f:
        f.write('OFF\n')

        n_vert = len(verts)
        n_face = len(faces)
        n_edge = 0
        f.write('{} {} {}\n'.format(n_vert, n_face, n_edge))

        for vert in verts:
            f.write(' '.join(map(str, vert)) + '\n')

        for face in faces:
            f.write(' '.join([str(len(face))] + map(str, face)) + '\n')


def dump_obj(filename, verts, faces):
    """Dump mesh data to obj file."""
    with open(filename, 'w') as f:
        # write vertices
        f.write('g\n# %d vertex\n' % len(verts))
        for vert in verts:
            f.write('v %f %f %f\n' % tuple(vert))

        # write faces
        f.write('# %d faces\n' % len(faces))
        for face in faces:
            f.write('f %d %d %d\n' % tuple(face))


def _get_tile_shape(num):
    x_num = int(math.sqrt(num))
    y_num = 0
    while x_num * y_num < num:
        y_num += 1
    return x_num, y_num


def tileplot(plot, args_lst, shape=None):
    if shape is None:
        shape = _get_tile_shape(len(args_lst))

    x_num, y_num = shape
    for i, args in enumerate(args_lst):
        plt.subplot(x_num, y_num, i + 1)
        plot(*args)


def imgplot(img, title=None):
    if img.ndim == 2:
        if np.issubdtype(img.dtype, np.floating):
            cmap = 'jet'
        else:
            cmap = 'gray'
        plt.imshow(img, cmap=cmap)
    else:
        plt.imshow(img)
    if title is not None:
        plt.title(title)


def tileimg(imgs, shape=None):
    return tileplot(imgplot, [(img,) for img in imgs], shape=shape)


def plot_tile(imgs, shape=None):
    warnings.warn('`plot_tile` is deprecated. Please use `tileimg` instead.')
    return tileplot(imgplot, [(img,) for img in imgs], shape=shape)


def meshplot(verts, faces=None, ax=None, **kwargs):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection='3d')
    if faces is not None:
        tri = Poly3DCollection(verts[faces, :])
        tri.set_alpha(0.2)
        tri.set_color('grey')
        ax.add_collection3d(tri)
    ax.plot(verts[:, 0], verts[:, 1], verts[:, 2], **kwargs)


def plot_mesh(verts, faces):
    warnings.warn('`plot_mesh` is deprecated. Please use `meshplot` instead.')
    return meshplot(verts, faces)


def show():
    return plt.show()


def imshow(img, window_name=None):
    if img.ndim == 3:
        img = img[:, :, ::-1]
    return cv2.imshow(window_name, img)


def imread(*args, **kwargs):
    return skimage.io.imread(*args, **kwargs)


def lbread(lbl_file):
    return np.asarray(PIL.Image.open(lbl_file)).astype(np.int32)


def imsave(*args, **kwargs):
    if len(args) >= 1:
        fname = args[0]
    else:
        fname = kwargs['fname']
    dirname = osp.dirname(fname)
    if dirname and not osp.exists(dirname):
        os.makedirs(dirname)
    return skimage.io.imsave(*args, **kwargs)


def waitkey(time=0):
    return cv2.waitKey(time)


def waitKey(time=0):
    warnings.warn('`waitKey` is deprecated. Please use `waitkey` instead.')
    return waitkey(time=time)
