#!/usr/bin/env python

import argparse
import os.path as osp

import cv2
import jsk_recognition_utils
import rospkg

import jsk_arc2017_common


PKG_DIR = rospkg.RosPack().get_path('jsk_arc2017_common')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--tile-shape', default='5x8',
                        help='Default: 5x8')
    args = parser.parse_args()

    tile_shape = map(int, args.tile_shape.split('x'))

    obj_names = jsk_arc2017_common.get_object_names()

    imgs = []
    data_dir = osp.join(PKG_DIR, 'data')
    for i, obj in enumerate(obj_names):
        obj_id = i + 1
        img_file = osp.join(data_dir, 'objects', obj, 'top.jpg')
        img = cv2.imread(img_file)

        # put obj_id
        height, width = img.shape[:2]
        x1, y1, x2, y2 = 10, 10, width - 10, height - 10
        img = img[y1:y2, x1:x2]
        cv2.putText(img, '%2d' % obj_id, (0, 60),
                    cv2.cv.CV_FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 3)

        imgs.append(img)

    img_viz = jsk_recognition_utils.get_tile_image(imgs, tile_shape=tile_shape)
    out_file = osp.join(PKG_DIR, 'data/others',
                        'object_list_{0}x{1}.jpg'.format(*tile_shape))
    cv2.imwrite(out_file, img_viz)
    print('==> Wrote file: %s' % out_file)


if __name__ == '__main__':
    main()
