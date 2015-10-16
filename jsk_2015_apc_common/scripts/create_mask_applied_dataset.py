#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import re

import cv2
from jsk_recognition_utils import bounding_rect_of_mask


parser = argparse.ArgumentParser()
parser.add_argument('container_path')
parser.add_argument('-O', '--output')
args = parser.parse_args()

container_path = args.container_path
output_dir = args.output or os.path.abspath(container_path + '_mask_applied')

if not os.path.exists(output_dir):
    print('creating output directory: {}'.format(output_dir))
    os.mkdir(output_dir)

categs = os.listdir(container_path)

os.chdir(container_path)

for categ in categs:
    os.chdir(categ)

    print('processing category: {}'.format(categ))
    files = os.listdir('.')
    img_files = filter(lambda x: re.match('^N\d*?_\d*?.jpg', x), files)
    print('found {} images'.format(len(img_files)))

    categ_output_dir = os.path.join(output_dir, categ)
    if not os.path.exists(categ_output_dir):
        os.mkdir(categ_output_dir)

    for img_file in img_files:
        base, _ = os.path.splitext(img_file)
        mask_file = os.path.join('masks', base + '_mask.pbm')
        img = cv2.imread(img_file)
        mask = cv2.imread(mask_file, 0)
        applied = bounding_rect_of_mask(img, ~mask)
        cv2.imwrite(os.path.join(output_dir, categ, img_file), applied)

    os.chdir('..')

os.chdir('..')
