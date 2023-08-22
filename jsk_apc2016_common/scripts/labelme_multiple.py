#!/usr/bin/env python

import subprocess
import argparse
import os
import pickle
import shutil

import json

import base64
from cStringIO import StringIO
import PIL.Image
import PIL.ImageDraw
import numpy as np
import cv2

import jsk_apc2016_common


"""Script to label multiple image sets using LabelMe

Usage:
    Create polygons --> Save --> (Next image set) --> Create polygons

    If you want to delete the displayed dataset, put label '-1'.

    You may want to skip labelling some images. In that case, you can just
    close Labelme.
"""


object_names = [object_data['name'] for
                object_data in jsk_apc2016_common.get_object_data()]
object_names_dict = {}
for i, name in enumerate(object_names):
    object_names_dict[name] = i


class LabelMeMultiple(object):
    def __init__(self, prefix, labeled_dir):
        self.prefix = prefix
        self.labeled_dir = labeled_dir

        self.json_path = prefix + '.json'
        self.img_path = prefix + '.jpg'
        self.mask_path = prefix + '.pbm'
        self.pkl_path = prefix + '.pkl'
        self.input_paths = [
            self.json_path, self.img_path, self.mask_path, self.pkl_path]

    def load_data(self):
        with open(self.json_path, 'rb') as f:
            self.json_data = json.load(f)

        with open(self.pkl_path, 'rb') as f:
            self.data = pickle.load(f)

    def _convert_to_path_labeled(self, path):
        return self.labeled_dir + os.path.split(path)[1]

    def print_target_objects(self):
        for _object in self.data['objects']:
            print('id: {},  {}'.format(object_names_dict[_object], _object))

    def convert_id_to_object_name(self):
        """Modify label name of json from ids to object names
        """
        for shape in self.json_data['shapes']:
            label = shape['label']
            label_name = object_names[int(label)]

            if label_name not in self.data['objects']:
                print('wrong object id')
                return False

            shape['label'] = label_name
        with open(self.json_path, 'w') as f:
            f.write(json.dumps(self.json_data, indent=2))
        return True

    def check_delete_command(self):
        for shape in self.json_data['shapes']:
            if shape['label'] == '-1':
                print('deleting images')
                return True
        return False

    def delete_this_set(self):
        for delete_item_path in self.input_paths:
            os.remove(delete_item_path)

    def move_data_to_labeled(self):
        if not os.path.exists(self.labeled_dir):
            print('entered')
            os.makedirs(self.labeled_dir, mode=0o755)
        for input_path in self.input_paths:
            output_path = self._convert_to_path_labeled(input_path)
            shutil.move(input_path, output_path)

    def save_bin_mask(self):
        """save all labels in json_file as binary mask images
        """
        f = StringIO()
        f.write(base64.b64decode(self.json_data['imageData']))
        img = np.array(PIL.Image.open(f))

        target_names = {'background': 0}
        label = np.zeros(img.shape[:2], dtype=np.int32)
        for shape in self.json_data['shapes']:
            # get label value
            label_value = target_names.get(shape['label'])
            if label_value is None:
                label_value = len(target_names)
                target_names[shape['label']] = label_value
            # polygon -> mask
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            mask = PIL.Image.fromarray(mask)
            xy = map(tuple, shape['points'])
            PIL.ImageDraw.Draw(mask).polygon(xy=xy, outline=1, fill=1)
            mask = np.array(mask)
            # fill label value
            label[mask == 1] = label_value
            file_name = self.prefix + '_' + shape['label'] + '.pbm'
            cv2.imwrite(
                file_name, 255 * (label == label_value).astype(np.uint8))
            self.input_paths.append(file_name)

    def __call__(self):
        with open(self.pkl_path, 'rb') as f:
            self.data = pickle.load(f)
        if os.path.exists(self.json_path):
            print('SKIPPING because json alreadly exists')
            return

        print('\n\n')
        print('working on......  ', self.prefix)
        print('==============================================================')
        self.print_target_objects()

        # create label json
        subprocess.call(
            'labelme -O ' + self.json_path + ' ' + self.img_path, shell=True)

        if not os.path.isfile(self.json_path):
            print('exiting')
            return

        # load json data and pkl data
        self.load_data()

        # check if delete message is in json
        if self.check_delete_command():
            self.delete_this_set()
            return

        # convert labeled ids to valid object names
        if not self.convert_id_to_object_name():
            return

        # create binary mask from json
        self.save_bin_mask()

        # move data to a directory for labeled images
        self.move_data_to_labeled()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    args = parser.parse_args()

    labeled_dir = args.path + '_labeled/'
    ls_stdout = subprocess.check_output('ls ' + args.path, shell=True)
    ls_results = ls_stdout.split()

    color_results = [result for result in ls_results if '.jpg' in result]

    for result in color_results:
        prefix = os.path.join(args.path, result[:-4])
        label_me_multiple = LabelMeMultiple(prefix, labeled_dir)
        label_me_multiple()


if __name__ == '__main__':
    main()
