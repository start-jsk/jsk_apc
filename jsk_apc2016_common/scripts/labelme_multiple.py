#!/usr/bin/env python

import sh
import argparse
import os
import pickle

import json

import base64
from cStringIO import StringIO
import PIL.Image
import PIL.ImageDraw
import numpy as np
import cv2


object_names = [
    'staples_index_cards', 'cloud_b_plush_bear', 'ticonderoga_12_pencils',
    'clorox_utility_brush', 'woods_extension_cord', 'peva_shower_curtain_liner',
    'creativity_chenille_stems', 'kleenex_tissue_box', 'fiskars_scissors_red',
    'safety_first_outlet_plugs', 'easter_turtle_sippy_cup', 'elmers_washable_no_run_school_glue',
    'kyjen_squeakin_eggs_plush_puppies', 'cool_shot_glue_sticks', 'i_am_a_bunny_book',
    'fitness_gear_3lb_dumbbell', 'command_hooks', 'womens_knit_gloves',
    'scotch_duct_tape', 'up_glucose_bottle', 'dasani_water_bottle',
    'dove_beauty_bar', 'hanes_tube_socks', 'soft_white_lightbulb',
    'kleenex_paper_towels', 'rolodex_jumbo_pencil_cup', 'folgers_classic_roast_coffee',
    'laugh_out_loud_joke_book', 'jane_eyre_dvd', 'platinum_pets_dog_bowl',
    'scotch_bubble_mailer', 'crayola_24_ct', 'rawlings_baseball',
    'barkely_hide_bones', 'dr_browns_bottle_brush', 'cherokee_easy_tee_shirt',
    'oral_b_toothbrush_red', 'oral_b_toothbrush_green', 'expo_dry_erase_board_eraser']
object_names_dict = {}
for i, name in enumerate(object_names):
    object_names_dict[name] = i


def print_target_objects(prefix):
    with open(prefix + '.pkl', 'rb') as f:
        data = pickle.load(f)
    for _object in data['objects']:
        print 'id: {},  {}'.format(object_names_dict[_object], _object)


def convert_id_to_object_name(json_path):
    """Modify label name of json from ids to object names

    Args:
        json_path (str): path to a json file that is going to be modified
    """
    with open(json_path, 'rb') as f:
        json_data = json.load(f)
    for shape in json_data['shapes']:
        label = shape['label']
        shape['label'] = object_names[int(label)]
    with open(json_path, 'w') as f:
        f.write(json.dumps(json_data, indent=2))


def save_bin_mask(json_file):
    """save all labels in json_file as binary mask images
    """
    data = json.load(open(json_file))
    f = StringIO()
    f.write(base64.b64decode(data['imageData']))
    img = np.array(PIL.Image.open(f))

    target_names = {'background': 0}
    label = np.zeros(img.shape[:2], dtype=np.int32)
    for shape in data['shapes']:
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
        file_name = json_file[:-5] + '_objects_' + shape['label'] + '.pbm'
        cv2.imwrite(file_name, 255 * (label == label_value).astype(np.uint8))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    args = parser.parse_args()

    ls_stdout = sh.ls(args.path).stdout
    ls_results = ls_stdout.split()

    color_results = [result for result in ls_results if '.jpg' in result]

    for result in color_results:
        json_path = os.path.join(args.path, result[:-4] + '.json')
        labelme_cmd = sh.Command('labelme')
        print '\n\n'
        print 'working on......  ', result[-9:-4]
        print '=============================================================='
        print_target_objects(os.path.join(args.path, result[:-4]))

        # create label json
        labelme_cmd('-O', json_path, os.path.join(args.path, result))
        convert_id_to_object_name(json_path)

        # create binary mask from json
        save_bin_mask(json_path)


if __name__ == '__main__':
    main()
