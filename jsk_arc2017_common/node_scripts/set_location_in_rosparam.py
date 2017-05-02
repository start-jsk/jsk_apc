#!/usr/bin/env python

import json
import os.path as osp
import rospy


def main():
    json_dir = rospy.get_param('~json_dir')
    location = rospy.get_param('~location')
    location_path = osp.join(json_dir, 'item_location_file.json')
    with open(location_path) as location_f:
        data = json.load(location_f)
    print('location: {}'.format(location))
    if location == 'bins':
        bins = data[location]
        bin_contents = {}
        for bin_ in bins:
            bin_contents[bin_['bin_id']] = bin_['contents']
            print('bin: {}'.format(bin_['bin_id']))
            for content in bin_['contents']:
                print('  {}'.format(content))
        rospy.set_param('~param', bin_contents)
    elif location == 'boxes':
        boxes = data[location]
        box_contents = {}
        for box in boxes:
            box_contents[box['size_id']] = box['contents']
            print('box: {}'.format(box['size_id']))
            for content in box['contents']:
                print('  {}'.format(content))
        rospy.set_param('~param', box_contents)
    else:  # tote contents
        rospy.set_param('~param', data[location]['contents'])
        for content in data[location]['contents']:
            print('  {}'.format(content))


if __name__ == '__main__':
    rospy.init_node('set_location_in_rosparam')
    main()
