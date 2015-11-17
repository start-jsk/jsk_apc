#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
PKG = 'jsk_2015_05_baxter_apc'
import os
import sys
import re
import unittest
import yaml
from termcolor import cprint, colored

import rospkg
rp = rospkg.RosPack()
pkg_path = rp.get_path('jsk_2015_05_baxter_apc')
sys.path.append(os.path.join(pkg_path, 'scripts'))
sys.path.append(os.path.join(pkg_path, 'test'))

import jsk_apc2015_common


class TestRobotInput(unittest.TestCase):
    def test_bin_contents(self,  json_file=None):
        from bin_contents import get_bin_contents
        if json_file is None:
            json_file = os.path.join(pkg_path, 'data/apc-a.json')
        bin_contents = list(get_bin_contents(json_file))
        object_list = jsk_apc2015_common.data.object_list()
        for bin_, objects in bin_contents:
            self.assertIn(bin_, 'abcdefghijkl')
            for object_ in objects:
                self.assertIn(object_, object_list)
        self.assertEqual(len(bin_contents), len('abcdefghijkl'))

    def test_work_order(self, json_file=None):
        from work_order import (
            get_work_order,
            get_sorted_work_order,
            get_work_order_msg,
            )
        if json_file is None:
            json_file = os.path.join(pkg_path, 'data/apc-a.json')
        # for original work order
        work_order = list(get_work_order(json_file))
        self.assertEqual(len(work_order), len('abcdefghijkl'))
        object_list = jsk_apc2015_common.data.object_list()
        for bin_, object_ in work_order:
            self.assertIn(bin_, 'abcdefghijkl')
            self.assertIn(object_, object_list)
        # for sorted work order
        sorted_work_order = list(get_sorted_work_order(json_file))
        self.assertEqual(len(sorted_work_order), len('abcdefghijkl'))
        for bin_, object_ in sorted_work_order:
            self.assertIn(bin_, 'abcdefghijkl')
            self.assertIn(object_, object_list)
        # for msg
        msg = get_work_order_msg(json_file)
        self.assertEqual(len(msg['left'].array), 8)
        self.assertEqual(len(msg['right'].array), 4)
        for lr in ('left', 'right'):
            for order in msg[lr].array:
                self.assertIn(order.bin, 'abcdefghijkl')
                self.assertIn(order.object, object_list)

    def test_json_file(self):
        from interface_test import interface_test
        files = os.listdir(os.path.join(pkg_path, 'data'))
        r = re.compile('^apc.*\.json$')
        json_files = filter(r.match, files)
        for json in json_files:
            cprint('\nChecking: {0}'.format(json), 'blue')
            json = os.path.join(pkg_path, 'data', json)
            self.test_work_order(json_file=json)
            self.test_bin_contents(json_file=json)
            interface_test(json)

    def test_classifier_weight_yaml(self):
        yaml_file = os.path.join(pkg_path, 'data/classifier_weight.yml')
        with open(yaml_file) as f:
            weight = yaml.load(f)
        object_list = jsk_apc2015_common.data.object_list()
        for object_, weights in weight.items():
            self.assertIn(object_, object_list)
            for clf, weight in weights.items():
                self.assertIn(clf, ['bof', 'color'])
                self.assert_(0. <= weight <= 1.)

if __name__ == '__main__':
    import rosunit
    rosunit.unitrun(PKG, 'test_robot_input', TestRobotInput)
