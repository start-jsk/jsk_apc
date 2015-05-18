#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
PKG = 'jsk_2014_picking_challenge'
import os
import sys
import re
import unittest
from termcolor import cprint, colored

import rospkg
rp = rospkg.RosPack()
pkg_path = rp.get_path('jsk_2014_picking_challenge')
sys.path.append(os.path.join(pkg_path, 'scripts'))
sys.path.append(os.path.join(pkg_path, 'test'))


class TestRobotInput(unittest.TestCase):
    def test_bin_contents(self,  json_file=None):
        from bin_contents import get_bin_contents
        from common import get_object_list
        if json_file is None:
            json_file = os.path.join(pkg_path, 'data/apc-a.json')
        bin_contents = list(get_bin_contents(json_file))
        object_list = list(get_object_list())
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
        from common import get_object_list
        if json_file is None:
            json_file = os.path.join(pkg_path, 'data/apc-a.json')
        # for original work order
        work_order = list(get_work_order(json_file))
        self.assertEqual(len(work_order), len('abcdefghijkl'))
        object_list = get_object_list()
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


if __name__ == '__main__':
    import rosunit
    rosunit.unitrun(PKG, 'test_robot_input', TestRobotInput)
