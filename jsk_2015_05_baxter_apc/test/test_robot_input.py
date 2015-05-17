#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
PKG = 'jsk_2014_picking_challenge'
import os
import sys
import unittest

import rospkg
rp = rospkg.RosPack()
pkg_path = rp.get_path('jsk_2014_picking_challenge')

sys.path.append(os.path.join(pkg_path, 'scripts'))


class TestRobotInput(unittest.TestCase):
    def test_bin_contents(self):
        from bin_contents import get_bin_contents
        json_file = os.path.join(pkg_path, 'data/apc-a.json')
        bin_contents = get_bin_contents(json_file)
        self.assertEqual(len(list(bin_contents)), len('abcdefghijkl'))

    def test_work_order(self):
        from work_order import (
            get_work_order,
            get_sorted_work_order,
            get_work_order_msg,
            )
        json_file = os.path.join(pkg_path, 'data/apc-a.json')
        work_order = get_work_order(json_file)
        self.assertEqual(len(list(work_order)), len('abcdefghijkl'))
        sorted_work_order = get_sorted_work_order(json_file)
        self.assertEqual(len(list(sorted_work_order)), len('abcdefghijkl'))
        msg = get_work_order_msg(json_file)
        self.assertEqual(len(msg['left'].array), 8)
        self.assertEqual(len(msg['right'].array), 4)

if __name__ == '__main__':
    import rosunit
    rosunit.unitrun(PKG, 'test_robot_input', TestRobotInput)
