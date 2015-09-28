#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
import os
import subprocess
import rospkg


rp = rospkg.RosPack()
pkg_path = rp.get_path('jsk_2015_05_baxter_apc')

test_path = os.path.join(pkg_path, 'test/test_robot_input.py')

output = subprocess.check_output(['rosunit', test_path, '--text'])
print(output)