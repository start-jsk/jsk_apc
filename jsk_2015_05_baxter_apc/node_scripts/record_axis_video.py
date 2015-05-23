#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
from __future__ import print_function
import os
import sys
import getpass
import datetime
import argparse
import subprocess

from termcolor import cprint


parser = argparse.ArgumentParser(description='JSK axis video recorder.')
parser.add_argument('-O', '--output-file', default='',
                    help='output file name. (ex. output.flv)')
args = parser.parse_args(sys.argv[1:])

output_file = args.output_file
if output_file == '':
    # ROS_HOME
    ros_home = os.environ.get('ROS_HOME', '')
    if ros_home == '':
        ros_home = os.path.join(os.environ['HOME'], '.ros')
    timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    output_file = os.path.join(ros_home, 'video_{0}.flv'.format(timestamp))

user = raw_input('axis username?: ')
password = getpass.getpass('axis password?: ')

try:
    print('Recording...')
    cprint('Please enter Ctr-C to stop recoding.', color='blue')
    mjpeg_file = '/tmp/video_{0}.mjpg'.format(timestamp)
    cmd = [
        'wget',
        '--quiet',
        '--http-user', user,
        '--http-password', password,
        'http://133.11.216.141/mjpg/video.mjpg',
        '-O', mjpeg_file,
        ]
    subprocess.call(cmd)
except KeyboardInterrupt:
    print('Encoding...')
    cmd = [
        'ffmpeg',
        '-i', mjpeg_file,
        '-vcodec', 'flv',
        '-loglevel', '8',
        output_file,
        ]
    subprocess.call(cmd)
    print('Output file: ', end='')
    cprint(output_file, color='green')
