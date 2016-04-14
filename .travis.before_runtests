#!/usr/bin/env python

import argparse
import os
import subprocess


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--packages', nargs='*',
                        help='packages which should exist')
    parser.add_argument('--workspace', help='catkin workspace')
    args = parser.parse_args()

    packages = args.packages
    workspace = args.workspace

    dotcatkin = os.path.join(workspace, 'devel/.catkin')
    pkg_paths = open(dotcatkin).read().split(';')
    for pkg in packages:
        pkg_path = subprocess.check_output(
            ['catkin', 'locate', pkg], cwd=workspace).strip()
        if pkg_path not in pkg_paths:
            pkg_paths.append(pkg_path)
    open(dotcatkin, 'w').write(';'.join(pkg_paths))
