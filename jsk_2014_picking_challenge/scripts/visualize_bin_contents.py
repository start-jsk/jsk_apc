#!/usr/bin/env python
# -*- coding:utf-8 -*-

import argparse
import os
import sys
from bin_contents import get_bin_contents

def main():
    arg_fmt = argparse.RawTextHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=arg_fmt,
                                     description=main.__doc__)
    required = parser.add_argument_group('required arguments')
    required.add_argument(
        '-f', '--file', required=True,
        help='select json file.'
    )
    args = parser.parse_args(sys.argv[1:])

    cmd = "montage"
    for bin_contents in sorted(get_bin_contents(args.file)):
        bin_label = bin_contents[0]
        bin_content = bin_contents[1]
        for obj_name in bin_content:
            cmd += ' ../data/raw_img/' + obj_name + '.jpg'
        for _ in range(3 - len(bin_content)):
            cmd += ' ../data/paste_mask.png'
    cmd += ' -tile 3x12 output.png'
    os.system(cmd)

if __name__ == "__main__":
    main()
