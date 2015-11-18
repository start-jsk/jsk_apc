#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import yaml


def object_list():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_file = os.path.join(this_dir, 'data/object_list.yml')
    with open(yaml_file) as f:
        objects = yaml.load(f)
    return objects