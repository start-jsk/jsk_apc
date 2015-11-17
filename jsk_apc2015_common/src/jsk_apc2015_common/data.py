#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import yaml


def _load_yaml(filename):
    with open(filename, 'r') as f:
        return yaml.load(f)


def object_list():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_file = os.path.join(this_dir, 'data/object_list.yml')
    objects = _load_yaml(yaml_file)
    return objects