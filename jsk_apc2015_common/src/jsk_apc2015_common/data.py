#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import yaml
from .util import _load_yaml


def object_list():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_file = os.path.join(this_dir, 'data/object_list.yml')
    objects = _load_yaml(yaml_file)
    return objects