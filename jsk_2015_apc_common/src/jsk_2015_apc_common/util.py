#!/usr/bin/env python
# -*- coding: utf-8 -*-

import yaml


def _load_yaml(filename):
    with open(filename, 'r') as f:
        return yaml.load(f)