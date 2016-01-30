#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2


def rescale(img, scale):
    src_h, src_w = img.shape[:2]
    dst_h, dst_w = int(scale * src_h), int(scale * src_w)
    return cv2.resize(img, (dst_w, dst_h))
