#!/usr/bin/env python

import rospkg
from jsk_apc2016_common.segmentation_in_bin.bin_data import BinData
from jsk_apc2016_common.rbo_segmentation.apc_data import APCSample
from image_geometry import cameramodels
import numpy as np
import pickle
import cv2

rospack = rospkg.RosPack()
pack_path = rospack.get_path('jsk_apc2016_common')


class RBOSegmentationInBin(object):
    """
    This class bridges data and RBO's segmentation in bin algorithm.
    """
    def __init__(self, *args, **kwargs):
        self.shelf = {}
        self.mask_img = None
        self.dist_img = None
        self._target_bin = None
        self.camera_model = cameramodels.PinholeCameraModel()

    def from_bin_info_array(self, bin_info_arr):
        for bin_info in bin_info_arr.array:
            self.shelf[bin_info.name] = BinData(bin_info=bin_info)

    def load_trained(self, path):
        with open(path, 'rb') as f:
            self.trained_segmenter = pickle.load(f)

    def set_apc_sample(self):
        assert self.target_object is not None
        # TODO: work on define_later later
        define_later = np.zeros((
            self.camera_info.height, self.camera_info.width))
        data = {}
        data['objects'] = self.target_bin.objects
        data['dist2shelf_image'] = self.dist_img
        data['depth_image'] = define_later
        data['has3D_image'] = define_later
        data['height3D_image'] = self.height_img
        data['height2D_image'] = define_later
        self.apc_sample = APCSample(
                image_input=self.img_color,
                bin_mask_input=self.mask_img,
                data_input=data,
                labeled=False,
                infer_shelf_mask=False,
                pickle_mask=False)

    def segmentation(self):
        zoomed_predicted_segment = self.trained_segmenter.predict(
                apc_sample=self.apc_sample,
                desired_object=self.target_object)
        self.predicted_segment = self.apc_sample.unzoom_segment(
                zoomed_predicted_segment)

        # Masked region needs to contain value 255.
        self.predicted_segment = 255 * self.predicted_segment.astype('uint8')
