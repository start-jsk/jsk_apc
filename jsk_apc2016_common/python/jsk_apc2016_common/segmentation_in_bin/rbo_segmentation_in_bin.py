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
        if 'trained_pkl_path' in kwargs:
            self.load_trained(kwargs['trained_pkl_path'])
        if 'target_bin_name' in kwargs:
            self.target_bin_name = kwargs['target_bin_name']

    def from_bin_info_array(self, bin_info_arr):
        for bin_info in bin_info_arr.array:
            self.shelf[bin_info.name] = BinData(bin_info=bin_info)
        if self.target_bin_name is not None:
            self.target_bin = self.shelf[self.target_bin_name]
            self.target_object = self.target_bin.target

    def load_trained(self, path):
        with open(path, 'rb') as f:
            self.trained_segmenter = pickle.load(f)

    @property
    def target_bin_name(self):
        return self._target_bin_name

    @target_bin_name.setter
    def target_bin_name(self, target_bin_name):
        if target_bin_name is not None:
            self._target_bin_name = target_bin_name
        if target_bin_name in self.shelf:
            self.target_object = self.shelf[target_bin_name].target
            assert self.target_object is not None
            self.target_bin = self.shelf[target_bin_name]

    @property
    def camera_info(self):
        return self._camera_info

    @camera_info.setter
    def camera_info(self, camera_info):
        self._camera_info = camera_info
        if camera_info is not None:
            self.camera_model.fromCameraInfo(camera_info)

    @property
    def img_color(self):
        return self._img_color

    @img_color.setter
    def img_color(self, img_color):
        # following RBO's convention that img is loaded as HSV
        if img_color is not None:
            self._img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)
        else:
            self._img_color = None

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

        self.predicted_segment = self.predicted_segment.astype('uint8')
