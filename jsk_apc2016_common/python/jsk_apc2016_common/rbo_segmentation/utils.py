__author__ = 'rico'

from copy import deepcopy
import cv2
import os.path

import cPickle as pickle
import matplotlib.pyplot as plt
import numpy as np


def visualize_color_image(img, title=''):
    plt.figure()
    plt.title(title)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_HSV2RGB))

def basic_image_check(img, title):
    print('max of img ' + title + ' ', np.max(img))
    print('dtype of img ' + title + ' ', img.dtype)
    plt.figure()
    plt.title(title)
    plt.imshow(img)


def visualize_apc_sample_data_dict(apc_sample):
    basic_image_check(apc_sample.data_dict['depth'], 'depth')
    basic_image_check(apc_sample.data_dict['mask_image'], 'mask_image')
    basic_image_check(
        apc_sample.data_dict['dist2shelf_image'], 'dist2shelf_image')
    basic_image_check(apc_sample.data_dict['height3D_image'], 'height3D_image')


def visualize_apc_sample_feature(apc_sample):
    visualize_color_image(apc_sample.image)

    basic_image_check(apc_sample.feature_images['depth'], 'depth')
    basic_image_check(apc_sample.bin_mask, 'mask_image')
    basic_image_check(
        apc_sample.feature_images['dist2shelf'], 'dist2shelf_image')
    basic_image_check(apc_sample.feature_images['height3D'], 'height3D_image')
    

class Utils:
    HUE_WHITE = 180
    HUE_GRAY = 181
    HUE_BLACK = 182

    @staticmethod
    def gaussian(x, mu, sigma):
        return np.exp(-(x - mu) ** 2.0 / (2.0 * sigma ** 2.0))

    @staticmethod
    def backproject(histogram, image):
        if len(image.shape) > 1:
            return histogram[image.reshape(-1)].reshape(image.shape)
        else:
            return histogram[image]

    @staticmethod
    def hsv2hwgb(image):  # hue white gray black
        """transform an hsv image to a single channel image with values 0-179 for hues, 180 for white, 181 for gray, and 182 for black """
        unsaturated = np.clip(1 * (image[:, :, 1] < 90) + 1 * (image[:, :, 2] < 20), 0, 1)  # 150
        dark = 1 * (image[:, :, 2] < 50)
        bright = 1 * (image[:, :, 2] > 200)
        image_hue = np.copy(image[:, :, 0])
        return (unsaturated * bright * Utils.HUE_WHITE + unsaturated * (1 - dark) * (1 - bright) * Utils.HUE_GRAY + unsaturated * dark * Utils.HUE_BLACK + (1 - unsaturated) * image_hue).astype('uint8')

    @staticmethod
    def hwgb2hsv(hwgb):
        """transform a single channel (hue white gray black) image to an hsv image"""
        image = 255 * np.ones(list(hwgb.shape) + [3])
        image[:, :, 0] = hwgb

        saturated = 1 * (hwgb < 180)[:, :, None]
        white = 1 * (hwgb == 180)[:, :, None]
        gray = 1 * (hwgb == 181)[:, :, None]
        black = 1 * (hwgb == 182)[:, :, None]

        image_black = np.zeros(image.shape).astype('int')
        image_white = np.zeros(image.shape).astype('int')
        image_white[:, :, 2] = 255
        image_gray = np.zeros(image.shape).astype('int')
        image_gray[:, :, 2] = 128

        return (saturated * image + white * image_white + gray * image_gray + black * image_black).astype('uint8')

    @staticmethod
    def hsv2edge_image(image):
        # 70 / 210
        edge_image = (cv2.Canny(image[:, :, 2], 80, 240) / 255.0).astype('uint8')
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)).astype('uint8')
        return cv2.dilate(edge_image, kernel, iterations=1)

    @staticmethod
    def load_mask(filename):
        result = cv2.imread(filename, 0)
        if result is not None:
            return result.astype('bool')

    @staticmethod
    def load_image(filename):
        return cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2HSV)

    @staticmethod
    def load_supplementary_data(filename):
        # load pickled data, skip if it does not exist
        if os.path.isfile(filename):
            with open(filename) as f:
                data = pickle.load(f)
            return data

    @staticmethod
    def compute_feature_images(image, data):

        feature_images = dict()

        # compute all feature images
        rgb_image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
        feature_images['red'] = image[:, :, 0]
        feature_images['green'] = image[:, :, 1]
        feature_images['blue'] = image[:, :, 2]

        feature_images['hue'] = image[:, :, 0]
        feature_images['saturation'] = image[:, :, 1]
        feature_images['value'] = image[:, :, 2]
        feature_images['color'] = Utils.hsv2hwgb(image)
        feature_images['edge'] = Utils.hsv2edge_image(image)
        feature_images['miss3D'] = (1 * (data['has3D_image'] == 0)).astype('uint8')
        feature_images['dist2shelf'] = np.clip(data['dist2shelf_image'].astype('uint8'), 0, 100)
        feature_images['height3D'] = np.clip((data['height3D_image'] + 0.1) * 500, 0, 255).astype('uint8')
        feature_images['height2D'] = np.clip((data['height2D_image'] + 0.1) * 500, 0, 255).astype('uint8')
        feature_images['depth'] = data['depth_image']

        return feature_images


class Display:

    def __init__(self, bin_mask=None):
        #plt.ion()
        self.heatmap = plt.get_cmap('coolwarm')
        #self.heatmap = plt.get_cmap('jet')
        self.heatmap.set_bad('w', 1.)
        self.bin_mask = None
        self.figure_num = 0

    def set_bin_mask(self, bin_mask):
        self.bin_mask = bin_mask.astype('bool')
        contours, hierarchy = cv2.findContours(self.bin_mask.astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if self.bin_mask is not None:
            x, y, w, h = cv2.boundingRect(contours[0])
            self.xlim = [x, x + w]
            self.ylim = [y + h, y]

    def figure(self, title):

        if title is None:
            title = str(self.figure_num)
            self.figure_num += 1
        fig = plt.figure(title)
        fig.clear()

    def set_limits(self):
        if self.bin_mask is not None:
            plt.xlim(self.xlim)
            plt.ylim(self.ylim)
            plt.axis('off')

    def plot_heat_image(self, image, mask=None, title=None, colorbar_label='', vmin=None, vmax=None):

        self.figure(title)

        if mask is None:
            mask = self.bin_mask
        else:
            if self.bin_mask is not None:
                mask = np.logical_and(mask, self.bin_mask)

        if mask is not None:
            image = np.ma.array(image, mask=np.logical_not(mask))

        plt.imshow(image, interpolation='nearest', cmap=self.heatmap, vmin=vmin, vmax=vmax)
        # cb = plt.colorbar()
        # cb.set_label(colorbar_label)

        self.set_limits()

    def plot_image(self, image, title=None):

        self.figure(title)

        image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
        if self.bin_mask is not None:
            image[np.logical_not(self.bin_mask)] = np.zeros(image[np.logical_not(self.bin_mask)].shape) + 255
        plt.imshow(image.astype('uint8'), interpolation='nearest')

        self.set_limits()

    def plot_color_image(self, color_image, title=None):

        self.figure(title)

        self.plot_image(Utils.hwgb2hsv(color_image), title=title)


    def plot_segment(self, image, segment, title=None):

        self.figure(title)

        image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)

        alpha = 0.3
        image[np.logical_not(segment)] = alpha * image[np.logical_not(segment)] + (1 - alpha) * np.array([[150, 150, 150]])
        image[np.logical_not(self.bin_mask)] = np.zeros(image[np.logical_not(self.bin_mask)].shape) + 255

        plt.imshow(image.astype('uint8'), interpolation='nearest')

        self.set_limits()

    def plot_contours(self, image, contours, title=None):

        self.figure(title)

        image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)

        image[np.logical_not(self.bin_mask)] = np.zeros(image[np.logical_not(self.bin_mask)].shape) + 255

        for cnt in contours:
            cv2.drawContours(image, [cnt], 0, (0, 255, 0), 2, lineType=8)

        plt.imshow(image.astype('uint8'), interpolation='nearest')

        self.set_limits()


    def plot_color_likelihood(self, color_likelihood, title=None):

        self.figure(title)

        width = 10.0
        plt.bar(range(180), color_likelihood[:180], color=plt.get_cmap('hsv')((np.arange(183) * 255.0 / 180.0).astype('int')), width=1)
        plt.bar(180 + width * np.arange(3), color_likelihood[180:183] / width, color=['white', 'gray', 'black'], width=width)
        plt.xlim([0, 180 + width * 3])
        plt.xticks([])
        plt.xlabel('Color')
        plt.ylim([0, 0.05])
        plt.yticks([])
        plt.ylabel('Probability density')
        plt.axis('off')


    def plot_binary_likelihood(self, binary_likelihood, xlabel='0 / 1', title=None):

        self.figure(title)

        width = 10.0
        plt.bar(width * np.arange(2), binary_likelihood, color=self.heatmap([0, 255]), width=width)

        plt.xlim([0, width * 2])
        plt.xticks([])
        plt.xlabel(xlabel)
        plt.ylim([0, 1])
        plt.yticks([])
        plt.ylabel('Probability density')


    def plot_edge_likelihood(self, edge_likelihood, title=None):

        self.plot_binary_likelihood(edge_likelihood, xlabel='No edge  /  edge   ', title=title)


    def plot_miss3D_likelihood(self, miss3D_likelihood, title=None):

        self.plot_binary_likelihood(miss3D_likelihood, xlabel='No 3D info  /  3D info   ', title=title)


    def plot_range_likelihood(self, range_likelihood, xmin=0, xmax=1, xlabel='', title=None):

        self.figure(title)

        # plt.bar(range(range_likelihood.size), color=self.heatmap(xrange.astype('int')), width=width)

        xrange = np.linspace(xmin, xmax, range_likelihood.size)
        width = xrange[1] - xrange[0]
        plt.bar(xrange, range_likelihood, color=self.heatmap(np.linspace(0, 255, range_likelihood.size).astype('int')), width=width)

        plt.xlim([xmin, xmax + width])
        plt.xlabel(xlabel)
        plt.ylim([0, 0.2])
        plt.yticks([])
        plt.ylabel('Probability density')

# execute this function and you can use utils.display in every other module
def global_display():
    global display
    display = Display()
