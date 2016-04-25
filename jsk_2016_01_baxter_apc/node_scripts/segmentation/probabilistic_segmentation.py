__author__ = 'rico jonschkowski'

import cv2
import scipy.ndimage

import numpy as np
from sklearn import preprocessing  # @UnresolvedImport
from sklearn.svm import SVC  # @UnresolvedImport
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, BaggingClassifier

import utils
from utils import Utils
from apc_data import APCDataSet

import matplotlib.pyplot as plt

class ProbabilisticSegmentation:

    sigma_smoothing = 4.0
    p_greedy_is_correct = 0.8

    def __init__(self, use_features=['color', 'edge', 'miss3D', 'height3D', 'dist2shelf', 'height2D'],
                 segmentation_method='edge_cut', selection_method='max_smooth', make_convex=False,
                 do_shrinking_resegmentation=True, do_greedy_resegmentation=True):
        """
        :param use_features: e.g. ['hue', 'saturation', 'value', 'edge', 'miss3D', 'height3D', 'dist2shelf', 'height2D']
        :param segmentation_method: "max", "max_smooth", "simple_cut", or "edge_cut"
        :param selection_method: "all", "largest", or "max_smooth"
        :param make_convex: boolean
        :param shrinking_resegmentation: boolean
        :param greedy_resegmentation: boolean
        """

        self.use_features = list(set(use_features))
        self.segmentation_method = segmentation_method
        self.selection_method = selection_method
        self.make_convex = make_convex
        self.do_shrinking_resegmentation = do_shrinking_resegmentation
        self.do_greedy_resegmentation = do_greedy_resegmentation

        self.posterior_images = []
        self.posterior_images_smooth = []

    def compute_posterior_smooth(self):
        self.posterior_images_smooth = {o: cv2.GaussianBlur(self.posterior_images[o], (31, 31), self.sigma_smoothing)
                                        for o in self.candidate_objects}

    def segment_max(self):
        self.segmentation = np.argmax(np.array([self.posterior_images[o] for o in self.candidate_objects]), 0)

    def segment_max_smooth(self):
        self.segmentation = np.argmax(np.array([self.posterior_images_smooth[o] for o in self.candidate_objects]), 0)

    def segment_simple_cut(self):
        # noinspection PyUnresolvedReferences
        from pygco import cut_simple

        unaries = -np.log(np.array([np.clip(self.posterior_images[o],a_min=0.01, a_max=1.0) for o in self.candidate_objects])).transpose((1, 2, 0))
        num_labels = unaries.shape[2]
        p_same_label = 0.99
        pairwise = -np.log(1 - p_same_label) * (1 - np.eye(num_labels)) - np.log(p_same_label) * np.eye(num_labels)
        k = 10  # scaling factor for potentials to reduce aliasing because the potentials need to be converted to integers
        self.segmentation = cut_simple(np.copy((k * unaries).astype('int32'), order='C'), np.copy((k * pairwise).astype('int32'), order='C'))


    def segment_edge_cut(self):
        # noinspection PyUnresolvedReferences
        from pygco import cut_from_graph

        height = self.image.shape[0]
        width = self.image.shape[1]

        # first, we construct the grid graph
        # inds is a matrix of cell indices
        inds = np.arange(height * width).reshape(height, width)
        # list of all horizontal and vertical edges
        horizontal_edges = np.c_[inds[:, :-1].ravel(), inds[:, 1:].ravel()]
        vertical_edges = np.c_[inds[:-1, :].ravel(), inds[1:, :].ravel()]
        # concatenate both lists
        edges = np.vstack([horizontal_edges, vertical_edges]).astype(np.int32)

        # the edge weight multiplies the logarithm of the probability, this is the same as potentiating the probability
        # and renormalizing
        def effective_probability(probability, weight):
            return probability ** weight / (probability ** weight + (1 - probability) ** weight)

        depth = self.feature_images['depth'].reshape(-1)
        depth[depth == -1] = np.nan

        self.edge_depth_diff = np.abs(depth[edges[:, 0]] - depth[edges[:, 1]])

        weights = np.ones([edges.shape[0], 1])  # weight is 1 for nan
        depth_diff_thres = 0.02  # in m

        # replace nans with threshold to prevent comparing to nan warnings
        self.edge_depth_diff[np.isnan(self.edge_depth_diff)] = depth_diff_thres
        smooth_depth_edges = self.edge_depth_diff < depth_diff_thres
        sharp_depth_edges = self.edge_depth_diff > depth_diff_thres

        weights[smooth_depth_edges] = 2  # effective for 0.95 + weight 2 -> 0.9972375
        weights[sharp_depth_edges] = 0

        # add this for displaying the depth edges
        self.depth_edge_image = np.zeros(height * width)
        self.depth_edge_image[edges[smooth_depth_edges].reshape(-1)] = -1
        self.depth_edge_image[edges[sharp_depth_edges].reshape(-1)] = 1
        self.depth_edge_image = self.depth_edge_image.reshape(height, width)

        edges = np.hstack([edges, weights])

        # compute potentials (log of probabilities)
        unaries = -np.log(np.array([np.clip(self.posterior_images[o],a_min=0.01, a_max=1.0) for o in self.candidate_objects])).transpose((1, 2, 0))
        num_labels = unaries.shape[2]
        p_same_label = 0.95
        pairwise = -np.log(1 - p_same_label) * (1 - np.eye(num_labels)) - np.log(p_same_label) * np.eye(num_labels)

        k = 10  # scaling factor for potentials to reduce aliasing because the potentials need to be converted to integers

        unaries_c = np.copy((k * unaries).astype('int32'), order='C').reshape(-1, num_labels)
        pairwise_c = np.copy((k * pairwise).astype('int32'), order='C')
        edges_c = edges.astype('int32')
        self.segmentation = cut_from_graph(edges_c, unaries_c, pairwise_c).reshape(height, width)


    def select_largest(self):
        ''' This method picks the largest segment of the desired object and computes its convex hull.'''

        index_desired = self.candidate_objects.index(self.desired_object)
        self.contours, hierarchy = cv2.findContours(np.logical_and(self.bin_mask, self.segmentation == index_desired).astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        self.segment_desired = np.zeros(self.segmentation.shape, np.uint8)
        if self.contours:
            # http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_imgproc/py_contours/py_contour_features/py_contour_features.html
            contour_areas = [cv2.contourArea(cnt) for cnt in self.contours]
            i = np.argmax(contour_areas)

            hull = cv2.convexHull(self.contours[i])

            cv2.drawContours(self.segment_desired, [hull], 0, 255, -1)


    def select_and_convexify(self, selection_method, make_convex, object):

        # index_desired = self.candidate_objects.index(self.desired_object)
        object_index = self.candidate_objects.index(object)
        self.segments[object] = np.zeros(self.segmentation.shape, np.uint8)
        self.contours, hierarchy = cv2.findContours(np.logical_and(self.bin_mask, self.segmentation == object_index).astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if self.contours:
            # http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_imgproc/py_contours/py_contour_features/py_contour_features.html
            # compute the area for each contour

            if selection_method == "all":
                self.selected_contours = range(len(self.contours))
            elif selection_method == "largest":
                contour_areas = [cv2.contourArea(cnt) for cnt in self.contours]
                self.selected_contours = [np.argmax(contour_areas)]
            elif selection_method == "max_smooth":
                # find most likely point for object on the parts that were segmented
                argmax = (self.posterior_images_smooth[object] * (self.segmentation == object_index)).argmax()
                # compute the image coordinates of this point
                max_posterior_point = np.unravel_index(argmax, self.posterior_images_smooth[object].shape)[::-1]
                # select contour that includes this point
                self.selected_contours = [np.argmax([cv2.pointPolygonTest(cnt,max_posterior_point,False) for cnt in self.contours])]
            else:
                print("I DON'T KNOW THIS SELECTION METHOD!")

            for i in self.selected_contours:

                if make_convex:
                    self.contours[i] = cv2.convexHull(self.contours[i])

                cv2.drawContours(self.segments[object], [self.contours[i]], 0, 255, -1)


    def segment_and_all(self):

        for object_name in self.candidate_objects:
            self.posterior_images[object_name][np.logical_not(self.bin_mask)] = 1 if object_name == 'shelf' else 0

        #print("trying to find {}".format(self.desired_object))
        resegment_same_object = False
        shrinking_posterior_penalty = 0
        self.segments = dict()
        while True:

            if "smooth" in self.segmentation_method or "smooth" in self.selection_method:
                self.compute_posterior_smooth()

            if self.segmentation_method == "max":
                self.segment_max()
            elif self.segmentation_method == "max_smooth":
                self.segment_max_smooth()
            elif self.segmentation_method == "simple_cut":
                self.segment_simple_cut()
            elif self.segmentation_method == "edge_cut":
                self.segment_edge_cut()
            else:
                print("I DON'T KNOW THIS SEGMENTATION METHOD!")

            if not resegment_same_object:
                # find next object for greedy segementation
                if self.do_greedy_resegmentation:
                    # find largest object, because this is the one we are probably most certain about
                    count = np.bincount(self.segmentation.flatten(), minlength=len(self.candidate_objects))
                    for object in self.candidate_objects:
                        # ignore shelf and objects that have already been segmented
                        if object == 'shelf' or object in self.segments.keys():
                            count[self.candidate_objects.index(object)] = -1
                        current_object = self.candidate_objects[np.argmax(count)]
                else:
                    current_object = self.desired_object

                #print('-> current object is {}'.format(current_object))

            self.select_and_convexify(self.selection_method, self.make_convex, current_object)

            # compute size of selected segment
            segment_size = np.sum(self.segments[current_object]>0).astype('float')
            size_relative_to_max = segment_size / np.max(self.mask_sizes['num'][current_object])
            size_relative_to_mean = segment_size / np.mean(self.mask_sizes['num'][current_object])

            if self.do_shrinking_resegmentation and size_relative_to_max > 1.2:
                #print("----> too big, I will shrink it ({:.2f} x max size)".format(size_relative_to_max))
                # reduce probability of that object
                self.posterior_images[current_object] -= 0.05
                shrinking_posterior_penalty += 0.05
                resegment_same_object = True
            else:
                # stop if the desired object was reached
                if current_object == self.desired_object:
                    break

                # else continue segmentation
                else:

                    # reset changes to the posterior of the object
                    if resegment_same_object:
                        self.posterior_images[current_object] += shrinking_posterior_penalty
                        shrinking_posterior_penalty = 0

                    if size_relative_to_mean < 0.5:
                        #print("----> too small, I can't use it ({:.2f} x mean size)".format(size_relative_to_mean))
                        pass
                    elif size_relative_to_max > 1.2:
                        #print("----> too big, I can't use it ({:.2f} x mean size)".format(size_relative_to_mean))
                        pass
                    else:
                        #print("----> greedy segmentation!")
                        self.posterior_images[current_object] = self.posterior_images[current_object] \
                            * ((1.0 - self.p_greedy_is_correct) + self.p_greedy_is_correct * (self.segments[current_object]>0))

                    resegment_same_object = False

    # call this in every fit of inheriting classes
    def fit(self, dataset):

        self.mask_sizes = dict()
        for size_name in ['num', 'radius']:
            self.mask_sizes[size_name] = dict()
            for object_name in APCDataSet.object_names:
                self.mask_sizes[size_name][object_name] = []

        # go through the training data
        for sample in dataset.samples:
            for object_name, object_mask in sample.object_masks.iteritems():
                # list of mask sizes for each object
                self.mask_sizes['num'][object_name].append(np.sum(object_mask))
                cnt = cv2.findContours((object_mask>0).astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0][0]
                self.mask_sizes['radius'][object_name].append(cv2.minEnclosingCircle(cnt)[1])

class ProbabilisticSegmentationRF(ProbabilisticSegmentation):
    """
    Probabilistic object segmentation using a random forest classifier
    """

    def __init__(self, classifier_params, lazy=True, num_samples=5000, **kwargs):
        ProbabilisticSegmentation.__init__(self, **kwargs)

        # use the seperate color features
        if 'color' in self.use_features:
            self.use_features.remove('color')
            for feature_name in ['hue', 'saturation', 'value']:
                if feature_name not in self.use_features:
                    self.use_features.append(feature_name)

        self.classifier_params = classifier_params
        self.num_samples = num_samples
        self.lazy=lazy
        self.feature_importances = dict()
        for feature_name in self.use_features:
            self.feature_importances[feature_name] = []

    def fit(self, dataset):

        ProbabilisticSegmentation.fit(self, dataset)

        #print('Building a feature matrix for each object ...')

        self.X_dict = dict()
        for object_name in APCDataSet.object_names:
            self.X_dict[object_name] = np.zeros([0, len(self.use_features)])

        for sample in dataset.samples:

            # pick pixel features for all labeled objects and put them into a matrix X
            for object_name, object_mask in sample.object_masks.iteritems():
                # features for all pixels labeled as the object
                x = np.vstack([sample.feature_images[feature_name][object_mask] for feature_name in self.use_features]).T
                # append these features to the feature matrix of the object
                self.X_dict[object_name] = np.vstack([self.X_dict[object_name], x])

        # subsample the pixel features for each object (if there is any data for the object)
        for object_name, X in self.X_dict.iteritems():
            if X.shape[0] > 0 and self.num_samples > 0:
                #print(X.shape[0])
                #self.X_dict[object_name] = X[np.random.choice(X.shape[0], min(X.shape[0], self.num_samples), replace=False), :].astype('double')
                # throws an error if there are less training samples then we want to sample
                self.X_dict[object_name] = X[np.random.choice(X.shape[0], self.num_samples, replace=False), :].astype('double')

        self.classifier = RandomForestClassifier(n_jobs = 2, **self.classifier_params)
        #self.classifier = RandomForestClassifier(n_estimators=100, min_weight_fraction_leaf=0.01, n_jobs = 2)

        # old code for using a similar color description as in our MRF
        # color_list = sample.feature_images['color'][object_mask]
        # color_flag_list = color_list < 180
        # color_feature = np.zeros([color_list.size, 3])  # color flag, hue, white/gray/black
        # color_feature[:, 0] = color_flag_list.astype('double')
        # color_feature[color_flag_list, 1] = color_list[color_flag_list]
        # color_feature[np.logical_not(color_flag_list), 2] = color_list[np.logical_not(color_flag_list)] - 180


        if not self.lazy:

            X = np.zeros([0, len(self.use_features)])
            Y = np.array([])

            for object_name in APCDataSet.object_names:

                x = self.X_dict[object_name]

                if x.shape[0] > 0:

                    y = APCDataSet.object_names.index(object_name) * np.ones(x.shape[0])
                    X = np.vstack([X, x])
                    Y = np.hstack([Y, y])

                else:

                    print('I have no training data for ' + object_name)

            # training the SVM
            self.scaler = preprocessing.StandardScaler().fit(X)
            self.classifier.fit(self.scaler.transform(X), Y)

            for feature_name, importance in zip(self.use_features, self.classifier.feature_importances_):
                self.feature_importances[feature_name].append(importance)


    def predict(self, apc_sample, desired_object):

        self.image = apc_sample.image
        self.bin_mask = apc_sample.bin_mask
        self.feature_images = apc_sample.feature_images
        self.candidate_objects = apc_sample.candidate_objects
        self.desired_object = desired_object

        if self.lazy:

            X = np.zeros([0, len(self.use_features)])
            Y = np.array([])

            for object_name in self.candidate_objects[:]:

                x = self.X_dict[object_name]

                if x.shape[0] > 0:

                    y = self.candidate_objects.index(object_name) * np.ones(x.shape[0])
                    X = np.vstack([X, x])
                    Y = np.hstack([Y, y])

                else:

                    print('I have no training data for ' + object_name)

                    # if there is no training data for this object
                    # return empty segment if this was the desired object
                    # otherwise, try to segment the image ignoring it

                    if object_name == self.desired_object:
                        return np.zeros(self.image.shape[:2]).astype('bool')
                    else:
                        self.candidate_objects.remove(object_name)

            # training the SVM
            self.scaler = preprocessing.StandardScaler().fit(X)
            self.classifier.fit(self.scaler.transform(X), Y)

            for feature_name, importance in zip(self.use_features, self.classifier.feature_importances_):
                self.feature_importances[feature_name].append(importance)

        # prediction
        X_pred = np.vstack([self.feature_images[feature_name].flatten() for feature_name in self.use_features]).T.astype('double')

        self.probabilities = self.classifier.predict_proba(self.scaler.transform(X_pred))

        self.posterior_images = dict()
        for i in range(self.probabilities.shape[1]):
            if self.lazy:
                self.posterior_images[self.candidate_objects[i]] = self.probabilities[:, i].reshape(self.image.shape[:2])
            elif APCDataSet.object_names[i] in self.candidate_objects:
                self.posterior_images[APCDataSet.object_names[i]] = self.probabilities[:, i].reshape(self.image.shape[:2])

        self.posterior_images['shelf'] *= ProbabilisticSegmentationBP.prior_shelf

        # segmentation and postprocessing
        self.segment_and_all()

        return self.segments[desired_object].astype('bool')

class ProbabilisticSegmentationBP(ProbabilisticSegmentation):
    """
    Probabilistic object segmentation using likelihood backprojection
    """

    sigmas = {'color_hue': 3, 'color_gray': 1, 'dist2shelf': 7.5, 'height3D': 3, 'height2D': 6}
    p_uniforms = {'color': 0.2, 'dist2shelf': 0.05, 'edge': 0.4, 'miss3D': 0.2, 'height3D': 0.4, 'height2D': 0.8}
    prior_shelf = 3

    def __init__(self, seattle_color_weight=0.8, **kwargs):
        ProbabilisticSegmentation.__init__(self, **kwargs)
        self.seattle_color_weight = seattle_color_weight


    def fit(self, dataset):

        ProbabilisticSegmentation.fit(self, dataset)

        self.prior = dict()

        # initialize for each feature, a histogram and likelihood dictionary, which will have one entry per object
        self.histograms = dict()
        self.likelihoods = dict()

        num_feature_values = {'color': 183, 'edge': 2, 'miss3D':2, 'height3D': 256, 'height2D': 256, 'dist2shelf': 101}
        num_feature_values['color_seattle'] = num_feature_values['color']

        for feature_name in list(self.use_features) + ['color_seattle']:
            self.histograms[feature_name] = dict()
            self.likelihoods[feature_name] = dict()
            for object_name in APCDataSet.object_names:
                self.histograms[feature_name][object_name] = np.zeros(num_feature_values[feature_name])

        def hist(feature_samples, range):
            return np.histogram(feature_samples, bins=range[1] + 1, range=range)[0] * 1.0

        def normalize_and_mix(histogram, p_uniform):
            # normalize
            if np.sum(histogram) == 0:
                print('stop')
            likelihood = histogram / np.sum(histogram)
            # mix with uniform distribution
            likelihood = (1 - p_uniform) * likelihood + p_uniform / likelihood.size
            return likelihood

        # go through the training data and fill the histograms
        for sample in dataset.samples:

            for object_name, object_mask in sample.object_masks.iteritems():

                # use mainly data from berlin to build the histograms

                masks = {feature_name: object_mask for feature_name in self.use_features}
                # use only pixels without 3D info to compute the height2D histogram
                masks['height2D'] = np.logical_and(object_mask, sample.feature_images['miss3D'])
                # use only pixels with 3D info to compute the height3D histogram
                masks['height3D'] = np.logical_and(object_mask, np.logical_not(sample.feature_images['miss3D']))

                for feature_name in self.use_features:
                    if feature_name not in ['dist2shelf', 'color']:
                        self.histograms[feature_name][object_name] += hist(sample.feature_images[feature_name][masks[feature_name]], range=(0, num_feature_values[feature_name] - 1))

                if 'color' in self.use_features:
                    if 'berlin' in sample.filenames['image']:
                        self.histograms['color'][object_name] += hist(sample.feature_images['color'][object_mask], range=(0, num_feature_values['color'] - 1))
                    elif 'seattle' in sample.filenames['image']:
                        self.histograms['color_seattle'][object_name] += hist(sample.feature_images['color'][object_mask], range=(0, num_feature_values['color'] - 1))
                    else:
                        print('This data is not from Berlin, nor from Seattle. What should I do with it?')

        # mix color histogram from seattle and berlin
        if 'color' in self.use_features:

            for object_name in APCDataSet.object_names:
                if np.sum(self.histograms['color'][object_name]) == 0:
                    #print('no berlin data for {}'.format(object_name))
                    self.histograms['color'][object_name] = self.histograms['color_seattle'][object_name]
                elif np.sum(self.histograms['color_seattle'][object_name]) == 0:
                    #print('no seattle data for {}'.format(object_name))
                    pass
                else:
                    self.histograms['color'][object_name] = \
                        (1-self.seattle_color_weight) * self.histograms['color'][object_name] / np.sum(self.histograms['color'][object_name]) \
                        + self.seattle_color_weight * self.histograms['color_seattle'][object_name] / np.sum(self.histograms['color_seattle'][object_name])

        # COMPUTE LIKELIHOODs FOR ALL OBJECTS AND ALL FEATURES
        for feature_name in self.use_features:
            if feature_name != 'dist2shelf':
                for object_name in APCDataSet.object_names:

                    if feature_name == 'color':
                        color_histogram_smooth_0 = scipy.ndimage.filters.gaussian_filter1d(self.histograms['color'][object_name][:180], sigma=ProbabilisticSegmentationBP.sigmas['color_hue'], mode='wrap')
                        color_histogram_smooth_1 = scipy.ndimage.filters.gaussian_filter1d(self.histograms['color'][object_name][180:], sigma=ProbabilisticSegmentationBP.sigmas['color_gray'], mode='nearest')
                        self.likelihoods['color'][object_name] = normalize_and_mix(np.concatenate([color_histogram_smooth_0, color_histogram_smooth_1]), ProbabilisticSegmentationBP.p_uniforms['color'])

                    else:
                        # do smoothing for histograms with more than two values
                        if num_feature_values[feature_name] > 2:
                            histogram_smooth = scipy.ndimage.filters.gaussian_filter1d(self.histograms[feature_name][object_name], sigma=ProbabilisticSegmentationBP.sigmas[feature_name], mode='nearest')
                        else:
                            histogram_smooth = self.histograms[feature_name][object_name]
                        # compute likelihood
                        self.likelihoods[feature_name][object_name] = normalize_and_mix(histogram_smooth, ProbabilisticSegmentationBP.p_uniforms[feature_name])

        # DISTANCE2SHELF LIKELIHOOD [in mm up to 100 mm]
        # not computed from data but heuristically set
        # only discriminates between shelf and all other objects (to avoid dependence with height feature)
        max_dist = np.arange(101)
        for object_name in APCDataSet.object_names:

            if 'dist2shelf' in self.use_features:
                if 'shelf' in object_name:
                    self.histograms['dist2shelf'][object_name] = Utils.gaussian(max_dist, 0, ProbabilisticSegmentationBP.sigmas['dist2shelf'])
                else:
                    self.histograms['dist2shelf'][object_name] = np.ones(max_dist.shape)
                self.likelihoods['dist2shelf'][object_name] = normalize_and_mix(self.histograms['dist2shelf'][object_name], ProbabilisticSegmentationBP.p_uniforms['dist2shelf'])

            # PRIOR FOR SHELF SHOULD BE HIGHER THAN OBJECTS (WHEN IN DOUBT, THE ROBOT SHOULD TREAT PIXELS AS PART OF THE SHELF)
            self.prior[object_name] = self.prior_shelf if 'shelf' in object_name else 1

    def backproject(self):

        # initialize dictionaries for the likelihood images
        self.likelihood_images = dict()
        for feature_name in list(self.use_features) + ['all']:
            self.likelihood_images[feature_name] = dict()

        # BACKPROJECT LIKELIHOODS TO FEATURE IMAGES FOR DIFFERENT OBJECTS
        for object_name in self.candidate_objects:

            # initialize likelihood image given all features
            self.likelihood_images['all'][object_name] = np.ones(self.image.shape[:2])

            for feature_name in self.use_features:
                self.likelihood_images[feature_name][object_name] = np.ones(self.image.shape[:2])
                # different features are used for different parts of the image, i.e. depending on whether there is 3D data for these parts
                if feature_name in ['color', 'edge', 'miss3D']:
                    self.likelihood_images[feature_name][object_name][self.bin_mask] = Utils.backproject(self.likelihoods[feature_name][object_name], self.feature_images[feature_name][self.bin_mask])
                elif feature_name in ['dist2shelf', 'height3D', 'height2D']:
                    # set a default likelihood for parts in the bin for which the likelihood cannot be computed
                    self.likelihood_images[feature_name][object_name][self.bin_mask] += 1.0 / self.likelihoods[feature_name][object_name].size
                    if feature_name in ['dist2shelf', 'height3D']:
                        self.likelihood_images[feature_name][object_name][self.has3D_mask] = Utils.backproject(self.likelihoods[feature_name][object_name], self.feature_images[feature_name][self.has3D_mask])
                    elif feature_name == 'height2D':
                        self.likelihood_images[feature_name][object_name][self.miss3D_mask] = Utils.backproject(self.likelihoods[feature_name][object_name], self.feature_images[feature_name][self.miss3D_mask])


                # multiply likelihoods
                self.likelihood_images['all'][object_name] *= self.likelihood_images[feature_name][object_name]

    def apply_bayes_rule(self):
        self.posterior_images = dict()
        normalization = np.zeros(self.image.shape[:2])

        for object_name in self.candidate_objects:
            self.posterior_images[object_name] = self.likelihood_images['all'][object_name] * self.prior[object_name]
            normalization += self.posterior_images[object_name]

        for object_name in self.candidate_objects:
            self.posterior_images[object_name] /= normalization

    # feature_images should have the following keys: color, edge, miss3D, dist2shelf, height3D, height2D
    def predict(self, apc_sample, desired_object):

        # true image, we might use this for visualization
        self.image = apc_sample.image
        self.bin_mask = apc_sample.bin_mask
        self.not_bin_mask = np.logical_not(self.bin_mask)
        self.candidate_objects = apc_sample.candidate_objects
        self.desired_object = desired_object
        self.feature_images = apc_sample.feature_images

        # compute mask for which pixels have 3D data and which have not
        self.miss3D_mask = self.feature_images['miss3D'].astype('bool')
        self.has3D_mask = np.logical_not(self.miss3D_mask)

        # MAIN STEPS FOR PREDICTION
        self.backproject()
        self.apply_bayes_rule()

        self.segment_and_all()

        #path = "/home/rico/PHD/2015/workspace/catkin_ws/src/object_recognition/data/experiment_results/video/"

        #utils.display.set_bin_mask(self.bin_mask)
        #utils.display.plot_image(self.image)
        #plt.savefig(path+'{}_cropped.png'.format(desired_object), bbox_inches = 'tight')
        #utils.display.plot_heat_image(self.posterior_images_smooth[desired_object])
        #plt.savefig(path+'{}_posterior.png'.format(desired_object), bbox_inches = 'tight')
        #utils.display.plot_contours(self.image, [self.contours[self.selected_contours[0]]])
        #plt.savefig(path+'{}_segment.png'.format(desired_object), bbox_inches = 'tight')
        #plt.close('all')

        return self.segments[desired_object].astype('bool')