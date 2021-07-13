__author__ = 'rico, Yusuke Niitani'

from copy import deepcopy
import cv2
import os.path

import cPickle as pickle
import matplotlib.pyplot as plt
import numpy as np

import utils
from utils import Utils


class APCSample(object):
    def __init__(self, image_filename=None, apc_sample=None, labeled=True,
                 infer_shelf_mask=False, pickle_mask=False, image_input=None,
                 bin_mask_input=None, data_input=None,
                 data_dict=None, data_2016_prefix=None, is_2016=False):
        """

        Args:
          data_2016_prefix (str): ex: '/leus/home/.../save_pick_layout_1_2016061006_bin_l'
        
        """
        self.image = None
        self.bin_mask = None
        self.feature_images = None
        data = None
        self.is_2016 = is_2016

        # copy properties of a passed APCImage and remove labels if needed
        if apc_sample is not None:
            self.__dict__ = deepcopy(apc_sample.__dict__)
            if not labeled:
                # remove object masks (= labels)
                self.object_masks = dict()

        if image_filename is not None:
            self.load_by_filename(image_filename)
            self.file_prefix = image_filename[-4:]

        if ((image_input is not None)
                and (bin_mask_input is not None)
                and (data_input is not None)):
            # make sure that the image is HSV
            self.image = image_input
            self.bin_mask = bin_mask_input
            self.data_dict = data_input

        if data_2016_prefix is not None:
            self.data_dict = self.get_data_from_2016_path(data_2016_prefix)
            self.image = self.data_dict['color']
            self.bin_mask = self.data_dict['mask_image']
            self.file_prefix = data_2016_prefix 
        elif data_dict is not None:
            self.image = data_dict['color']
            self.bin_mask = data_dict['mask_image']
            self.data_dict = data_dict

        self.feature_images = Utils.compute_feature_images(self.image, self.data_dict)

        self.compute_masked_data(self.data_dict['objects'], labeled, infer_shelf_mask)

    def get_data_from_2016_path(self, path):
        """Create APCSample data_dict from path

        Args:
            path (str): example... '/leus/home/.../save_pick_layout_1_2016061006_bin_l'

        """
        data = {}
        with open(path + '.pkl', 'rb') as f:
            data = pickle.load(f)

        data['color'] = cv2.cvtColor(cv2.imread(path + '.jpg'), cv2.COLOR_BGR2HSV)

        data['mask_image'] = np.sum(cv2.imread(path + '.pbm'), axis=2).astype(np.bool)

        # scale x5 is due to saving procedure
        #data['depth_image'] = cv2.imread(path + '_depth.png').astype(np.float32) * 5.
        #data['dist2shelf_image'] = cv2.imread(path + '_dist.png').astype(np.float32)
        #data['height3D_image'] = cv2.imread(path + '_height.png').astype(np.float32)

        # complete neccesary data
        data['height2D_image'] = np.zeros_like(data['height3D_image'])
        data['has3D_image'] = (data['height3D_image'] > 0).astype(np.uint8) * 255
        return data


    def load_by_filename(self, image_filename):
        # load image, bin_mask, and supplementary data
        bin_mask_filename = image_filename[:-4] + '.pbm'
        bin_pickle_filename = image_filename[:-4] + '_bin' + '.pkl'
        data_filename = image_filename[:-4] + '.pkl'

        self.filenames = {'image': image_filename, 'bin_mask': bin_mask_filename, 'data': data_filename}

        self.image = Utils.load_image(image_filename)
        if pickle_mask:
            self.bin_mask = Utils.load_mask_pkl(bin_pickle_filename)
        else:
            self.bin_mask = Utils.load_mask(bin_mask_filename)
        self.data_dict = Utils.load_supplementary_data(data_filename)

        if self.image is None: print('-> image file not found'); return
        if self.bin_mask is None: print('-> mask file not found'); return
        if self.data_dict is None: print('-> data file not found'); return




    def compute_masked_data(self, objects, labeled, infer_shelf_mask):
        assert self.feature_images is not None
        assert self.bin_mask is not None
        assert self.image is not None
        # compute all features images

        self.candidate_objects = objects + ['shelf']
        self.has_duplicate_objects = len(self.candidate_objects) != len(set(self.candidate_objects))

        self.object_masks = dict()

        if labeled:
            # try to load masks for all objects that are supposed to be in the image
            for object_name in self.candidate_objects:
                mask_filename = \
                    self.file_prefix + '_' + object_name + '.pbm'
                object_mask = Utils.load_mask(mask_filename)
                if object_mask is not None and np.sum(object_mask) != 0:
                    self.object_masks[object_name] = object_mask

            # compute the mask for the shelf if all other objects are masked
            if infer_shelf_mask:
                if (all([object_name in self.object_masks.keys() 
                        for object_name in self.candidate_objects 
                        if object_name != 'shelf']) 
                        and 
                        'shelf' not in self.object_masks.keys()):
                    if len(self.object_masks.keys()) > 1:
                        self.object_masks['shelf'] = np.logical_and(self.bin_mask, np.logical_not(np.logical_or.reduce(self.object_masks.values())))
                    elif len(self.object_masks.keys()) == 1:
                        self.object_masks['shelf'] = np.logical_and(self.bin_mask, np.logical_not(self.object_masks.values()[0]))

        # 'zoom' into the bounding box around the bin_mask
        contours, _hierarchy = cv2.findContours(
            self.bin_mask.astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        x, y, w, h = cv2.boundingRect(contours[0])

        self.bounding_box = {'x':x, 'y':y, 'w':w, 'h':h}
        self.height, self.width, _color = self.image.shape

        self.image = self.image[y:y + h, x:x + w, :]
        self.bin_mask = self.bin_mask[y:y + h, x:x + w]

        for feature_name in self.feature_images.keys():
            self.feature_images[feature_name] = \
                self.feature_images[feature_name][y:y + h, x:x + w]

        for object_name in self.object_masks.keys():
            self.object_masks[object_name] = \
                self.object_masks[object_name][y:y + h, x:x + w]

    def unzoom_segment(self, segment):
        """
        unzoom binary segment
        """
        return_image = np.zeros((self.height, self.width)).astype('uint8')
        x = self.bounding_box['x']
        y = self.bounding_box['y']
        w = self.bounding_box['w']
        h = self.bounding_box['h']

        return_image[y:y+h, x:x+w] = segment
        return return_image


class APCDataSet(object):
    # TODO: fix object_names

    object_names_2015 = ["champion_copper_plus_spark_plug", "kyjen_squeakin_eggs_plush_puppies", "cheezit_big_original",
                      "laugh_out_loud_joke_book", "crayola_64_ct", "mark_twain_huckleberry_finn", "mead_index_cards",
                      "dr_browns_bottle_brush", "mommys_helper_outlet_plugs", "elmers_washable_no_run_school_glue",
                      "munchkin_white_hot_duck_bath_toy", "expo_dry_erase_board_eraser", "oreo_mega_stuf", "first_years_take_and_toss_straw_cup",
                      "paper_mate_12_count_mirado_black_warrior", "genuine_joe_plastic_stir_sticks", "rolodex_jumbo_pencil_cup",
                      "highland_6539_self_stick_notes", "safety_works_safety_glasses", "kong_duck_dog_toy", "sharpie_accent_tank_style_highlighters",
                      "kong_sitting_frog_dog_toy", "stanley_66_052", "feline_greenies_dental_treats", "kong_air_dog_squeakair_tennis_ball", "shelf"]

    object_names = [
        'shelf',
        'barkely_hide_bones',
        'cherokee_easy_tee_shirt',
        'clorox_utility_brush',
        'cloud_b_plush_bear',
        'command_hooks',
        'cool_shot_glue_sticks',
        'crayola_24_ct',
        'creativity_chenille_stems',
        'dasani_water_bottle',
        'dove_beauty_bar',
        'dr_browns_bottle_brush',
        'easter_turtle_sippy_cup',
        'elmers_washable_no_run_school_glue',
        'expo_dry_erase_board_eraser',
        'fiskars_scissors_red',
        'fitness_gear_3lb_dumbbell',
        'folgers_classic_roast_coffee',
        'hanes_tube_socks',
        'i_am_a_bunny_book',
        'jane_eyre_dvd',
        'kleenex_paper_towels',
        'kleenex_tissue_box',
        'kyjen_squeakin_eggs_plush_puppies',
        'laugh_out_loud_joke_book',
        'oral_b_toothbrush_green',
        'oral_b_toothbrush_red',
        'peva_shower_curtain_liner',
        'platinum_pets_dog_bowl',
        'rawlings_baseball',
        'rolodex_jumbo_pencil_cup',
        'safety_first_outlet_plugs',
        'scotch_bubble_mailer',
        'scotch_duct_tape',
        'soft_white_lightbulb',
        'staples_index_cards',
        'ticonderoga_12_pencils',
        'up_glucose_bottle',
        'womens_knit_gloves',
        'woods_extension_cord']

    def __init__(self, name='APCDataSet', samples=[], dataset_path=".", cache_path=".", compute_from_images=False, load_from_cache=False, save_to_cache=False, infer_shelf_masks=False, from_pkl=False):

        self.name = name
        self.samples = samples
        self.dataset_path = dataset_path
        self.cache_path = cache_path

        if load_from_cache:
            self.load_from_cache()
        elif compute_from_images:
            self.samples = [APCSample(image_filename, labeled=True, infer_shelf_mask=infer_shelf_masks) for image_filename in self.apc_image_filenames()]
        elif from_pkl:
            self.samples = []
        if save_to_cache:
            self.save_to_cache()


    def apc_image_filenames(self):
        ''' returns all filenames for image files in the training_data_path'''

        image_filenames = []
        # search for mask (.pbm file) with the same name as the object
        key = '.jpg'
        for root, _dirs, files in os.walk(self.dataset_path):
            for f in files:
                if key == f[-len(key):]:
                    image_filenames.append(os.path.join(root, f))

        return image_filenames

    def apc_2016_data_filenames(self):
        '''Returns 2016 data path'''
        pass

    def cache_filename(self):

        return os.path.join(self.cache_path, self.name.replace("/","_") + '.pkl')


    def load_from_cache(self):

        cache_filename = self.cache_filename()
        if os.path.isfile(cache_filename):
            with open(cache_filename, 'rb') as f:
                loaded = pickle.load(f)
            self.__dict__.update(loaded)
        else:
            print("Cache file not found:\n{}".format(cache_filename))


    def save_to_cache(self):

        cache_filename = self.cache_filename()

        with open(cache_filename, 'wb') as f:
            pickle.dump(self.__dict__, f, 2)


    def split_simple(self, portion_training=0.5, seed=1):

        n = len(self.samples)
        k = int(round(portion_training * n))

        np.random.seed(seed)
        permutation = np.random.permutation(n)

        return self._split(permutation[:k], permutation[k:])


    def _split(self, training_indices, validation_indices):

        training_set = APCDataSet(samples=[self.samples[i] for i in training_indices])
        validation_set = APCDataSet(samples=[self.samples[i] for i in validation_indices])

        return (training_set, validation_set)


    def samples_with_duplicate_objects(self):

        return [sample for sample in self.samples if sample.has_duplicate_objects]

    def without_labels(self):

        return APCDataSet(samples=[APCSample(apc_sample=sample, labeled=False) for sample in self.samples])

#         self.training_set = dict()
#
#         for object_name in self.object_names:
#
#             total_num_images = len(self.object_samples[object_name].filenames['image'])
#             if total_num_images == 0:
#                 print('-> no training samples for {}!!!'.format(object_name))
#                 self.training_set[object_name] = np.array([])
#             else:
#                 # draw a random subset for training
#                 choice = np.random.choice(total_num_images, num_traning_images_per_object, False)
#                 self.training_set[object_name] = np.array([x in choice for x in range(total_num_images)])

    def rbo_training_data_filenames(self, object_name):
        ''' returns all filenames for images, masks, and pkl files in the training_data_path for a given object name'''

        # search for mask (.pbm file) with the same name as the object
        key = object_name + '.pbm'
        mask_filenames = []
        for root, dirs, files in os.walk(self.dataset_path):
            for f in files:
                if key == f[-len(key):]:
                    mask_filenames.append(os.path.join(root, f))

        # add filename associated to the mask file
        image_filenames = [f[:-(len(key) + 1)] + ".jpg" for f in mask_filenames]
        pkl_filenames = [f[:-(len(key) + 1)] + ".pkl" for f in mask_filenames]

        return (image_filenames, mask_filenames, pkl_filenames)

    def visualize_dataset(self):

        print('visualizing the entire dataset')

        i = 0
        for j, sample in enumerate(self.samples):
           if sample.filenames['image'].split('/')[-1] == '2015-05-09_12-24-42-683421_bin_H.jpg':
               i = j
        #i = 165

        for j, sample in enumerate(self.samples[i:]):

            print("{}/{}: {}".format(i + j, len(self.samples), sample.filenames['image'].split('/')[-1]))
            print(sample.candidate_objects)
            for obj in sample.candidate_objects:
                if obj in sample.object_masks.keys():
                    utils.display.plot_segment(sample.image, sample.object_masks[obj], title=obj)
            plt.draw()
            raw_input('Press Enter for next sample...')
            plt.close('all')
