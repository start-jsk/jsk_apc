from itertools import combinations
from twisted.python.versions import _inf
import time

__author__ = 'rico jonschkowski'

import matplotlib.pyplot as plt
import matplotlib.colorbar
import numpy as np
import cv2
import os, sys
import cPickle as pickle
import copy

import utils
from apc_data import APCDataSet, APCSample
from probabilistic_segmentation import ProbabilisticSegmentationRF, ProbabilisticSegmentationBP

# http://colorbrewer2.org/?type=qualitative&scheme=Dark2&n=8
scheme_dark2 = np.array([(27,158,119),(217,95,2),(117,112,179),(231,41,138),(102,166,30),(230,171,2),(166,118,29),(102,102,102)])/255.0

# implements F_beta score: the weighted harmonic mean between precision and recal.as
# beta measures the importance of recall.
def f_score(precision, recall, beta=1):
    if precision == recall == 0.0:
        return 0.0
    else:
        return (1.0 + beta**2) * precision * recall / (beta**2 * precision + recall)

def train_and_evaluate(rf, training_set, evaluation_sets_dict, show_images=False, save_image_path=None, withhold_object_info=False, add_candidate_objects=0):
        
    # fit
    rf.fit(training_set)  

    precisions = dict()
    recalls = dict()
    candidate_objects = dict()
    desired_objects = dict()

    # test
    for mode, set in evaluation_sets_dict.iteritems():
        # print (mode)

        i = 0

        precisions[mode] = []
        recalls[mode] = []
        candidate_objects[mode] = []
        desired_objects[mode] = []
        for sample in set.samples:

            # copy original sample, because sample might be
            orig_sample = sample
            sys.stdout.write('.')
            sys.stdout.flush()
            i += 1
            for desired_object in sample.object_masks.keys():
                sample = copy.deepcopy(orig_sample)
                if desired_object != 'shelf':

                    true_segment = sample.object_masks[desired_object]

                    if add_candidate_objects > 0:

                        other_objects = [object_names for object_names in APCDataSet.object_names if object_names not in sample.candidate_objects]
                        if other_objects == []:
                            print("empty list ... this should not happen")
                        sample.candidate_objects += np.random.choice(other_objects, add_candidate_objects, replace=False).tolist()

                    if withhold_object_info:
                        sample.candidate_objects = APCDataSet.object_names

                    predicted_segment = rf.predict(APCSample(apc_sample=sample, labeled=False), desired_object)

                    true_positives = np.sum(np.logical_and(true_segment, predicted_segment)).astype('double')
                    positives = np.sum(predicted_segment.astype('bool')).astype('double')
                    relevant = np.sum(true_segment.astype('bool')).astype('double')
                    
                    precision = true_positives / positives if positives != 0 else 0.0
                    recall = true_positives / relevant
                    #print('{} -> precision: {:.2}, recall: {:.2}'.format(mode, precision, recall))
                    precisions[mode].append(precision)
                    recalls[mode].append(recall)
                    desired_objects[mode].append(desired_object)
                    candidate_objects[mode].append(sample.candidate_objects)

                    if show_images:
                        utils.display.set_bin_mask(rf.bin_mask)
                        #utils.display.plot_segment(rf.image, predicted_segment, title='predicted {}'.format(desired_object))

                        cnts = cv2.findContours((predicted_segment>0).astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
                        if cnts:
                            cnt = cnts[0]
                            utils.display.plot_contours(rf.image, [cnt], title='predicted {}'.format(desired_object))

                            plt.draw()

                            if save_image_path:
                                plt.savefig(save_image_path+'{}_{}_{}.png'.format(i, mode, desired_object), bbox_inches = 'tight')

                        plt.close('all')

        print

        #print('\r{} mean precision: {:.2}'.format(mode, np.mean(precisions[mode])))
        #print('{} mean recall: {:.2}'.format(mode, np.mean(recalls[mode])))
        #print('{} f_0.5 score: {:.3}'.format(mode, f_score(np.mean(precisions[mode]), np.mean(recalls[mode]), 0.5)))
            
    return {'precisions': precisions, 'recalls': recalls, 'candidate_objects': candidate_objects, 'desired_objects': desired_objects}

def combine_datasets(datasets):
    samples = []
    for d in datasets:
        samples += d.samples
    return APCDataSet(samples = samples)

def load_datasets(dataset_names, data_path, cache_path):

    datasets = dict()

    for dataset_name in dataset_names:
        dataset_path = os.path.join(data_path, 'rbo_apc/{}'.format(dataset_name))
        datasets[dataset_name] = APCDataSet(name=dataset_name, dataset_path=dataset_path, cache_path=cache_path, load_from_cache=True)

    return datasets

def compute_datasets(dataset_names, data_path, cache_path):

    datasets = dict()

    for dataset_name in dataset_names:
        dataset_path = os.path.join(data_path, 'rbo_apc/{}'.format(dataset_name))
        if 'runs' in dataset_name:
            datasets[dataset_name] = APCDataSet(name=dataset_name, dataset_path=dataset_path, cache_path=cache_path, compute_from_images=True, save_to_cache=True, infer_shelf_masks=True)
        else:
            datasets[dataset_name] = APCDataSet(name=dataset_name, dataset_path=dataset_path, cache_path=cache_path, compute_from_images=True, save_to_cache=True, infer_shelf_masks=False)

    return datasets

# EXPERIMENTS FROM THE PAPER

# A. Performance Evaluation
# 1) Performance in the Amazon Picking Challenge

def experiment_APC(datasets, results_path, only_berlin_data=False):

    path = os.path.join(results_path, 'experiment_APC/')
    if not os.path.isdir(path):
        os.makedirs(path)

    time_string = time.strftime("%Y-%m-%d_%H-%M")
    result_file = os.path.join(path,time_string+'.pkl')

    params = {'use_features':['color', 'height2D', 'edge', 'miss3D', 'height3D', 'dist2shelf'], # 'height2D''edge'
         'segmentation_method':"max_smooth", 'selection_method': "max_smooth",
         'make_convex':True, 'do_shrinking_resegmentation':True, 'do_greedy_resegmentation':True}

    methods = {
            'our_method': ProbabilisticSegmentationBP(**params),
        }

    if only_berlin_data:
        datasets['training_berlin_and_seattle'] = combine_datasets([datasets['berlin_selected'], datasets['berlin_runs']])
    else:
        datasets['training_berlin_and_seattle'] = combine_datasets([datasets['berlin_selected'], datasets['berlin_runs'], datasets['seattle_runs']])

    results = dict()

    for method_name, method in methods.iteritems():

        results[method_name] = train_and_evaluate(method, datasets['training_berlin_and_seattle'],
            {name: datasets[name] for name in ['seattle_test']},
             show_images=True, save_image_path=path)

        with open(result_file, 'wb') as f:
            pickle.dump(results, f, 2)

    print('Segmentation results are written to {}'.format(path))

    return result_file

# 2) Performance by Object

def experiment_our_method(datasets, results_path):

    path = os.path.join(results_path, 'experiment_our_method/')
    if not os.path.isdir(path):
        os.makedirs(path)

    time_string = time.strftime("%Y-%m-%d_%H-%M")
    filename_results = os.path.join(path,time_string+'.pkl')

    params = {'use_features':['color', 'height2D', 'edge', 'miss3D', 'height3D', 'dist2shelf'], # 'height2D''edge'
         'segmentation_method':"max_smooth", 'selection_method': "max_smooth",
         'make_convex':True, 'do_shrinking_resegmentation':True, 'do_greedy_resegmentation':True}

    methods = {
            'our_method': ProbabilisticSegmentationBP(**params),
    }

    results = dict()

    for method_name, method in methods.iteritems():

        results[method_name] = train_and_evaluate(method, datasets['berlin_selected'],
            {name: datasets[name] for name in ['berlin_selected','berlin_runs', 'seattle_runs', 'seattle_test']}, show_images=False, save_image_path=path)

        with open(filename_results, 'wb') as f:
            pickle.dump(results, f, 2)

    return filename_results

def display_experiment_objects(filename):

    object_names = ["champion_copper_plus_spark_plug", "kyjen_squeakin_eggs_plush_puppies", "cheezit_big_original",
                      "laugh_out_loud_joke_book", "crayola_64_ct", "mark_twain_huckleberry_finn", "mead_index_cards",
                      "dr_browns_bottle_brush", "mommys_helper_outlet_plugs", "elmers_washable_no_run_school_glue",
                      "munchkin_white_hot_duck_bath_toy", "expo_dry_erase_board_eraser", "oreo_mega_stuf", "first_years_take_and_toss_straw_cup",
                      "paper_mate_12_count_mirado_black_warrior", "genuine_joe_plastic_stir_sticks", "rolodex_jumbo_pencil_cup",
                      "highland_6539_self_stick_notes", "safety_works_safety_glasses", "kong_duck_dog_toy", "sharpie_accent_tank_style_highlighters",
                      "kong_sitting_frog_dog_toy", "stanley_66_052", "feline_greenies_dental_treats", "kong_air_dog_squeakair_tennis_ball", "shelf"]

    short_names = ["spark_plug", "plush_eggs", "cheezits",
                      "joke_book", "crayons", "mark_twain_book", "index_cards",
                      "bottle_brush", "outlet_plugs", "glue",
                      "bath_duck", "board_eraser", "oreos", "straw_cup",
                      "paper_mate", "stir_sticks", "pencil_cup",
                      "stick_notes", "safety_glasses", "duck_toy", "highlighters",
                      "frog_toy", "stanley_66", "dental_treats", "tennis_ball", "shelf"]

    import cPickle as pickle

    with open(filename,'rb') as f:
        results = pickle.load(f)

    # only use first mode to compute the statistics
    results = results['our_method']

    # initialize object statistics
    object_statistics = dict()
    for object_name in APCDataSet.object_names:
        object_statistics[object_name] = dict()
        for measure in ['recalls', 'precisions']:
            object_statistics[object_name][measure] = []

    # fill object statistics
    for run in ['seattle_runs', 'berlin_runs']:
        for object_name, recall, precision in zip(results['desired_objects'][run], results['recalls'][run], results['precisions'][run]):
            object_statistics[object_name]['recalls'].append(recall)
            object_statistics[object_name]['precisions'].append(precision)


    # initialize # statistics
    number_statistics = dict()
    for number in range(1,5):
        object_statistics[number] = dict()
        for measure in ['recalls', 'precisions']:
            object_statistics[number][measure] = []

    # fill # statistics
    for run in ['seattle_runs', 'berlin_runs']:
        for candidate_objects, recall, precision in zip(results['candidate_objects'][run], results['recalls'][run], results['precisions'][run]):
            number = len(candidate_objects) - 1 # subtract shelf
            object_statistics[number]['recalls'].append(recall)
            object_statistics[number]['precisions'].append(precision)

    # plot statistics
    plt.figure()
    plt.hold(True)

    cmap = plt.get_cmap('coolwarm')
    plt.set_cmap(cmap)

    n = 100
    f_image = np.zeros([n,n])
    for i in range(n):
        for j in range(n):
            f_image[n-(j+1), i] = f_score(j/(n-1.), i/(n-1.), 0.5)

    plt.imshow(f_image, extent=[0,1,0,1], aspect='auto')

    text_offset_x = 0.01
    text_offset_y = 0.00

    import matplotlib as mpl
    mpl.rcParams['font.size'] = 12

    for object_name, short_name in zip(object_names, short_names):
        if object_name != 'shelf':
            recall = np.mean(object_statistics[object_name]['recalls'])
            precision = np.mean(object_statistics[object_name]['precisions'])
            f = f_score(precision, recall, 0.5)
            plt.text(recall+text_offset_x, precision+text_offset_y, short_name)
            plt.plot(recall, precision, '.k', label=object_name, markersize=10)

    recalls = []
    precisions = []
    for number in range(1,4):
        recall = np.mean(object_statistics[number]['recalls'])
        precision = np.mean(object_statistics[number]['precisions'])
        recalls.append(recall)
        precisions.append(precision)
        f = f_score(precision, recall, 0.5)
        plt.text(recall+text_offset_x, precision+text_offset_y, repr(number), color='w')
        plt.plot(recall, precision, '.', color='w', markersize=10)
    plt.plot(recalls, precisions, '-w')


    plt.colorbar(label='F0.5 score')

    #matplotlib.colorbar.ColorbarBase(plt.gca(), cmap=cmap)

    plt.axis('scaled')
    plt.axis([0.4,1.0,0.4,1.0])
    plt.xlabel('Recall')
    plt.ylabel('Precision')

    figure_name = filename.replace('.pkl', '.svg')
    print('Saved plot to {}'.format(figure_name))
    plt.savefig(figure_name)

# 3) Increasing the Number of Objects per Bin

def experiment_candidates(datasets, results_path):

    path = os.path.join(results_path, 'experiment_candidates/')
    if not os.path.isdir(path):
        os.makedirs(path)

    time_string = time.strftime("%Y-%m-%d_%H-%M")
    filename_results = os.path.join(path,time_string+'.pkl')

    params = {'use_features':['color', 'height2D', 'edge', 'miss3D', 'height3D', 'dist2shelf'], # 'height2D''edge'
         'segmentation_method':"max_smooth", 'selection_method': "max_smooth",
         'make_convex':True, 'do_shrinking_resegmentation':True, 'do_greedy_resegmentation':True}

    methods = {
            'our_method': ProbabilisticSegmentationBP(**params),
    }

    results = dict()

    for method_name, method in methods.iteritems():

        for i in range(0,21):

            name  = method_name + repr(i)

            results[name] = train_and_evaluate(method, datasets['berlin_selected'],
                {n: datasets[n] for n in ['berlin_selected','berlin_runs', 'seattle_runs']}, add_candidate_objects=i)

            with open(filename_results, 'wb') as f:
                pickle.dump(results, f, 2)

    return filename_results

def display_experiment_candidates(filename):

    results = dict()
    with open(filename,'rb') as f:
        results.update(pickle.load(f))

    dataset_names = ['berlin_selected', 'berlin_runs', 'seattle_runs']

    model_names = ['our_method'+repr(i) for i in range(len(results.keys()))]

    plt.figure()
    plt.hold(True)

    for dataset_name in ['seattle_runs']:
        recalls = []
        precisions = []
        f_scores = []
        for i, model_name in enumerate(model_names):

            recalls.append(np.mean(results[model_name]['recalls'][dataset_name]))
            precisions.append(np.mean(results[model_name]['precisions'][dataset_name]))
            f_scores.append(f_score(precisions[-1], recalls[-1], beta=0.5))

        plt.plot(recalls, label='Recall', color=scheme_dark2[0])
        plt.plot(precisions, label='Precision', color=scheme_dark2[1])
        plt.plot(f_scores, label='F0.5 score', color=scheme_dark2[2])

    plt.axis([0,20,0,0.8])
    plt.xlabel('# added candidate objects')

    plt.legend(loc='lower left')

    figure_name = filename.replace('.pkl', '.svg')
    print('Saved plot to {}'.format(figure_name))
    plt.savefig(figure_name)

# B. Comparison to CRF

def experiment_baseline(datasets, results_path):

    path = os.path.join(results_path, 'experiment_baseline/')
    if not os.path.isdir(path):
        os.makedirs(path)

    time_string = time.strftime("%Y-%m-%d_%H-%M")
    filename_results = os.path.join(path,time_string+'.pkl')

    params_our_method = {'use_features':['color', 'height2D', 'edge', 'miss3D', 'height3D', 'dist2shelf'],
         'segmentation_method':"max_smooth", 'selection_method': "max_smooth",
         'make_convex':True, 'do_shrinking_resegmentation':True, 'do_greedy_resegmentation':True}

    params_baseline_rgbd = {'use_features':['red', 'green', 'blue', 'depth'],
         'segmentation_method':"simple_cut", 'selection_method': "largest",
         'make_convex':False, 'do_shrinking_resegmentation':False, 'do_greedy_resegmentation':False}

    methods = {
            'Our method': ProbabilisticSegmentationBP(**params_our_method),
            'CRF RGB-D (0.00)': ProbabilisticSegmentationRF({'n_estimators':100, 'min_weight_fraction_leaf':0.0}, lazy=False, **params_baseline_rgbd),
            'CRF RGB-D (0.005)': ProbabilisticSegmentationRF({'n_estimators':100, 'min_weight_fraction_leaf':0.005}, lazy=False, **params_baseline_rgbd),
            'CRF RGB-D (0.01)': ProbabilisticSegmentationRF({'n_estimators':100, 'min_weight_fraction_leaf':0.01},lazy=False, **params_baseline_rgbd),
            'CRF RGB-D (0.02)': ProbabilisticSegmentationRF({'n_estimators':100, 'min_weight_fraction_leaf':0.02},lazy=False, **params_baseline_rgbd),
            'CRF RGB-D (0.04)': ProbabilisticSegmentationRF({'n_estimators':100, 'min_weight_fraction_leaf':0.04},lazy=False, **params_baseline_rgbd),
            'CRF RGB-D (0.08)': ProbabilisticSegmentationRF({'n_estimators':100, 'min_weight_fraction_leaf':0.08},lazy=False, **params_baseline_rgbd)
            #'CRF RGB-D (0.00) lazy': ProbabilisticSegmentationRF({'n_estimators':100, 'min_weight_fraction_leaf':0.0}, lazy=True, **params_baseline_rgbd),
            #'CRF RGB-D (0.005) lazy': ProbabilisticSegmentationRF({'n_estimators':100, 'min_weight_fraction_leaf':0.005}, lazy=True, **params_baseline_rgbd),
            #'CRF RGB-D (0.01) lazy': ProbabilisticSegmentationRF({'n_estimators':100, 'min_weight_fraction_leaf':0.01},lazy=True, **params_baseline_rgbd),
            #'CRF RGB-D (0.02) lazy': ProbabilisticSegmentationRF({'n_estimators':100, 'min_weight_fraction_leaf':0.02},lazy=True, **params_baseline_rgbd),
            #'CRF RGB-D (0.04) lazy': ProbabilisticSegmentationRF({'n_estimators':100, 'min_weight_fraction_leaf':0.04},lazy=True, **params_baseline_rgbd),
            #'CRF RGB-D (0.08) lazy': ProbabilisticSegmentationRF({'n_estimators':100, 'min_weight_fraction_leaf':0.08},lazy=True, **params_baseline_rgbd)
        }

    results = dict()

    for method_name, rf in methods.iteritems():

        results[method_name] = train_and_evaluate(rf, datasets['berlin_selected'],
            {name: datasets[name] for name in ['berlin_selected', 'berlin_runs', 'seattle_runs', 'seattle_test']},
             show_images=False, save_image_path=path+method_name+'_')

        with open(filename_results, 'wb') as f:
            pickle.dump(results, f, 2)

    return filename_results

def display_experiment_baseline(filename):

    import cPickle as pickle
    with open(filename,'rb') as f:
        results = pickle.load(f)

    dataset_names = ['berlin_selected', 'berlin_runs', 'seattle_runs', 'seattle_test']

    model_names = ['CRF RGB-D (0.00)', 'CRF RGB-D (0.005)', 'CRF RGB-D (0.01)', 'CRF RGB-D (0.02)', 'CRF RGB-D (0.04)', 'Our method']
    colors = {'CRF RGB-D (0.00)': 1.1*scheme_dark2[5], 'CRF RGB-D (0.005)': 0.95*scheme_dark2[5], 'CRF RGB-D (0.01)': 0.8*scheme_dark2[5], 'CRF RGB-D (0.02)': 0.65*scheme_dark2[5], 'CRF RGB-D (0.04)': 0.5*scheme_dark2[5], 'Our method': scheme_dark2[4]}

    #model_names = ['CRF RGB-D (0.00) lazy', 'CRF RGB-D (0.005) lazy', 'CRF RGB-D (0.01) lazy', 'CRF RGB-D (0.02) lazy', 'CRF RGB-D (0.04) lazy', 'Our method']
    #colors = {'CRF RGB-D (0.00) lazy': 1.1*scheme_dark2[5], 'CRF RGB-D (0.005) lazy': 0.95*scheme_dark2[5], 'CRF RGB-D (0.01) lazy': 0.8*scheme_dark2[5], 'CRF RGB-D (0.02) lazy': 0.65*scheme_dark2[5], 'CRF RGB-D (0.04) lazy': 0.5*scheme_dark2[5], 'Our method': scheme_dark2[4]}

    plt.figure()
    plt.hold(True)
    width_all = 0.8
    width_bar_rel = 0.9
    width_bar = width_all/len(model_names)
    x = np.arange(len(dataset_names)) - width_all/2
    for i, model_name in enumerate(model_names):
        recalls = [np.mean(results[model_name]['recalls'][dataset_name]) for dataset_name in dataset_names]
        precisions = [np.mean(results[model_name]['precisions'][dataset_name]) for dataset_name in dataset_names]
        f_scores = [f_score(precision, recall, beta=0.5) for precision, recall in zip(precisions, recalls)]
        plt.bar(x + i*width_bar, f_scores, width=width_bar*width_bar_rel, label=model_name.replace('our method', 'Our method'), color=colors[model_name], linewidth=0)

    plt.gca().set_ylabel('F0.5 score')
    plt.gca().set_xticks(range(len(dataset_names)))
    plt.gca().set_xticklabels(('Training (Berlin)', 'Test (Berlin)', 'Test (Seattle)', 'APC (Seattle)'))

    plt.legend(loc='lower left')

    figure_name = filename.replace('.pkl', '.svg')
    print('Saved plot to {}'.format(figure_name))
    plt.savefig(figure_name)


# C. Variants of the Algorithm
# 1) Changing Features

def experiment_single_features(datasets, results_path):

    path = os.path.join(results_path, 'experiment_single_features/')
    if not os.path.isdir(path):
        os.makedirs(path)

    time_string = time.strftime("%Y-%m-%d_%H-%M")
    filename_results = os.path.join(path,time_string+'.pkl')

    all_features = ['color', 'edge', 'miss3D', 'height2D', 'height3D', 'dist2shelf']
    params = {'use_features':all_features,
         'segmentation_method':"max_smooth", 'selection_method': "max_smooth",
         'make_convex':True, 'do_shrinking_resegmentation':True, 'do_greedy_resegmentation':True}

    param_dict = dict()
    for feature_name in all_features:
        param_dict[feature_name] = copy.deepcopy(params)
        param_dict[feature_name]['use_features'] = [feature_name]

    methods = {'only_'+feature_name: ProbabilisticSegmentationBP(**param_dict[feature_name]) for feature_name in all_features}
    methods['all'] = ProbabilisticSegmentationBP(**params)

    results = dict()

    for method_name, method in methods.iteritems():

        results[method_name] = train_and_evaluate(method, datasets['berlin_selected'],
            {name: datasets[name] for name in ['berlin_selected', 'berlin_runs', 'seattle_runs']})

        with open(filename_results, 'wb') as f:
            pickle.dump(results, f, 2)

    return filename_results

def experiment_removed_features(datasets, results_path):

    path = os.path.join(results_path, 'experiment_removed_features/')
    if not os.path.isdir(path):
        os.makedirs(path)

    time_string = time.strftime("%Y-%m-%d_%H-%M")
    filename_results = os.path.join(path,time_string+'.pkl')

    all_features = ['color', 'edge', 'miss3D', 'height2D', 'height3D', 'dist2shelf']

    params = {'use_features':all_features,
         'segmentation_method':"max_smooth", 'selection_method': "max_smooth",
         'make_convex':True, 'do_shrinking_resegmentation':True, 'do_greedy_resegmentation':True}

    param_dict = dict()
    for feature_name in all_features:
        param_dict[feature_name] = copy.deepcopy(params)
        param_dict[feature_name]['use_features'] = [n for n in all_features if n != feature_name]

    methods = {'all_except_'+feature_name: ProbabilisticSegmentationBP(**param_dict[feature_name]) for feature_name in all_features}
    methods['all'] = ProbabilisticSegmentationBP(**params)

    results = dict()

    for method_name, method in methods.iteritems():

        results[method_name] = train_and_evaluate(method, datasets['berlin_selected'],
            {name: datasets[name] for name in ['berlin_selected', 'berlin_runs', 'seattle_runs']})

        with open(filename_results, 'wb') as f:
            pickle.dump(results, f, 2)

    return filename_results

def display_experiment_features(filenames):

    import cPickle as pickle

    results = dict()
    for filename in filenames:
        with open(filename,'rb') as f:
            results.update(pickle.load(f))

    model_names = sorted(results.keys())
    #dataset_names = sorted(results[model_names[0]]['recalls'].keys())
    dataset_names = ['berlin_runs', 'seattle_runs']

    def flatten(list_of_lists):
        if type(list_of_lists[0]) == type([]):
            flattened = []
            for list in list_of_lists:
                flattened += list
            return flattened
        else:
            return list_of_lists


    model_names_only = ['all', 'only_color', 'only_edge', 'only_miss3D', 'only_height2D', 'only_height3D', 'only_dist2shelf']
    model_names_except = [model_name.replace('only_', 'all_except_') for model_name in model_names_only if 'all' not in model_name]
    colors = {'all': scheme_dark2[7], 'color': scheme_dark2[5], 'edge': scheme_dark2[4], 'miss3D': scheme_dark2[3], 'height2D': scheme_dark2[2], 'height3D': scheme_dark2[1], 'dist2shelf': scheme_dark2[0]}

    plt.figure()
    plt.hold(True)
    width_all = 0.6
    width_bar_rel = 0.9
    width_bar = width_all/len(model_names_only)
    x = np.arange(len(dataset_names)) - width_all/2
    for i, model_name in enumerate(model_names_only):
        recalls = [np.mean(results[model_name]['recalls'][dataset_name]) for dataset_name in dataset_names]
        precisions = [np.mean(results[model_name]['precisions'][dataset_name]) for dataset_name in dataset_names]
        f_scores = [f_score(precision, recall, beta=0.5) for precision, recall in zip(precisions, recalls)]
        if model_name == 'all':
            f_scores_all = f_scores
            plt.bar(x + (i-0.75)*width_bar, np.clip(f_scores,0.01,1), width=width_bar*width_bar_rel*2, label=model_name.replace('only_', ''), color=colors[model_name.replace('only_', '')], linewidth=0)
            for i in range(2):
                plt.plot([x[i]+0.1, x[i] + 0.7], [f_scores[i], f_scores[i]], '--', color=colors['all'])

        else:
            plt.bar(x + (i+0.75)*width_bar, np.clip(f_scores,0.01,1), width=width_bar*width_bar_rel, label=model_name.replace('only_', ''), color=colors[model_name.replace('only_', '')], linewidth=0)

    for i, model_name in enumerate(model_names_except):
        recalls = [np.mean(results[model_name]['recalls'][dataset_name]) for dataset_name in dataset_names]
        precisions = [np.mean(results[model_name]['precisions'][dataset_name]) for dataset_name in dataset_names]
        f_scores = [f_score(precision, recall, beta=0.5) for precision, recall in zip(precisions, recalls)]
        plt.bar(x + (i+1+0.75)*width_bar, -np.array(f_scores_all) + np.array(f_scores), bottom=f_scores_all, width=width_bar*width_bar_rel, facecolor='w', linewidth = 2, edgecolor=colors[model_name.replace('all_except_', '')])

    plt.gca().set_ylabel('F0.5 score')
    plt.gca().set_xticks(range(len(dataset_names)))
    plt.gca().set_xticklabels(('Test (Berlin)', 'Test (Seattle)'))

    plt.legend(loc='lower left')

    plt.savefig(filename.replace('.pkl', '.svg'))

    plt.show()

    plt.figure()
    plt.hold(True)
    width_all = 0.7
    width_bar = width_all/len(model_names)
    x = np.arange(len(dataset_names)) - width_all/2
    for i, model_name in enumerate(model_names):
        recalls = [np.mean(flatten(results[model_name]['recalls'][dataset_name])) for dataset_name in dataset_names]
        precisions = [np.mean(flatten(results[model_name]['precisions'][dataset_name])) for dataset_name in dataset_names]
        f_scores = [f_score(precision, recall, beta=0.5) for precision, recall in zip(precisions, recalls)]
        #print(model_name, f_scores)
        plt.bar(x + i*width_bar, f_scores, width=width_bar, label=' '+model_name, color=colors[model_name])


    plt.axis([-0.5,0.8 + len(dataset_names),0,1])

    plt.gca().set_ylabel('F_0.5 score')
    plt.gca().set_xticks(range(len(dataset_names)))
    #plt.gca().set_xticklabels(('Trainnig (Berlin)', 'Runs (Berlin)', 'Runs (Seattle)'))
    plt.gca().set_xticklabels(dataset_names)

    plt.legend(loc='lower right')

    figure_name = filename.replace('.pkl', '.svg')
    print('Saved plot to {}'.format(figure_name))
    plt.savefig(figure_name)

# 2) Pixel Labeling and Selection

def experiment_segmentation(datasets, results_path):

    path = os.path.join(results_path, 'experiment_segmentation/')
    if not os.path.isdir(path):
        os.makedirs(path)

    time_string = time.strftime("%Y-%m-%d_%H-%M")
    filename_results = os.path.join(path,time_string+'.pkl')

    params = {'use_features':['color', 'edge', 'miss3D', 'height3D', 'height2D', 'dist2shelf'],
         'make_convex':False, 'do_shrinking_resegmentation':False, 'do_greedy_resegmentation':False}

    results = dict()

    for segmentation_method in ["max", "max_smooth", "simple_cut", "edge_cut"]:
        for selection_method in ["all", "max_smooth", "largest"]:
            params['segmentation_method'] = segmentation_method
            params['selection_method'] = selection_method
            method = ProbabilisticSegmentationBP(**params)

            name = "seg_{}_sel_{}".format(segmentation_method, selection_method)

            results[name] = train_and_evaluate(method, datasets['berlin_selected'],
                {name: datasets[name] for name in ['berlin_selected','berlin_runs', 'seattle_runs']})

            with open(filename_results, 'wb') as f:
                pickle.dump(results, f, 2)

    return filename_results

def display_experiment_segmentation(filename):

    with open(filename,'rb') as f:
        results = pickle.load(f)

    dataset_name = 'seattle_runs'

    plt.figure()
    plt.hold(True)

    segmentation_methods = ['seg_max', 'seg_max_smooth', 'seg_simple_cut', 'seg_edge_cut']
    selection_methods = ['sel_all', 'sel_largest', 'sel_max_smooth']

    colors = scheme_dark2
    markers = ['^','o', '*']

    for i, segmentation_method in enumerate(segmentation_methods):
        for j, selection_method in enumerate(selection_methods):
            model_name = segmentation_method + '_' + selection_method
            recall = np.mean(results[model_name]['recalls'][dataset_name])
            precision = np.mean(results[model_name]['precisions'][dataset_name])

            if markers[j] == '*':
                markersize = 15
            else:
                markersize = 10
            plt.plot(recall, precision, color='k', markeredgecolor='k', marker=markers[j], markersize=markersize, label=model_name)

    plt.axis('scaled')
    plt.axis([0.4,0.7,0.5,0.8])
    plt.xticks(np.arange(0.4,0.8,0.1))
    plt.yticks(np.arange(0.5,0.9,0.1))

    plt.xlabel('Recall')
    plt.ylabel('Precision')

    #plt.gca().set_ylabel('F_0.5 score')
    #plt.gca().set_xticks(range(3))
    #plt.gca().set_xticklabels(('Training (Berlin)', 'Runs (Berlin)', 'Runs (Seattle)'))

    plt.legend(loc='lower left')

    figure_name = filename.replace('.pkl', '.svg')
    print('Saved plot to {}'.format(figure_name))
    plt.savefig(figure_name)

# 3) Re-Labeling and Post-Processing

def experiment_reseg_and_postproc(datasets, results_path):

    path = os.path.join(results_path, 'experiment_reseg_and_postproc/')
    if not os.path.isdir(path):
        os.makedirs(path)

    time_string = time.strftime("%Y-%m-%d_%H-%M")
    filename_results = os.path.join(path,time_string+'.pkl')

    params = {'use_features':['color', 'edge', 'miss3D', 'height3D', 'height2D', 'dist2shelf'],
         'segmentation_method':"max_smooth", 'selection_method': "max_smooth",
         'make_convex':False, 'do_shrinking_resegmentation':False, 'do_greedy_resegmentation':False}

    results = dict()

    for make_convex in [True, False]:
        for do_shrinking_resegmentation in [True, False]:
            for do_greedy_resegmentation in [True, False]:
                params['make_convex'] = make_convex
                params['do_shrinking_resegmentation'] = do_shrinking_resegmentation
                params['do_greedy_resegmentation'] = do_greedy_resegmentation

                method = ProbabilisticSegmentationBP(**params)
                name = "use"
                if make_convex:
                    name += "_convex"
                if do_shrinking_resegmentation:
                    name += "_shrinking"
                if do_greedy_resegmentation:
                    name += "_greedy"

                results[name] = train_and_evaluate(method, datasets['berlin_selected'],
                    {name: datasets[name] for name in ['berlin_selected','berlin_runs', 'seattle_runs']})

                with open(filename_results, 'wb') as f:
                    pickle.dump(results, f, 2)

    return filename_results

def display_experiment_reseg_and_postproc(filename):

    results = dict()
    with open(filename,'rb') as f:
        results.update(pickle.load(f))

    dataset_name = 'seattle_runs'

    model_names = results.keys()

    recalls = np.zeros((4,3))
    precisions = np.zeros((4,3))
    f_scores = np.zeros((4,3))

    for convex in range(2):
        for shrinking in range(2):
            for greedy in range(2):
                color = scheme_dark2[6] if greedy else (0.0,0.0,0.0)
                marker = 'h' if convex else '*' # or 'H'
                facecolor = color if shrinking else (1.0,1.0,1.0)

                model_name = 'use'
                if convex:
                    model_name += '_convex'
                if shrinking:
                    model_name += '_shrinking'
                if greedy:
                    model_name += '_greedy'


                recall = np.mean(results[model_name]['recalls'][dataset_name])
                precision = np.mean(results[model_name]['precisions'][dataset_name])

                #print(model_name + " {:.2} {:.2}".format(recall, precision))

                if marker == "*":
                    markersize = 15
                else:
                    markersize = 10
                plt.plot(recall, precision, linewidth=0, markeredgecolor=color, markerfacecolor=facecolor,  marker=marker, markersize=markersize, mew=2, label=model_name)

    plt.axis('scaled')
    plt.axis([0.55,0.75,0.62,0.82])
    #plt.xticks(np.arange(0.4,0.8,0.1))
    #plt.yticks(np.arange(0.5,0.9,0.1))

    plt.xlabel('Recall')
    plt.ylabel('Precision')

    #plt.gca().set_ylabel('F_0.5 score')
    #plt.gca().set_xticks(range(3))
    #plt.gca().set_xticklabels(('Training (Berlin)', 'Runs (Berlin)', 'Runs (Seattle)'))

    plt.legend(loc='lower left')

    figure_name = filename.replace('.pkl', '.svg')
    print('Saved plot to {}'.format(figure_name))
    plt.savefig(figure_name)

def display_experiment_reseg_and_postproc_old(filename):

    with open(filename,'rb') as f:
        results = pickle.load(f)

    #dataset_names = ['berlin_selected', 'berlin_runs', 'seattle_runs']
    #dataset_names = ['training_berlin_and_seattle', 'berlin_runs', 'seattle_runs_b']
    dataset_names = ['seattle_runs']

    cmap = plt.get_cmap('coolwarm_r')
    #plt.ion()
    plt.figure()
    plt.hold(True)
    model_names = sorted(results.keys())
    colors = dict()
    i = 0
    for model_name in model_names:
        if model_name == "mrf":
            colors[model_name] = 'g'
        else:
            colors[model_name] = cmap(float(i)/(len(model_names)-1))
            i += 1

    for i, model_name in enumerate(model_names):

        recalls = [np.mean(results[model_name]['recalls'][dataset_name]) for dataset_name in dataset_names]
        precisions = [np.mean(results[model_name]['precisions'][dataset_name]) for dataset_name in dataset_names]
        plt.plot(recalls, precisions, '-', color=colors[model_name], label=model_name)
        plt.plot(recalls[-1], precisions[-1], 'ok', color=colors[model_name])

    plt.legend()
    plt.axis([0,1,0,1])

    #plt.gca().set_ylabel('F_0.5 score')
    #plt.gca().set_xticks(range(3))
    #plt.gca().set_xticklabels(('Training (Berlin)', 'Runs (Berlin)', 'Runs (Seattle)'))

    plt.legend(loc='lower right')

    figure_name = filename.replace('.pkl', '.svg')
    print('Saved plot to {}'.format(figure_name))
    plt.savefig(figure_name)

# 4) Random Forest for Pixel Probability Estimation

def experiment_model(datasets, results_path):

    path = os.path.join(results_path, 'experiment_model/')
    if not os.path.isdir(path):
        os.makedirs(path)

    time_string = time.strftime("%Y-%m-%d_%H-%M")
    filename_results = os.path.join(path,time_string+'.pkl')

    params = {'use_features':['color', 'edge', 'miss3D', 'height3D', 'height2D', 'dist2shelf'],
         'segmentation_method':"max_smooth", 'selection_method': "max_smooth",
         'make_convex':True, 'do_shrinking_resegmentation':True, 'do_greedy_resegmentation':True}

    methods = {
            'our_method': ProbabilisticSegmentationBP(**params),
            'rf_.000_lazy': ProbabilisticSegmentationRF({'n_estimators':100, 'min_weight_fraction_leaf':0.0}, **params),
            'rf_.005_lazy': ProbabilisticSegmentationRF({'n_estimators':100, 'min_weight_fraction_leaf':0.005}, **params),
            'rf_.010_lazy': ProbabilisticSegmentationRF({'n_estimators':100, 'min_weight_fraction_leaf':0.01}, **params),
            'rf_.020_lazy': ProbabilisticSegmentationRF({'n_estimators':100, 'min_weight_fraction_leaf':0.02}, **params),
            'rf_.040_lazy': ProbabilisticSegmentationRF({'n_estimators':100, 'min_weight_fraction_leaf':0.04}, **params),
        }

    results = dict()

    for method_name, method in methods.iteritems():

        results[method_name] = train_and_evaluate(method, datasets['berlin_selected'],
            {name: datasets[name] for name in ['berlin_selected','berlin_runs', 'seattle_runs', 'seattle_test']}, withhold_object_info=False)

        with open(filename_results, 'wb') as f:
            pickle.dump(results, f, 2)

    return filename_results

def display_experiment_model(filename):

    import cPickle as pickle
    with open(filename,'rb') as f:
        results = pickle.load(f)

    dataset_names = ['berlin_selected', 'berlin_runs', 'seattle_runs', 'seattle_test']

    print(results.keys())

    model_names = ['rf_.000_lazy', 'rf_.005_lazy', 'rf_.010_lazy', 'rf_.020_lazy', 'rf_.040_lazy', 'our_method']
    colors = {'rf_.000_lazy': 1.1*scheme_dark2[5], 'rf_.005_lazy': 0.95*scheme_dark2[5], 'rf_.010_lazy': 0.8*scheme_dark2[5], 'rf_.020_lazy': 0.65*scheme_dark2[5], 'rf_.040_lazy': 0.5*scheme_dark2[5], 'our_method': scheme_dark2[4]}

    plt.figure()
    plt.hold(True)
    width_all = 0.8
    width_bar_rel = 0.9
    width_bar = width_all/len(model_names)
    x = np.arange(len(dataset_names)) - width_all/2
    for i, model_name in enumerate(model_names):
        recalls = [np.mean(results[model_name]['recalls'][dataset_name]) for dataset_name in dataset_names]
        precisions = [np.mean(results[model_name]['precisions'][dataset_name]) for dataset_name in dataset_names]
        f_scores = [f_score(precision, recall, beta=0.5) for precision, recall in zip(precisions, recalls)]
        plt.bar(x + i*width_bar, f_scores, width=width_bar*width_bar_rel, label=model_name.replace('mrf', 'Our method'), color=colors[model_name], linewidth=0)

    plt.gca().set_ylabel('F0.5 score')
    plt.gca().set_xticks(range(len(dataset_names)))
    plt.gca().set_xticklabels(('Training (Berlin)', 'Test (Berlin)', 'Test (Seattle)', 'APC (Seattle)'))

    plt.legend(loc='lower left')

    figure_name = filename.replace('.pkl', '.svg')
    print('Saved plot to {}'.format(figure_name))
    plt.savefig(figure_name)


global path

if __name__ == "__main__":

    # initialize the global variable display
    utils.global_display()

    data_path = "./data/"

    cache_path = os.path.join(data_path, 'cache')
    results_path = os.path.join(data_path, 'experiment_results')
    dataset_path = os.path.join(data_path, 'rbo_apc')

    dataset_names = ["berlin_runs/"+str(i+1) for i in range(3)] + ["berlin_samples", "berlin_selected"] \
                    + ["seattle_runs/"+str(i+1) for i in range(5)] + ["seattle_test"]

    #datasets = compute_datasets(dataset_names, dataset_path, cache_path) # compute from raw data
    datasets = load_datasets(dataset_names, dataset_path, cache_path) # load from cached data

    datasets['berlin_runs'] = combine_datasets([datasets["berlin_runs/"+str(i+1)] for i in range(3)])
    datasets['seattle_runs'] = combine_datasets([datasets["seattle_runs/"+str(i+1)] for i in range(5)])

    #plt.ion()

    # vizualize the dataset
    #datasets['seattle_test'].visualize_dataset()

    print('\nThis script repeats all experiments from the paper "Probabilistic Object Segmentation for the Amazon Picking Challenge".')
    print('\nIt may take a while ... :D')
    print('\nA. Performance Evaluation')
    print('1) Performance in the Amazon Picking Challenge')

    experiment_APC(datasets, results_path)

    print('\n2) Performance by Object')

    filename_results = experiment_our_method(datasets, results_path)
    display_experiment_objects(filename_results)

    print('\n3) Increasing the Number of Objects per Bin')

    filename_results = experiment_candidates(datasets, results_path)
    display_experiment_candidates(filename_results)

    print('\nB. Comparison to CRF')

    filename_results = experiment_baseline(datasets, results_path)
    display_experiment_baseline(filename_results)

    print('\nC. Variants of the Algorithm')
    print('1) Changing Features')

    filename_result = []
    filename_result.append(experiment_single_features(datasets, results_path))
    filename_result.append(experiment_removed_features(datasets, results_path))
    display_experiment_features(filename_result)

    print('\n2) Pixel Labeling and Selection')

    filename_results = experiment_segmentation(datasets, results_path)
    display_experiment_segmentation(filename_results)

    print('\n3) Re-Labeling and Post-Processing')

    filename_results = experiment_reseg_and_postproc(datasets, results_path)
    display_experiment_reseg_and_postproc(filename_results)

    print('\n4) Random Forest for Pixel Probability Estimation')

    filename_results = experiment_model(datasets, results_path)
    display_experiment_model(filename_results)
    
    print('\nDone.')