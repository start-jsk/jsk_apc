#!/usr/bin/env python
import sys
import os
import rospkg
rospack = rospkg.RosPack()
pack_path = rospack.get_path('jsk_2016_01_baxter_apc')
sys.path.insert(0, pack_path)

from node_scripts.segmentation.utils import Utils
import node_scripts.segmentation.utils as utils
from node_scripts.segmentation.apc_data import APCSample, APCDataSet
from node_scripts.segmentation.probabilistic_segmentation import ProbabilisticSegmentationBP
import node_scripts.segmentation.apc_data as apc_data

import pickle

params = {'use_features':['color', 'dist2shelf', 'height3D'], # 'height2D''edge'
        'segmentation_method':"max_smooth", 'selection_method': "max_smooth",
        'make_convex':True, 'do_shrinking_resegmentation':True, 'do_greedy_resegmentation':True}

# TODO:following two methods should be included somewhere else
# previously declared in main.py
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

def train(pkl_path):
    sys.modules['apc_data'] = apc_data

    data_path =  pack_path + "/node_scripts/segmentation/data"
    cache_path = os.path.join(data_path, 'cache')
    results_path = os.path.join(data_path, 'experiment_results')
    dataset_path = os.path.join(data_path, 'rbo_apc')

    dataset_names = ["berlin_runs/"+str(i+1) for i in range(3)] + ["berlin_samples", "berlin_selected"] \
                        + ["seattle_runs/"+str(i+1) for i in range(5)] + ["seattle_test"]
    datasets = load_datasets(dataset_names, dataset_path, cache_path) # load from cached data

    datasets['berlin_runs'] = combine_datasets([datasets["berlin_runs/"+str(i+1)] for i in range(3)])
    datasets['seattle_runs'] = combine_datasets([datasets["seattle_runs/"+str(i+1)] for i in range(5)])
    datasets['training_berlin_and_seattle'] = combine_datasets([datasets['berlin_selected'], datasets['berlin_runs']])
    bp = ProbabilisticSegmentationBP(**params)

    train_set = datasets['berlin_selected']
    bp.fit(train_set)

    with open(pkl_path, 'wb') as f:
        pickle.dump(bp, f)
    print "saved"


if __name__ == '__main__':
    train(pack_path + '/data/trained_segmenter.pkl')
