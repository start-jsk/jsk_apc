from apc_data import APCDataSet, APCSample
from probabilistic_segmentation import ProbabilisticSegmentationRF, ProbabilisticSegmentationBP
import pickle
import os

import matplotlib.pyplot as plt
import numpy as np
import copy
import rospkg


def _fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    hist = np.bincount(n * a[k].astype(int) +
                       b[k], minlength=n**2).reshape(n, n)
    return hist

def label_accuracy_score(label_true, label_pred, n_class):
    """Returns accuracy score evaluation result.
      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    hist = _fast_hist(label_true.flatten(), label_pred.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum().astype(np.float64)
    acc_cls = np.diag(hist) / hist.sum(axis=1).astype(np.float64)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)).astype(np.float64)
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum().astype(np.float64)
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc


# previously declared in main.py
def combine_datasets(datasets):
    samples = []
    for d in datasets:
        samples += d.samples
    return APCDataSet(samples=samples)


def load_datasets(dataset_names, data_path, cache_path):
    datasets = dict()

    for dataset_name in dataset_names:
        dataset_path = os.path.join(
                data_path, 'rbo_apc/{}'.format(dataset_name))
        datasets[dataset_name] = APCDataSet(
                name=dataset_name, dataset_path=dataset_path,
                cache_path=cache_path, load_from_cache=True)
    return datasets




def evaluate(bp, test_data):
    acc_list = []
    acc_cls_list = []
    mean_iu_list = []
    fwavacc_list = []
    for sample in test_data.samples:
        if len(sample.object_masks) == 0:
            continue
        pred_target = sample.object_masks.keys()[0]
        if pred_target == 'shelf':
            if len(sample.object_masks.keys()) == 1:
                continue
            pred_target = sample.object_masks.keys()[1]
        bp.predict(sample, pred_target)
        print('done')

        images = []
        images.append(bp.posterior_images_smooth['shelf'])
        objects = []
        objects.append('shelf')
        for _object in  bp.posterior_images_smooth.keys():
            if _object != 'shelf':
                images.append(bp.posterior_images_smooth[_object])
                objects.append(_object)
        pred = np.argmax(np.array(images), axis=0)



        # remove dataset that does not have complete set
        objects_copy = copy.copy(objects)
        object_masks_keys = sample.object_masks.keys()
        if 'shelf' in objects_copy: objects_copy.remove('shelf')
        if 'shelf' in object_masks_keys: object_masks_keys.remove('shelf')
        if set(objects_copy) != set(object_masks_keys):
            #print('skip posterior_image keys ', objects_copy)
            #print('skip object_mask keys ', object_masks_keys)
            continue
        true = np.zeros_like(pred)


        for i, _object in enumerate(objects):
            if _object != 'shelf':
                true[sample.object_masks[_object]] = i

        masked_pred = pred[sample.bin_mask] 
        masked_true = true[sample.bin_mask]


        acc, acc_cls, mean_iu, fwavacc = label_accuracy_score(masked_true, masked_pred, len(objects))
        acc_list.append(acc)
        acc_cls_list.append(acc_cls)
        mean_iu_list.append(mean_iu)
        fwavacc_list.append(fwavacc)
        
        """
        label_pred = np.zeros(pred.shape[1:]).astype(np.int64)
        label_true = np.zeros(pred.shape[1:]).astype(np.int64)

        for i in range(pred.shape[0]):
            label_pred[pred[i]] = i
            label_true[true[i]] = i
        label_pred_masked = label_pred[sample.bin_mask]
        label_true_masked = label_true[sample.bin_mask]
        """
    return acc_list, acc_cls_list, mean_iu_list, fwavacc_list




def create_dataset(dataset_path):
    # initialize empty dataset
    dataset = APCDataSet(from_pkl=False)

    data_file_prefixes = []
    key = '.jpg'
    for dir_name, sub_dirs, files in os.walk(dataset_path):
        for f in files:
            if key == f[-len(key):]:
                data_file_prefixes.append(
                    os.path.join(dir_name, f[:-len(key)]))

    print(data_file_prefixes)
    for file_prefix in data_file_prefixes:
        dataset.samples.append(
            APCSample(data_2016_prefix=file_prefix,
                        labeled=True, is_2016=True, infer_shelf_mask=True))
    return dataset
        



###############################################################################
#                               prepare dataset                               #
###############################################################################
#data_path = '/home/leus/ros/indigo/src/start-jsk/jsk_apc/jsk_apc2016_common/data'
#cache_path = os.path.join(data_path, 'cache')
#dataset_path = os.path.join(data_path, 'rbo_apc')
rospack = rospkg.RosPack()

common_path = rospack.get_path('jsk_apc2016_common')
data_path = common_path + '/data/'
dataset_name = 'tokyo_run/single_item_labeled'
dataset_path = os.path.join(data_path, dataset_name)


data = create_dataset(dataset_path)


###############################################################################
#                                   dataset                                   #
###############################################################################
train_data, test_data = data.split_simple(portion_training=0.7)


###############################################################################
#                                all features                                 #
###############################################################################
all_features = ['color', 'height3D', 'dist2shelf']
params = {
        'use_features': all_features,
        'segmentation_method': "max_smooth", 'selection_method': "max_smooth",
        'make_convex': True, 'do_shrinking_resegmentation': True,
        'do_greedy_resegmentation': True}
bp = ProbabilisticSegmentationBP(**params)


bp.fit(train_data)

acc_list, acc_cls_list, mean_iu_list, fwavacc_list = evaluate(bp, test_data)
        
print('all features acc ', np.mean(acc_list))
print('all features acc_cls ', np.mean(acc_cls_list))
print('all features mean_iu ', np.mean(mean_iu_list))
print('all features fwavcc ', np.mean(fwavacc_list))


###############################################################################
#                                # Color only                                 #
###############################################################################
params = {
        'use_features': ['color'],
        'segmentation_method': "max_smooth", 'selection_method': "max_smooth",
        'make_convex': True, 'do_shrinking_resegmentation': True,
        'do_greedy_resegmentation': True}
bp = ProbabilisticSegmentationBP(**params)

bp.fit(train_data)
acc_list, acc_cls_list, mean_iu_list, fwavacc_list = evaluate(bp, test_data)
        
print('trained only by color features acc ', np.mean(acc_list))
print('trained only by color features acc_cls ', np.mean(acc_cls_list))
print('trained only by color features mean_iu ', np.mean(mean_iu_list))
print('trained only by color features fwavcc ', np.mean(fwavacc_list))

