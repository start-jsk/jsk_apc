#!/bin/sh

jsk_data get 2015-10-18-14-07-50_berkeley_dataset_mask_applied.tgz
mv 2015-10-18-14-07-50_berkeley_dataset_mask_applied.tgz $(rospack find jsk_apc2015_common)/dataset/berkeley_dataset_mask_applied.tgz

jsk_data get 2015-10-18-14-03-44_berkeley_dataset_bof_hist.pkl.gz
mv 2015-10-18-14-03-44_berkeley_dataset_bof_hist.pkl.gz $(rospack find jsk_apc2015_common)/dataset/berkeley_dataset_bof_hist.pkl.gz

jsk_data get 2015-10-18-14-05-19_berkeley_dataset_sift_feature.pkl.gz
mv 2015-10-18-14-05-19_berkeley_dataset_sift_feature.pkl.gz $(rospack find jsk_apc2015_common)/dataset/berkeley_dataset_sift_feature.pkl.gz

# Take too much time
# python $(rospack find jsk_apc2015_common)/scripts/download_berkeley_dataset.py -O $(rospack find jsk_apc2015_common)/dataset/berkeley_dataset
