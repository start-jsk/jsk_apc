#!/usr/bin/env bash

mkdir -p $(rospack find jsk_apc2016_common)/data/tokyo_run
gdown https://drive.google.com/uc?id=0BzBTxmVQJTrGR3BHVkRPck9wTEE -O $(rospack find jsk_apc2016_common)/data/tokyo_run/single_item_labeled.zip
unzip $(rospack find jsk_apc2016_common)/data/tokyo_run/single_item_labeled.zip -d $(rospack find jsk_apc2016_common)/data/tokyo_run/single_item_labeled
rm $(rospack find jsk_apc2016_common)/data/tokyo_run/single_item_labeled.zip

rosrun jsk_apc2016_common train_segmenter_2016.py
