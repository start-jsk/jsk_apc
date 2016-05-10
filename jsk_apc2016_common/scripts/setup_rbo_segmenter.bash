#!/usr/bin/env bash

PKG_DIR=$(rospack find jsk_apc2016_common)
RBO_DATA_DIR=$PKG_DIR/python/jsk_apc2016_common/rbo_segmentation/data
if [ ! -e $RBO_DATA_DIR/cache ] || [ ! -e $RBO_DATA_DIR/raw_data ]; then
  bash $RBO_DATA_DIR/download_data.sh
fi
python $PKG_DIR/scripts/train_segmenter.py
