#!/usr/bin/env bash

if [ $# -ne 2 ]; then
  echo "Please specify two arguments: objects_dir and config_dir"
  exit 1
fi

set -x

objects_dir=$1
config_dir=$2

ln -sf $objects_dir/* $(rospack find jsk_arc2017_common)/data/objects/
ln -sf $config_dir/label_names.yaml $(rospack find jsk_arc2017_common)/config/
ln -sf $config_dir/object_graspability.yaml $(rospack find jsk_arc2017_common)/config/
ln -sf $config_dir/object_weights.yaml $(rospack find jsk_arc2017_common)/config/

set +x
