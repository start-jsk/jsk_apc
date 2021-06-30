#!/usr/bin/env bash
RBO_DATA_DIR=$(cd $(dirname $0) && pwd)
# download cache
gdown "https://drive.google.com/uc?id=0BzBTxmVQJTrGR3hDenk2LXNWa00" -O $RBO_DATA_DIR/cache.zip
unzip $RBO_DATA_DIR/cache.zip -d $RBO_DATA_DIR 
rm $RBO_DATA_DIR/cache.zip

# download raw_data 
gdown "https://drive.google.com/uc?id=0BzBTxmVQJTrGOEppMVJNeTdsVk0" -O $RBO_DATA_DIR/raw_data.zip
unzip $RBO_DATA_DIR/raw_data.zip -d $RBO_DATA_DIR/raw_data
rm $RBO_DATA_DIR/raw_data.zip
