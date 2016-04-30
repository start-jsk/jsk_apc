#!/usr/bin/env bash

DATA_DIR=$(rospack find jsk_2016_01_baxter_apc)/test_data

gdown "https://drive.google.com/uc?id=0B9P1L--7Wd2vZ2xLZG55OWNYTDQ" -O $DATA_DIR/2016-04-30-16-33-54_apc2016-bin-boxes.bag
