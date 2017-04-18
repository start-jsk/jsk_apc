#!/usr/bin/env python3
# -*- coding:utf-8 -*-

'''
This program generates visible color histogram set for dataset.
Usage : python histogram_plot.py NAME_OF_DATASET
ex) python histogram_plot.py masked_data
'''

import argparse
import os
import re
import numpy as np
import cv2
import matplotlib.pyplot as plt
from common import get_data_dir
from color_histogram_features import ColorHistogramFeatures

def plot():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset')
    args = parser.parse_args()
    
    dataset = args.dataset
    data_dir = get_data_dir()
    container_path = os.path.join(data_dir,dataset)
    output_dir = os.path.abspath(container_path + '_histograms')
    if not os.path.exists(output_dir):
        print('creating histograms directory: {}'.format(output_dir))
        os.mkdir(output_dir)
    categs = os.listdir(container_path)
    os.chdir(container_path)
    for categ in categs:
        os.chdir(categ)
        print('processing category: {}'.format(categ))
        img_files = os.listdir('.')
        print('found {} images'.format(len(img_files)))
        categ_output_dir = os.path.join(output_dir,categ)
    
        if not os.path.exists(categ_output_dir):
            os.mkdir(categ_output_dir)
        
        c = ColorHistogramFeatures()
        for img_file in img_files:
            img = cv2.imread(img_file)
            hist = c.color_hist(img)
            x=np.arange(len(hist))
            plt.bar(x,hist)
            plt.xlabel('hue')
            plt.axis([0,64,0,300])
            plt.title(img_file)
            plt.savefig(os.path.join(output_dir, categ, img_file))
            plt.close()
        os.chdir('..')

    os.chdir('..')

def main():
    plot()

if __name__ == "__main__":
    main()
