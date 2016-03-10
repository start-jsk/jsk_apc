#!/usr/bin/env python

#/*********************************************************************
# *
# * Software License Agreement (BSD License)
# *
# *  Copyright (c) 2015, Kiva Systems
# *  All rights reserved.
# *
# *  Redistribution and use in source and binary forms, with or without
# *  modification, are permitted provided that the following conditions
# *  are met:
# *
# *   * Redistributions of source code must retain the above copyright
# *     notice, this list of conditions and the following disclaimer.
# *   * Redistributions in binary form must reproduce the above
# *     copyright notice, this list of conditions and the following
# *     disclaimer in the documentation and/or other materials provided
# *     with the distribution.
# *   * Neither the name of the authors nor the names of the
# *     contributors may be used to endorse or promote products derived
# *     from this software without specific prior written permission.
# *
# *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# *  POSSIBILITY OF SUCH DAMAGE.
# *
# *********************************************************************/


# This is demo program of stow task json file generator
'''
Usage : python interface_generator_stow.py
ex) python interface_generator_pick.py 2 => generates pick_layout_2.json at ../json directory
'''

import copy
import json
import random
import os
import argparse
import jsk_apc2016_common
#-------------------------------------------------------------------------------

def select_item():
    while True:
        item_index = random.randint(0,len(items_bins) - 1)
        if (items_stock[item_index] > 0):
            item_name = items_bins[item_index]
            items_stock[item_index] -= 1
            break
    return item_name

# define our bin and item names to use
CONST_BIN_NAMES = ['bin_A',
                   'bin_B',
                   'bin_C',
                   'bin_D',
                   'bin_E',
                   'bin_F',
                   'bin_G',
                   'bin_H',
                   'bin_I',
                   'bin_J',
                   'bin_K',
                   'bin_L']


NBINS = len(CONST_BIN_NAMES)
N1_2_BINS = 3 + random.randint(0,1)
N3_4_BINS = 5 + random.randint(0,1)
N5__BINS = NBINS - (N1_2_BINS + N3_4_BINS) #assumed to be more than 5 items

ITEMS_DATA = jsk_apc2016_common.get_object_data()
CONST_ITEM_NAMES = []
CONST_N_ITEMS=[]
for item_data in ITEMS_DATA :
    CONST_ITEM_NAMES.append(item_data['name'])
    CONST_N_ITEMS.append(item_data['stock'])

NTOTE_TOTAL = 12 # total number of items
N_TOTAL_ITEMS = 35 ###CHANGE THIS TO 40 AFTER FULL ITEMS BE DELIEVERED FROM AMAZON

count_items = N_TOTAL_ITEMS
parser = argparse.ArgumentParser()
parser.add_argument("version")
args = parser.parse_args()
version = args.version

# generate the bin contents data structure
#-------------------------------------------------------------------------------
bin_contents = {bin_name:[] for bin_name in CONST_BIN_NAMES}
tote_contents = []
bin_list = [1] * len(CONST_BIN_NAMES)
bin_check = [False] * len(CONST_BIN_NAMES)
items_bins = copy.deepcopy(CONST_ITEM_NAMES) # create a destroyable copy of the items
items_stock = copy.deepcopy(CONST_N_ITEMS)
abins = copy.deepcopy(CONST_BIN_NAMES) # create a destroyable copy of the bins

# generate tote items. Some items are chosen multiple times
for i in range(0,NTOTE_TOTAL):
    item_name = select_item()
    tote_contents.append(item_name)

for i in range(N5__BINS):
    while True:
        index = random.randint(0, NBINS - 1)
        if bin_check[index] == False:
            break
    n_item = random.randint(5,10)
    bin_list[index] = n_item
    count_items -= n_item
    bin_check[index] = True

for i in range(N3_4_BINS):
    while True:
        index = random.randint(0, NBINS - 1)
        if bin_check[index] == False:
            break
    n_item = random.randint(3,4)
    bin_list[index] = n_item
    count_items -= n_item
    bin_check[index] = True

while count_items > 6:
    while True:
        index = random.randint(0, NBINS - 1)
        if bin_check[index] == True:
            break
    bin_list[index] += 1
    count_items -= 1

while count_items < N1_2_BINS:
    while True:
        index = random.randint(0, NBINS - 1)
        if (bin_check[index] == True) and (bin_list[index] > 1):
            break
    bin_list[index] -= 1
    count_items += 1

n_1 = - count_items + 2 * N1_2_BINS
n_2 =  count_items - N1_2_BINS

for i in range(n_2):    
    while True:
        index = random.randint(0, NBINS - 1)
        if bin_check[index] == False:
            break
    n_item = 2
    bin_list[index] = n_item

if not sum(bin_list) == N_TOTAL_ITEMS:
    print "warning : number of items unmatched. Recommended to try again"
    print "number of total items should be: {}".format(N_TOTAL_ITEMS)
    print "sum of items in bin: {}".format(sum(bin_list))

# generate all item bins
# make two bin with multiple copy of items

for i in range(0,NBINS):
    bin_name = random.choice(abins)
    abins.remove(bin_name)
    for j in range(0,bin_list[i]):
        item_name = select_item()
        bin_contents[bin_name].append(item_name)

no_items = [x for x in items_stock if x < 0]
if len(no_items) > 0:
    print "warning: not enough items expected for this json. Please try again"

# write the dictionary to the appropriately names json file
#-------------------------------------------------------------------------------
data = {'bin_contents': bin_contents, 'work_order': tote_contents}
this_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(this_dir,'../json')
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
os.chdir(output_dir)
with open('stow_layout_'+version+'.json', 'w') as outfile:
    json.dump(data, outfile, sort_keys=True, indent=4, separators=(',',': '))
print('stow_layout_'+version+'.json generated at ../json')
os.chdir(this_dir)
