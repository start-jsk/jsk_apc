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


# This is demo program of pick task json file generator
# extra_item_(number) on CONST_ITEM_NAMES must be changed when exact item list is announced by Amazon

import copy
import json
import random
import os
#-------------------------------------------------------------------------------

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

CONST_ITEM_NAMES = ["oreo_mega_stuf",
                    "champion_copper_plus_spark_plug",
                    "expo_dry_erase_board_eraser",
                    "kong_duck_dog_toy",
                    "genuine_joe_plastic_stir_sticks",
                    "munchkin_white_hot_duck_bath_toy",
                    "crayola_64_ct",
                    "mommys_helper_outlet_plugs",
                    "sharpie_accent_tank_style_highlighters",
                    "kong_air_dog_squeakair_tennis_ball",
                    "stanley_66_052",
                    "safety_works_safety_glasses",
                    "dr_browns_bottle_brush",
                    "laugh_out_loud_joke_book",
                    "cheezit_big_original",
                    "paper_mate_12_count_mirado_black_warrior",
                    "feline_greenies_dental_treats",
                    "elmers_washable_no_run_school_glue",
                    "mead_index_cards",
                    "rolodex_jumbo_pencil_cup",
                    "first_years_take_and_toss_straw_cup",
                    "highland_6539_self_stick_notes",
                    "mark_twain_huckleberry_finn",
                    "kyjen_squeakin_eggs_plush_puppies",
                    "kong_sitting_frog_dog_toy",
                    "extra_item_1",
                    "extra_item_2",
                    "extra_item_3",
                    "extra_item_4",
                    "extra_item_5",
                    "extra_item_6",
                    "extra_item_7",
                    "extra_item_8",
                    "extra_item_9",
                    "extra_item_10",
                    "extra_item_11",
                    "extra_item_12",
                    "extra_item_13",
                    "extra_item_14",
                    "extra_item_15"]

NBINS = len(CONST_BIN_NAMES)
N_TOTAL_ITEMS = 50


# generate the bin contents data structure
#-------------------------------------------------------------------------------
bin_contents = {bin_name:[] for bin_name in CONST_BIN_NAMES}
bin_list = [1] * len(CONST_BIN_NAMES)
items_bins = copy.deepcopy(CONST_ITEM_NAMES) # create a destroyable copy of the items
items_tote = copy.deepcopy(CONST_ITEM_NAMES)
abins = copy.deepcopy(CONST_BIN_NAMES) # create a destroyable copy of the bins

# number of items stored at each bin

for i in range(0,N_TOTAL_ITEMS - 1 * len(CONST_BIN_NAMES)):
    index = random.randint(0,len(bin_list) - 1)
    bin_list[index] += 1

if not sum(bin_list) == N_TOTAL_ITEMS:
    print "warning : number of items unmatched"

while max(bin_list) > 10:
    index = bin_list.index(max(bin_list))
    bin_list[index] -= 1
    index = random.randint(0,len(bin_list) - 1)
    bin_list[index] += 1

# making one bin with more than 8 items
index = random.randint(0,len(bin_list) - 1)
prev_num = bin_list[index]
num_item = random.randint(8,10)
i = num_item - prev_num
while i > 0 :
    this_index = random.randint(0,len(bin_list) - 1)
    if not this_index == index :
        if bin_list[this_index] > 2 :
            i -= 1
            bin_list[this_index] -= 1
            bin_list[index] += 1



# generate all item bins
# make two bin with multiple copy of items

for i in range(0,len(bin_list)):
    bin_name = random.choice(abins)
    abins.remove(bin_name)
    if i == 0 :
        for j in range(0,bin_list[i]-1):
            item_name = random.choice(items_bins)
            bin_contents[bin_name].append(item_name)        

        if bin_list[i] == 1 :
            item_name = random.choice(items_bins)
            bin_contents[bin_name].append(item_name)
            bin_contents[bin_name].append(item_name)
        else :
            item_name = bin_contents[bin_name][-1]
            bin_contents[bin_name].append(item_name)
      
    elif i == 1 :
        for j in range(0,bin_list[i]-1):
            item_name = random.choice(items_bins)
            bin_contents[bin_name].append(item_name)        

        if bin_list[i] == 1 :
            item_name = random.choice(items_bins)
            bin_contents[bin_name].append(item_name)
            bin_contents[bin_name].append(item_name)
        else :
            item_name = bin_contents[bin_name][-1]
            bin_contents[bin_name].insert(0,item_name)
      
    else :
        for j in range(0,bin_list[i]):
            item_name = random.choice(items_bins)
            bin_contents[bin_name].append(item_name)


# generate the work order data structure
#-------------------------------------------------------------------------------
work_order = [{'bin':bin_name,'item':item_name} for bin_name in CONST_BIN_NAMES
              for item_name in (bin_contents[bin_name][0:1])]

# write the dictionary to the appropriately names json file
#-------------------------------------------------------------------------------
data = {'bin_contents': bin_contents, 'work_order': work_order}
this_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(this_dir,'../json')
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
os.chdir(output_dir)
with open('apc_pick.json', 'w') as outfile:
    json.dump(data, outfile, sort_keys=True, indent=4, separators=(',',': '))
print('apc_pick.json generated at ../json')
os.chdir(this_dir)
