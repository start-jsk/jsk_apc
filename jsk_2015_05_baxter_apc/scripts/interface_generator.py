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

import copy
import json
import random

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
                    "kong_sitting_frog_dog_toy"]

NBINS = len(CONST_BIN_NAMES)
NSINGLE_BINS = 3 + random.randint(0,1)
NDOUBLE_BINS = 5 + random.randint(0,1)
NMULTI_BINS = NBINS - (NSINGLE_BINS + NDOUBLE_BINS) #assumed to be 3 items

# generate the bin contents data structure
#-------------------------------------------------------------------------------
bin_contents = {bin_name:[] for bin_name in CONST_BIN_NAMES}

items = copy.deepcopy(CONST_ITEM_NAMES) # create a destroyable copy of the items
abins = copy.deepcopy(CONST_BIN_NAMES) # create a destroyable copy of the bins

# generate all the single-item bins
for i in range(0,NSINGLE_BINS):
    bin_name = random.choice(abins)
    abins.remove(bin_name)
    item_name = random.choice(items)
    items.remove(item_name)
    bin_contents[bin_name].append(item_name)

# generate all the double-item bins
for i in range(0,NDOUBLE_BINS):
    bin_name = random.choice(abins)
    abins.remove(bin_name)
    for j in range(0,2):
        item_name = random.choice(items)
        items.remove(item_name)
        bin_contents[bin_name].append(item_name)

# generate all the multi-item bins
for i in range(0,NMULTI_BINS):
    bin_name = random.choice(abins)
    abins.remove(bin_name)
    for j in range(0,3):
        item_name = random.choice(items)
        items.remove(item_name)
        bin_contents[bin_name].append(item_name)

# generate the work order data structure
#-------------------------------------------------------------------------------
work_order = [{'bin':bin_name,'item':item_name} for bin_name in CONST_BIN_NAMES
              for item_name in (bin_contents[bin_name][0:1])]

# write the dictionary to the appropriately names json file
#-------------------------------------------------------------------------------
data = {'bin_contents': bin_contents, 'work_order': work_order}
with open('apc.json', 'w') as outfile:
    json.dump(data, outfile, sort_keys=True, indent=4, separators=(',',': '))
