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

import sys
import argparse
import json

# define a terminal-output color class
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def interface_test(json_file):
    # open the json file
    json_data = open(json_file)
    data = json.load(json_data)

    #--------------------------------------------------------------------------

    # define some known constants to compare against
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
    CONST_NBINS = 12
    CONST_NPICKABLE_ITEMS = CONST_NBINS

    # do some validitiy testing on the input file
    #--------------------------------------------------------------------------

    print('')
    print('Checking that the bin contents meet our contest definition ...')
    bin_counts = [0,0,0,0,0,0,0]
    total_items = 0
    for bin_name in data['bin_contents']:
        n_items = len(data['bin_contents'][bin_name])
        total_items = total_items + n_items
        if n_items < len(bin_counts):
            bin_counts[n_items] = bin_counts[n_items] + 1
        else:
            error_msg = 'ERROR: Too many items in: %s' % bin_name
            raise ValueError(error_msg)
            break
        if not bin_name in CONST_BIN_NAMES:
            error_msg = 'ERROR: Unknown bin name: %s' % bin_name
            raise ValueError(error_msg)
            break
        for item in data['bin_contents'][bin_name]:
            if not item in CONST_ITEM_NAMES:
                error_msg = 'ERROR: Unknown item name: %s' % item
                raise ValueError(error_msg)

    if bin_counts[0] > 0:
        error_msg = 'ERROR: Found %d empty bins' % bin_counts[0]
    if bin_counts[1] < 2:
        error_msg = 'ERROR: Only found %d singles' % bin_counts[1]
    elif bin_counts[2] < 2:
        error_msg = 'ERROR: Only found %d doubles' % bin_counts[2]
    elif sum(bin_counts[3:len(bin_counts)]) < 2:
        error_msg = 'ERROR: Only found %d multis' \
            % sum(bin_counts[3:len(bin_counts)])
    else:
        print (bcolors.OKGREEN +
               '  SUCCESS! Found %d singles, %d doubles, %d multis' +
               bcolors.ENDC) % (bin_counts[1], bin_counts[2],
                                sum(bin_counts[3:len(bin_counts)]))

    #--------------------------------------------------------------------------

    print('')
    print('Checking if the work order is valid ...')
    n_found_items = 0
    for line_item in data['work_order']:
        item = line_item['item']
        kbin = line_item['bin']

        if item in data['bin_contents'][kbin]:
            n_found_items = n_found_items + 1
            #print 'Found item: %s inside of: %s' % (item, kbin)
        else:
            error_msg = 'ERROR: Item: %s does not exist in: %s' % (item, kbin)
            raise ValueError(error_msg)
            break
    if n_found_items == CONST_NPICKABLE_ITEMS:
        print (bcolors.OKGREEN + '  SUCCESS! Found %d items (1 per bin)' +
               bcolors.ENDC) % CONST_NPICKABLE_ITEMS
    else:
        error_msg = 'ERROR: Only found %d/%d items.' \
            % (n_found_items, CONST_NPICKABLE_ITEMS)
        raise ValueError(error_msg)

    #--------------------------------------------------------------------------

    # close the json file
    json_data.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('json')
    args = parser.parse_args(sys.argv[1:])
    interface_test(args.json)
