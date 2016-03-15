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
'''
Usage : python interface_generator_pick.py {version_of_pick}
ex) python interface_generator_pick.py 2 => generates pick_layout_2.json at ../json directory
'''
import copy
import json
import random
import os
import jsk_apc2016_common
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

NBINS = len(CONST_BIN_NAMES)
N1_2_BINS = 3 + random.randint(0,1)
N3_4_BINS = 5 + random.randint(0,1)
N5__BINS = NBINS - (N1_2_BINS + N3_4_BINS) #assumed to be more than 5 items
N_TOTAL_ITEMS = 47  ###CHANGE THIS TO 50 AFTER FULL ITEMS BE DELIEVERED FROM AMAZON

ITEMS_DATA = jsk_apc2016_common.get_object_data()
CONST_ITEM_NAMES = []
CONST_N_ITEMS=[]
for item_data in ITEMS_DATA :
    CONST_ITEM_NAMES.append(item_data['name'])
    CONST_N_ITEMS.append(item_data['stock'])

class InterfaceGeneratorPick():
    def __init__ (self):
        self.count_items = N_TOTAL_ITEMS
        self.bin_contents = {bin_name:[] for bin_name in CONST_BIN_NAMES}
        self.bin_list = [1] * NBINS
        self.bin_check = [False] * NBINS
        self.items_bins = copy.deepcopy(CONST_ITEM_NAMES) # create a destroyable copy of the items
        self.items_stock = copy.deepcopy(CONST_N_ITEMS)
        self.abins = copy.deepcopy(CONST_BIN_NAMES) # create a destroyable copy of the bins

    def run(self):
        self.num_items_bin()
        self.select_item_bin()
        self.gen_workorder()


    # number of items stored at each bin
    def num_items_bin(self):
        for i in range(N5__BINS):
            while True:
                index = random.randint(0, NBINS - 1)
                if self.bin_check[index] == False:
                    break
            n_item = random.randint(5,10)
            self.bin_list[index] = n_item
            self.count_items -= n_item
            self.bin_check[index] = True

        for i in range(N3_4_BINS):
            while True:
                index = random.randint(0, NBINS - 1)
                if self.bin_check[index] == False:
                    break
            n_item = random.randint(3,4)
            self.bin_list[index] = n_item
            self.count_items -= n_item
            self.bin_check[index] = True

        while self.count_items > 6:
            while True:
                index = random.randint(0, NBINS - 1)
                if self.bin_check[index] == True:
                    break
            self.bin_list[index] += 1
            self.count_items -= 1

        while self.count_items < N1_2_BINS:
            while True:
                index = random.randint(0, NBINS - 1)
                if (self.bin_check[index] == True) and (self.bin_list[index] > 1):
                    break
            self.bin_list[index] -= 1
            self.count_items += 1

        n_1 = - self.count_items + 2 * N1_2_BINS
        n_2 =  self.count_items - N1_2_BINS

        for i in range(n_2):
            while True:
                index = random.randint(0, NBINS - 1)
                if self.bin_check[index] == False:
                    break
            n_item = 2
            self.bin_list[index] = n_item

        if not sum(self.bin_list) == N_TOTAL_ITEMS:
            print "warning : number of items unmatched. Recommended to try again"
            print "number of total items should be: {}".format(N_TOTAL_ITEMS)
            print "sum of items in bin: {}".format(sum(self.bin_list))

    # generate all item bins
    def select_item_bin(self):
        abins_ = copy.deepcopy(self.abins)
        bin_contents_ = copy.deepcopy(self.bin_contents)
        for i in range(0,NBINS):
            bin_name = random.choice(abins_)
            abins_.remove(bin_name)
            item_list = []
            items_stock_ = copy.deepcopy(self.items_stock)
            for j in range(0,self.bin_list[i]):
                while True:
                    item_index = random.randint(0,len(self.items_bins) - 1)
                    if (items_stock_[item_index] > 0):
                        item_name = self.items_bins[item_index]
                        items_stock_[item_index] -= 1
                        break
                item_list.append(item_name)
            bin_contents_[bin_name] = item_list
        self.abins = abins_
        self.items_stock = items_stock_
        self.bin_contents = bin_contents_
        no_items = [x for x in self.items_stock if x < 0]
        if len(no_items) > 0:
            print "warning: not enough items expected for this json. Please try again"

    # generate the work order data structure
    def gen_workorder(self):
        self.work_order = [{'bin':bin_name,'item':item_name} for bin_name in CONST_BIN_NAMES
                    for item_name in (self.bin_contents[bin_name][0:1])]

    # write the dictionary to the appropriately names json file
    def write_json(self):
        data = {'bin_contents': self.bin_contents, 'work_order': self.work_order}
        this_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(this_dir,'../json')
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        os.chdir(output_dir)
        i=1
        while True:
            filename  = 'pick_layout_'+str(i)+'.json' 
            if os.path.isfile(filename):
                i +=1
            else:
                with open(filename, 'w') as outfile:
                    json.dump(data, outfile, sort_keys=True, indent=4, separators=(',',': '))
                print(filename+ ' generated at ../json')
                os.chdir(this_dir)
                break

if __name__ == "__main__":
    interface_pick = InterfaceGeneratorPick()
    interface_pick.run()
    interface_pick.write_json()
