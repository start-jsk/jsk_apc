#!/usr/bin/env python

import copy
import json
import random
import os
import jsk_apc2016_common

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

N_TOTAL_ITEMS = 47

ITEMS_DATA = jsk_apc2016_common.get_object_data()
CONST_ITEM_NAMES = []
CONST_N_ITEMS=[]
for item_data in ITEMS_DATA :
    CONST_ITEM_NAMES.append(item_data['name'])
    CONST_N_ITEMS.append(item_data['stock'])

class InterfaceGeneratorPick():
    def __init__ (self):
        self.total_item_num = N_TOTAL_ITEMS
        self.bin_names = copy.deepcopy(CONST_BIN_NAMES)
        self.bin_contents = {bin_name:[] for bin_name in self.bin_names}
        self.items = copy.deepcopy(CONST_ITEM_NAMES)
        self.items_stock = copy.deepcopy(CONST_N_ITEMS)

    def gen_random_item_list(self):
        self.item_list = []
        for item, stock in zip(self.items, self.items_stock):
            for i in range(0, stock):
                self.item_list.append(item)
        random.shuffle(self.item_list)
        self.item_list[:self.total_item_num]

    def select_bin(self):
        while True:
            item_list_bin =[]
            for i in range(0, self.total_item_num):
                item_list_bin.append(random.choice(self.bin_names))
            if set(self.bin_names) <= set(item_list_bin):
                break
        self.item_list_bin = item_list_bin

    def gen_bin_contents(self):
        for bin_name, item in zip(self.item_list_bin, self.item_list):
            self.bin_contents[bin_name].append(item)

    def run(self):
        self.gen_random_item_list()
        self.select_bin()
        self.gen_bin_contents()
        self.gen_workorder()
    
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
        i = 1
        while True:
            filename  = 'pick_layout_'+str(i)+'.json' 
            if os.path.isfile(filename):
                i += 1
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
