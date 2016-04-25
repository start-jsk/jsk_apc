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


NTOTE_TOTAL = 12
N_TOTAL_ITEMS = 35

ITEMS_DATA = jsk_apc2016_common.get_object_data()
CONST_ITEM_NAMES = []
CONST_N_ITEMS=[]
for item_data in ITEMS_DATA :
    CONST_ITEM_NAMES.append(item_data['name'])
    CONST_N_ITEMS.append(item_data['stock'])

class InterfaceGeneratorStow():
    def __init__(self):
        self.total_item_num = N_TOTAL_ITEMS
        self.bin_names = copy.deepcopy(CONST_BIN_NAMES)
        self.bin_contents = {bin_name:[] for bin_name in CONST_BIN_NAMES}
        self.items = copy.deepcopy(CONST_ITEM_NAMES) # create a destroyable copy of the items
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

    def select_tote(self):
        self.tote_contents = []
        for i in range(0, NTOTE_TOTAL):
            item_name = self.select_tote_item()
            self.tote_contents.append(item_name)

    def select_tote_item(self):
        while True:
            item_index = random.randint(0, len(self.items) - 1)
            if (self.items_stock[item_index] > 0):
                item_name = self.items[item_index]
                self.items_stock[item_index] -= 1
                break
        return item_name

    def write_json(self):
        data = {'bin_contents': self.bin_contents, 'work_order': self.tote_contents}
        this_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(this_dir,'../json')
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        os.chdir(output_dir)
        i=1
        while True:
            filename  = 'stow_layout_'+str(i)+'.json' 
            if os.path.isfile(filename):
                i +=1
            else:
                with open(filename, 'w') as outfile:
                    json.dump(data, outfile, sort_keys=True, indent=4, separators=(',',': '))
                print(filename+ ' generated at ../json')
                os.chdir(this_dir)
                break

    def run(self):
        self.select_tote()
        self.gen_random_item_list()
        self.select_bin()
        self.gen_bin_contents()

if __name__ == "__main__":
    interface_stow = InterfaceGeneratorStow()
    interface_stow.run()
    interface_stow.write_json()
