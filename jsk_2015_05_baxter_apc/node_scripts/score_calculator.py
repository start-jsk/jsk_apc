#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
from __future__ import print_function
import sys
import argparse

from termcolor import cprint, colored

from bin_contents import get_bin_contents
from work_order import get_work_order

import jsk_apc2015_common


parser = argparse.ArgumentParser('score_calculator')
parser.add_argument('json', help='contest interface json')
args = parser.parse_args(sys.argv[1:])

OBJECTS = jsk_apc2015_common.data.object_list()
BONUS = {
        'munchkin_white_hot_duck_bath_toy': 1,
        'stanley_66_052': 3,
        'safety_works_safety_glasses': 1,
        'rolodex_jumbo_pencil_cup': 2,
        'first_years_take_and_toss_straw_cup': 2,
        'mark_twain_huckleberry_finn': 3,
        'kyjen_squeakin_eggs_plush_puppies': 1,
        'kong_sitting_frog_dog_toy': 1,
        'kong_air_dog_squeakair_tennis_ball': 1,
        'dr_browns_bottle_brush': 2,
        'kong_duck_dog_toy': 1,
        'laugh_out_loud_joke_book': 3,
        }
SCORING = {
        'MOVE_FROM_MULTI_ITEM_BIN': 20,
        'MOVE_FROM_DOUBLE_ITEM_BIN': 15,
        'MOVE_FROM_SINGLE_ITEM_BIN': 10,
        'MOVE_NON_TARGET_ITEM': -12,
        'DAMAGE_ITEM': -5,
        'DROP_TARGET_ITEM': -3,
        }

bin_contents = dict(get_bin_contents(args.json))
work_order = dict(get_work_order(args.json))

cprint('#------------------#', 'blue')
cprint('# SCORE CALCULATOR #', 'blue')
cprint('#------------------#', 'blue')

def display_score(scores):
    cprint('---------------------', 'blue')
    for type_, score in scores.items():
        if score == 0:
            continue
        score = '+{0}'.format(score) if score > 0 else str(score)
        print('{0}: {1}'.format(type_, score))
    bin_score = sum(scores.values())
    cprint('SCORE of bin_{0}: {1}'.format(bin_.upper(), bin_score), 'magenta')
    cprint('---------------------', 'blue')

possible = 0
sum_ = 0
for bin_ in work_order:
    scores = {}
    contents = bin_contents[bin_]
    target = work_order[bin_]
    N = len(contents)
    print('What about ' + colored('bin_{0}'.format(bin_.upper()), 'magenta') + ' ?')
    print(colored('bin_{0}'.format(bin_.upper()), 'magenta') + ' includes')
    for content in contents:
        if target == content:
            print("- " + colored(target + " (Correct Target)", 'red'))
        else:
            print("- " + content)
    # succeeded or not
    while True:
        yn = raw_input(colored('Success? [y/n]: ', 'green'))
        if yn == '' or (not yn.lower() in 'yn'):
            cprint('Please enter y or n', 'red')
            continue
        if len(contents) == 1:
            score_success = SCORING['MOVE_FROM_SINGLE_ITEM_BIN']
        elif len(contents) == 2:
            score_success = SCORING['MOVE_FROM_DOUBLE_ITEM_BIN']
        else:
            score_success = SCORING['MOVE_FROM_MULTI_ITEM_BIN']
        if yn.lower() == 'y':
            scores['MOVE_TARGET_ITEM'] = score_success
            if target in BONUS:
                scores['BONUS'] = BONUS[target]
        possible += (score_success + BONUS.get(target, 0))
        break
    # carried wrong item or not
    while True:
        question = 'Moved non target objects? [0-{0}]: '.format(N)
        n_wrong = raw_input(colored(question, 'yellow'))
        try:
            n_wrong = int(n_wrong)
            if n_wrong > N or n_wrong < 0:
                cprint('Wrong input', 'red')
                continue
            break
        except ValueError:
            cprint('Please enter number', 'red')
    scores['MOVE_NON_TARGET_ITEM'] = (SCORING['MOVE_NON_TARGET_ITEM']
                                        * n_wrong)
    # dropped or not
    while True:
        question = 'Dropped objects? [0-{0}]: '.format(N)
        n_dropped = raw_input(colored(question, 'yellow'))
        try:
            n_dropped = int(n_dropped)
            if n_dropped > N or n_dropped < 0:
                cprint('Wrong input', 'red')
                continue
            break
        except ValueError:
            cprint('Please enter number', 'red')
    scores['DROP_TARGET_ITEM'] = SCORING['DROP_TARGET_ITEM'] * n_dropped
    # damaged or not
    while True:
        question = 'Damaged objects? [0-{0}]: '.format(N)
        n_damaged = raw_input(colored(question, 'yellow'))
        try:
            n_damaged = int(n_damaged)
            if n_damaged > N or n_damaged < 0:
                cprint('Wrong input', 'red')
                continue
            break
        except ValueError:
            cprint('Please enter number', 'red')
    scores['DAMAGE_ITEM'] = SCORING['DAMAGE_ITEM'] * n_damaged
    # do this before finish a loop
    display_score(scores)
    sum_ += sum(scores.values())

# display sum
cprint('# ================= #', 'magenta')
cprint('#        SUM        #', 'magenta')
cprint('# ================= #', 'magenta')
print('SCORE: {0}'.format(sum_))
print('POSSIBLE: {0}'.format(possible))
