#!/usr/bin/env python

import argparse

import torch

parser = argparse.ArgumentParser()
parser.add_argument('checkpoint')
parser.add_argument('model')
args = parser.parse_args()

checkpoint = torch.load(args.checkpoint)
torch.save(checkpoint['model_state_dict'], args.model)
