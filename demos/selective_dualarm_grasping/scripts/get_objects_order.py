import os.path as osp
import random
import yaml


filepath = osp.dirname(osp.realpath(__file__))
yamlpath = osp.join(filepath, '../yaml/dualarm_grasping_label_names.yaml')
with open(yamlpath, 'r') as f:
    object_names = yaml.load(f)['label_names'][1:]

order = range(len(object_names))
order = random.sample(order, len(order))
for i, ordr in enumerate(order):
    print('{}: {}'.format(i, object_names[ordr]))
