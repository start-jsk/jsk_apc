#!/usr/bin/env python

import glob
import os
import os.path as osp
import pprint

import numpy as np
import tabulate

import contrib


here = osp.dirname(osp.abspath(__file__))

dirname = osp.join(here, 'annotation_raw_data/20180204_annotated')

class_names = contrib.core.get_class_names()
object_freq_by_video = {class_id: 0 for class_id in range(1, 41)}
object_freq_by_frame = {class_id: 0 for class_id in range(1, 41)}
stats = dict(
    n_frames=0,
    n_annotations=0,
    n_class=len(class_names),
)
for video_dir in sorted(os.listdir(dirname)):
    if video_dir == '__invalid__':
        continue

    video_dir = osp.join(dirname, video_dir)
    # print('Checking: %s' % video_dir)
    json_files = sorted(glob.glob(osp.join(video_dir, '*.json')))

    class_ids_in_video = set()
    for json_file in json_files:
        img, lbl_ins, lbl_cls = contrib.core.load_json_file(json_file)

        class_ids = np.unique(lbl_cls)
        keep = ~np.isin(class_ids, [-1, 0])
        class_ids = class_ids[keep]

        # print(class_ids)
        for class_id in class_ids:
            object_freq_by_frame[class_id] += 1
            class_ids_in_video.add(class_id)
            stats['n_annotations'] += 1

        stats['n_frames'] += 1

    print('%s: %s' % (video_dir, class_ids_in_video))
    for class_id in class_ids_in_video:
        object_freq_by_video[class_id] += 1

print('Stats:')
pprint.pprint(stats)

object_freq_by_video = dict(object_freq_by_video)
object_freq_by_frame = dict(object_freq_by_frame)

# print('Object Frequency by Video:')
# pprint.pprint(object_freq_by_video)
# print('Object Frequency by Frame:')
# pprint.pprint(object_freq_by_frame)

headers = ['class_id', 'class_name', 'freq_video', 'freq_frame']
rows = []
for class_id, freq in sorted(object_freq_by_video.items(), key=lambda x: x[1]):
    freq_frame = object_freq_by_frame[class_id]
    rows.append((class_id, class_names[class_id], freq, freq_frame))
print(tabulate.tabulate(rows, headers=headers))
