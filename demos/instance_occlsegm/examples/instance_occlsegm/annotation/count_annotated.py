#!/usr/bin/env python

from __future__ import division

import os
import os.path as osp


def main():
    n_videos = 0
    n_videos_done = 0
    n_frames = 0
    n_frames_done = 0

    root_dir = 'data/annotation_raw_data/20180730'
    for video_dir in sorted(os.listdir(root_dir)):
        video_dir = osp.join(root_dir, video_dir)
        n_videos += 1

        undone_exists = False
        for frame_dir in sorted(os.listdir(video_dir)):
            frame_dir = osp.join(video_dir, frame_dir)
            n_frames += 1
            anno_file = osp.join(frame_dir, 'image.json')
            if osp.exists(anno_file):
                n_frames_done += 1
            else:
                undone_exists = True
        if not undone_exists:
            n_videos_done += 1

    print('n_videos: {:d} / {:d} ({:.2%})'
          .format(n_videos_done, n_videos, n_videos_done / n_videos))
    print('n_frames: {:d} / {:d} ({:.2%})'
          .format(n_frames_done, n_frames, n_frames_done / n_frames))


if __name__ == '__main__':
    main()
