import argparse
import datetime
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import os.path as osp
import PIL.Image
import PIL.ImageDraw
import scipy.misc
import yaml

from chainercv.utils.mask.mask_to_bbox import mask_to_bbox

from grasp_data_generator.visualizations \
    import vis_occluded_instance_segmentation


filepath = osp.dirname(osp.realpath(__file__))
dataset_dir = osp.join(filepath, '../data/evaluation_data')
yamlpath = osp.join(filepath, '../yaml/dualarm_grasping_label_names.yaml')


def main(datadir, visualize):
    time = datetime.datetime.now()
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    with open(yamlpath, 'r') as yaml_f:
        label_names = yaml.load(yaml_f)[1:]

    for scene_d in sorted(os.listdir(datadir)):
        scenedir = osp.join(datadir, scene_d)
        ins_imgs = []
        label = []
        for time_d in sorted(os.listdir(scenedir))[::-1]:
            timedir = osp.join(scenedir, time_d)
            savedir = osp.join(dataset_dir, timestamp, time_d)
            if not osp.exists(savedir):
                os.makedirs(savedir)

            rgbpath = osp.join(timedir, 'masked_rgb.png')
            annopath = osp.join(timedir, 'masked_rgb.json')
            rgb = scipy.misc.imread(rgbpath)

            with open(annopath, 'r') as json_f:
                data = json.load(json_f)
            H, W = data['imageHeight'], data['imageWidth']
            msk = np.zeros((H, W), dtype=np.uint8)

            msk = PIL.Image.fromarray(msk)
            draw = PIL.ImageDraw.Draw(msk)
            shape = data['shapes'][0]
            label_name = shape['label']
            points = shape['points']
            xy = [tuple(point) for point in points]
            draw.polygon(xy=xy, outline=1, fill=1)
            msk = np.array(msk, dtype=np.int32)

            next_ins_imgs = []
            next_label = []
            for ins_id, (ins_img, lbl) in enumerate(zip(ins_imgs, label)):
                occ_msk = np.logical_and(ins_img > 0, msk > 0)
                ins_img[occ_msk] = 2
                if not np.any(ins_img == 1):
                    print('{} is occluded and no more visible'
                          .format(label_names[lbl]))
                else:
                    next_ins_imgs.append(ins_img)
                    next_label.append(lbl)

            ins_imgs = next_ins_imgs
            label = next_label
            ins_imgs.append(msk[None])
            lbl = label_names.index(label_name)
            label.append(lbl)

            if visualize:
                vis_rgb = rgb.transpose((2, 0, 1))
                vis_ins_imgs = np.concatenate(
                    ins_imgs, axis=0).astype(np.int32)
                bbox = mask_to_bbox(vis_ins_imgs > 0)
                vis_occluded_instance_segmentation(
                    vis_rgb, vis_ins_imgs, label, bbox,
                    label_names=label_names)
                plt.show()

            rgb_savepath = osp.join(savedir, 'rgb.png')
            ins_imgs_savepath = osp.join(savedir, 'ins_imgs.npz')
            label_savepath = osp.join(savedir, 'labels.yaml')

            scipy.misc.imsave(rgb_savepath, rgb)
            np.savez_compressed(
                ins_imgs_savepath,
                ins_imgs=np.concatenate(ins_imgs, axis=0).astype(np.int32))
            np.savez_compressed
            with open(label_savepath, 'w+') as yaml_save_f:
                yaml_save_f.write(yaml.dump(label))

    with open(osp.join(dataset_dir, timestamp, 'label_names.yaml'), 'w+') as f:
        f.write(yaml.dump(label_names))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--visualize', '-v', action='store_true')
    parser.add_argument('--data-dir', '-d')
    args = parser.parse_args()

    datadir = osp.join(filepath, args.data_dir)
    main(datadir, args.visualize)
