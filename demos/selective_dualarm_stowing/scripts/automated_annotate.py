import chainer
import chainer.serializers as S
import datetime
import fcn
import numpy as np
import os
import os.path as osp
import scipy.misc
import shutil

import rospkg


rospack = rospkg.RosPack()
data_dir = osp.expanduser('~/data/datasets/selective_dualarm_stowing')
mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
label_names = [
    '__background__',
    'barkely_hide_bones',
    'cherokee_easy_tee_shirt',
    'clorox_utility_brush',
    'cloud_b_plush_bear',
    'command_hooks',
    'cool_shot_glue_sticks',
    'crayola_24_ct',
    'creativity_chenille_stems',
    'dasani_water_bottle',
    'dove_beauty_bar',
    'dr_browns_bottle_brush',
    'easter_turtle_sippy_cup',
    'elmers_washable_no_run_school_glue',
    'expo_dry_erase_board_eraser',
    'fiskars_scissors_red',
    'fitness_gear_3lb_dumbbell',
    'folgers_classic_roast_coffee',
    'hanes_tube_socks',
    'i_am_a_bunny_book',
    'jane_eyre_dvd',
    'kleenex_paper_towels',
    'kleenex_tissue_box',
    'kyjen_squeakin_eggs_plush_puppies',
    'laugh_out_loud_joke_book',
    'oral_b_toothbrush_green',
    'oral_b_toothbrush_red',
    'peva_shower_curtain_liner',
    'platinum_pets_dog_bowl',
    'rawlings_baseball',
    'rolodex_jumbo_pencil_cup',
    'safety_first_outlet_plugs',
    'scotch_bubble_mailer',
    'scotch_duct_tape',
    'soft_white_lightbulb',
    'staples_index_cards',
    'ticonderoga_12_pencils',
    'up_glucose_bottle',
    'womens_knit_gloves',
    'woods_extension_cord'
]


def main(inputdir, outputdir, evaldir):
    annotate_and_save(inputdir, outputdir)
    eval_annotate(outputdir, evaldir)


def eval_annotate(outputdir, evaldir):
    datadirs = os.listdir(outputdir)
    correct_count = 0
    total_count = 0
    for d in datadirs:
        output_labelpath = osp.join(outputdir, d, 'label.txt')
        eval_labelpath = osp.join(evaldir, d, 'label.txt')
        if osp.exists(output_labelpath) and osp.exists(eval_labelpath):
            with open(output_labelpath, 'r') as f:
                output_label = f.read()
            output_label = output_label.split('\n')[0]
            with open(eval_labelpath, 'r') as f:
                eval_label = f.read()
            eval_label = eval_label.split('\n')[0]
            if output_label == eval_label:
                correct_count += 1
            else:
                print('=============================')
                print('dir : {}'.format(d))
                print('pred: {}'.format(output_label))
                print('gt  : {}'.format(eval_label))
                print('=============================')
            total_count += 1
    accuracy = correct_count / float(total_count)
    print('accuracy: {}'.format(accuracy))


def annotate_and_save(inputdir, outputdir, gpu=None):
    dirs = [
        '20170905_stems',
        '20170905_socks',
        '20170906_robots_everywhere',
        '20170906_curtain',
        '20170913_tee_shirt',
        '20170913_ice_cube_tray',
        '20170913_expo_eraser',
        '20170913_paper_towels',
        '20170913_mark_twain_book',
        '20170913_feline_greenies',
        '20170914_stems',
    ]

    stows = [
        'dualarm',
        'singlearm'
    ]

    if gpu is None:
        model = None
    else:
        chainer.cuda.get_device_from_id(gpu).use()
        n_class = len(label_names)
        model_file = osp.join(
            rospack.get_path('selective_dualarm_stowing'),
            'models/fcn32s_v2_148000.chainermodel')
        model = fcn.models.FCN32s(n_class=n_class)
        S.load_hdf5(model_file, model)
        model.to_gpu()

    for d in dirs:
        item_name = d.split('_')[1]
        for stow in stows:
            day = d.split('_')[0]
            stow_d = '{}_{}_stow'.format(day, stow)
            stow_dir = osp.join(inputdir, d, stow_d)
            if not osp.exists(stow_dir):
                continue
            trial_dirs = os.listdir(stow_dir)
            for trial_d in trial_dirs:
                trial_d = osp.join(stow_dir, trial_d)
                annotate_dirs(
                    trial_d, outputdir, stow, item_name, model)


def annotate_dirs(trial_d, outputdir, stow, item_name, model=None):
    labels, imgpath, droppedpath = annotate_label(
        trial_d, stow, item_name, model)
    if len(labels) > 0:
        save(trial_d, outputdir, labels, imgpath, droppedpath)


def annotate_label(trial_d, stow, item_name, model):
    labels = []
    after_dir = osp.join(trial_d, 'after_stow')
    after_dirs = os.listdir(after_dir)
    if len(after_dirs) == 0:
        return labels, None, None
    for after_d in after_dirs:
        imgpath = osp.join(after_dir, after_d, 'default_camera_image.png')
        droppedpath = osp.join(after_dir, after_d, 'dropped.txt')
        if osp.exists(imgpath):
            if model is None:
                ann = annotate_heuristic(imgpath, item_name)
            else:
                ann = annotate_cnn(imgpath, item_name, model)
            labels.append('{}_{}'.format(stow, ann))
    return labels, imgpath, droppedpath


def annotate_heuristic(imgpath, item_name):
    img = scipy.misc.imread(imgpath)
    center_img = img[250:350, 150:450, :]
    gray_center_img = np.average(center_img, axis=2)
    center_mask = gray_center_img > 50
    down_img = img[350:, 150:450, :]
    gray_down_img = np.average(down_img, axis=2)
    down_mask = gray_down_img > 50
    if np.average(down_mask) > 0.4:
        label = 'protrude'
    elif np.average(center_mask) > 0.4:
        label = 'success'
    else:
        label = 'drop'
    return label


def annotate_cnn(imgpath, item_name, model):
    img = scipy.misc.imread(imgpath)
    img = img[:, :, ::-1]
    x_data = img - mean_bgr
    x_data = x_data.transpose((2, 0, 1))
    x_data = np.array([x_data], dtype=np.float32)
    x_data = chainer.cuda.to_gpu(x_data)
    x = chainer.Variable(x_data)
    x.to_gpu()
    with chainer.using_config('train', False):
        with chainer.no_backprop_mode():
            model(x)
    proba_img = chainer.functions.softmax(model.score)
    proba_img = proba_img.array
    proba_img = proba_img[0]
    proba_img = chainer.cuda.to_cpu(proba_img)
    mask = [item_name in label for label in label_names]
    mask[0] = True
    mask = np.array(mask)
    proba_img = proba_img[mask]
    proba_img = proba_img.transpose((1, 2, 0))
    proba_img = proba_img / np.sum(proba_img, axis=2)[:, :, None]
    item_mask = proba_img[:, :, 1] > 0.5
    center_item_mask = item_mask[250:350, 150:450]
    down_item_mask = item_mask[350:, 150:450]
    if np.average(center_item_mask) > 0.5:
        label = 'success'
    elif np.average(down_item_mask) > 0.3:
        label = 'protrude'
    else:
        label = 'drop'
    return label


def save(trial_d, outputdir, labels, imgpath, droppedpath):
    during_dir = osp.join(trial_d, 'during_stow')
    during_dirs = os.listdir(during_dir)
    for during_d in during_dirs:
        output_d = osp.join(outputdir, during_d)
        during_d = osp.join(during_dir, during_d)
        shutil.copytree(during_d, output_d)
        shutil.copy(imgpath, osp.join(output_d, 'default_camera_image.png'))
        shutil.copy(droppedpath, osp.join(output_d, 'dropped.txt'))
        label_path = osp.join(output_d, 'label.txt')
        text = ''
        for l in labels:
            text = text + l + '\n'
        with open(label_path, 'w') as f:
            f.write(text)


if __name__ == '__main__':
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    inputdir = osp.join(data_dir, 'data_collection')
    outputdir = osp.join(data_dir, 'annotated', timestamp)
    evaldir = osp.join(data_dir, 'human_annotated', 'v5')
    main(inputdir, outputdir, evaldir)
