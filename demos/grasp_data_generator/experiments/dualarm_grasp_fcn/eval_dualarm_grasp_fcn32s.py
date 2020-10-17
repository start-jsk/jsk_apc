import argparse
import fcn
import numpy as np
import os.path as osp

import chainer
from chainer.backends import cuda

from grasp_data_generator.datasets import SemanticRealAnnotatedDatasetV1
from grasp_data_generator.evaluations import eval_semantic_segmentation
from grasp_data_generator.models import DualarmGraspFCN32s


thisdir = osp.dirname(osp.abspath(__file__))


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--gpu', '-g', type=int, help='GPU id.', default=0)
    parser.add_argument('--pretrained-model', type=str)
    parser.add_argument('--iou-thresh', type=float, default=0.5)
    parser.add_argument('--dataset', default='ev1', choices=['ev1'])
    args = parser.parse_args()

    if args.dataset == 'ev1':
        test_data = SemanticRealAnnotatedDatasetV1(
            split='all', imgaug=False)
        n_class = len(test_data.label_names)
    else:
        raise ValueError(
            'Given dataset is not supported: {}'.format(args.dataset))

    if args.pretrained_model is None:
        raise ValueError('pretrained model is not set: --pretrained-model')

    model = DualarmGraspFCN32s(n_class=n_class)
    chainer.serializers.load_npz(
        osp.join(thisdir, args.pretrained_model), model)

    chainer.global_config.train = False
    chainer.global_config.enable_backprop = False
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    gt_labels = []
    pred_labels = []
    for i in range(len(test_data)):
        img_data, gt_label = test_data[i]
        gt_labels.append(gt_label)
        x_data = fcn.datasets.transform_lsvrc2012_vgg16([img_data])
        x_data = np.array(x_data, dtype=np.float32)
        x_data = cuda.to_gpu(x_data, device=args.gpu)
        x = chainer.Variable(x_data)
        model(x)

        scores = model.score
        probs = chainer.functions.softmax(scores).array
        pred_label = cuda.to_cpu(
            chainer.functions.argmax(probs, axis=1).array)[0]
        pred_label = pred_label
        pred_labels.append(pred_label)
    result = eval_semantic_segmentation(pred_labels, gt_labels)

    print('')
    print('pixel accuracy      : {:f}'.format(result['pixel_accuracy']))
    print('mean class accuracy : {:f}'.format(result['mean_class_accuracy']))
    print('miou                : {:f}'.format(result['miou']))
    print('fwiou               : {:f}'.format(result['fwiou']))


if __name__ == '__main__':
        main()
