import copy

from chainer import reporter
import chainer.training.extensions

from chainercv.evaluations import eval_semantic_segmentation
from chainercv.utils import apply_to_iterator

from ..evaluations import eval_occlusion_segmentation


class OcclusionSegmentationEvaluator(chainer.training.extensions.Evaluator):

    trigger = 1, 'epoch'
    default_name = 'validation'
    priority = chainer.training.PRIORITY_WRITER

    def __init__(self, iterator, target, label_names=None):
        super(OcclusionSegmentationEvaluator, self).__init__(
            iterator, target)
        self.label_names = label_names

    def evaluate(self):
        iterator = self._iterators['main']
        target = self._targets['main']

        if hasattr(iterator, 'reset'):
            iterator.reset()
            it = iterator
        else:
            it = copy.copy(iterator)

        in_values, out_values, rest_values = apply_to_iterator(
            target.predict, it)
        # delete unused iterators explicitly
        del in_values

        pred_labels, pred_labels_occ = out_values
        gt_labels, gt_labels_occ = rest_values

        result_vis = eval_semantic_segmentation(pred_labels, gt_labels)
        result_occ = eval_occlusion_segmentation(
            pred_labels_occ, gt_labels_occ
        )

        report = {
            'miou': (result_vis['miou'] + result_occ['miou']) / 2.,
            'miou/vis': result_vis['miou'],
            'miou/occ': result_occ['miou'],
        }

        # if self.label_names is not None:
        #     for l, label_name in enumerate(self.label_names):
        #         try:
        #             report['iou/{:s}'.format(label_name)] = result['iou'][l]
        #         except IndexError:
        #             report['iou/{:s}'.format(label_name)] = np.nan
        #
        #         if l == 0:
        #             continue
        #
        #         try:
        #             report['iou_occ/{:s}'.format(label_name)] = \
        #                 result_occ['iou'][l - 1]
        #         except IndexError:
        #             result_occ['iou_occ/{:s}'.format(label_name)] = np.nan

        observation = {}
        with reporter.report_scope(observation):
            reporter.report(report, target)
        return observation
