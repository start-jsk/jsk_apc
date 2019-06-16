import copy

from chainer import reporter
import chainer.training.extensions
from chainercv.utils import apply_to_iterator

from ..evaluations import eval_sigmoid_segmentation


class SigmoidSegmentationEvaluator(chainer.training.extensions.Evaluator):

    trigger = 1, 'epoch'
    default_name = 'validation'
    priority = chainer.training.PRIORITY_WRITER

    def __init__(self, iterator, target, channel_names=None):
        super(SigmoidSegmentationEvaluator, self).__init__(iterator, target)
        self.channel_names = channel_names

    def evaluate(self):
        iterator = self._iterators['main']
        target = self._targets['main']

        if hasattr(iterator, 'reset'):
            iterator.reset()
            it = iterator
        else:
            it = copy.copy(iterator)

        in_values, out_values, rest_values = apply_to_iterator(
            target.predict, it, n_input=2,
        )
        # delete unused iterators explicitly
        del in_values

        pred_labels, = out_values
        gt_labels, = rest_values

        report = eval_sigmoid_segmentation(
            pred_labels, gt_labels, channel_names=self.channel_names
        )

        observation = {}
        with reporter.report_scope(observation):
            reporter.report(report, target)
        return observation
