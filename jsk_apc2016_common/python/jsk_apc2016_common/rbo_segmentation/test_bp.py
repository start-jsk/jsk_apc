import unittest
import pickle
import numpy as np

class TestRBO(unittest.TestCase):
    def setUp(self):
        bp_path = '/home/leus/ros/indigo/src/start-jsk/jsk_apc/jsk_apc2016_common/data/trained_segmenter_2016.pkl'
        with open(bp_path, 'rb') as f:
            self.bp = pickle.load(f)

        dataset_path = '/home/leus/ros/indigo/src/start-jsk/jsk_apc/jsk_apc2016_common/data/dataset.pkl'
        with open(dataset_path, 'rb') as f:
            self.dataset = pickle.load(f)

        self.first_sample = self.dataset.samples[0]

        self.bp.fit(self.dataset)

    def test_histograms(self):
        self.assertIn('staples_index_cards', self.bp.histograms['color'].keys())
        self.assertEqual(len(self.bp.histograms['color'].keys()), 39)

        self.assertGreater(len(self.first_sample.object_masks), 0)

        




if __name__ == '__main__':
    unittest.main()

