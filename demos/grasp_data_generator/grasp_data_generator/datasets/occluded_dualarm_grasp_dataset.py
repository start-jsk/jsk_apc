from networkx.drawing.nx_pydot import read_dot
import numpy as np
import os.path as osp

from grasp_data_generator.datasets.dualarm_grasp_dataset \
    import DualarmGraspDataset


filepath = osp.dirname(osp.realpath(__file__))


class OccludedDualarmGraspDataset(DualarmGraspDataset):

    def get_example(self, i):
        img, label, sg, dg = super(
            OccludedDualarmGraspDataset, self).get_example(i)
        data_id = self._ids[self.split][i]
        graph = read_dot(
            osp.join(self.datadir, data_id, 'graph.dot'))
        n_label = len(self.label_names) - 1
        graph_img = np.zeros((n_label, n_label), dtype=np.int32)
        nodes = list(graph.nodes)
        edges = list(graph.edges)
        for node in nodes:
            index = self.label_names.index(node) - 1
            graph_img[index, index] = 1
        for edge in edges:
            start_index = self.label_names.index(edge[0]) - 1
            end_index = self.label_names.index(edge[1]) - 1
            graph_img[start_index, end_index] = 1
        return img, label, sg, dg, graph_img


class OccludedDualarmGraspDatasetV1(OccludedDualarmGraspDataset):

    datadir = osp.join(
        filepath, '../../data/training_data/', '20171116_091524')
