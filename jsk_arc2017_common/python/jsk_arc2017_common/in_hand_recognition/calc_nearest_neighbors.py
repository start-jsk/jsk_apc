import numpy as np
import cv2


def calc_nearest_neighbors(query_feats, db_feats, k):
    """Calculate K Nearest Neighbor

    """
    query_feats = np.array(list(query_feats))
    db_feats = np.array(list(db_feats))
    labels = np.arange(len(db_feats))
    if cv2.__version__.startswith('2.4'):
        knn = cv2.KNearest()
        knn.train(db_feats, labels)
        _, _, indices, dists = knn.find_nearest(query_feats, k)
    elif cv2.__version__ == '3.2.0':
        knn = cv2.ml.KNearest_create()
        knn.train(db_feats, cv2.ml.ROW_SAMPLE, labels)
        _, _, indices, dists = knn.findNearest(query_feats, k)
    else:
        raise ValueError('only version OpenCV with version 2.4.* and 3.2 '
                         'are supported.')
    return indices.astype(np.int32), dists
