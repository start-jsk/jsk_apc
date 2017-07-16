import numpy as np
import warnings

import chainer
from chainer import cuda

from chainercv.transforms import center_crop
from chainercv.transforms import scale
from chainercv.utils.iterator import apply_prediction_to_iterator

from jsk_arc2017_common.in_hand_recognition.calc_nearest_neighbors import calc_nearest_neighbors 


# RGB order
_imagenet_mean = np.array(
    [123.68, 116.779, 103.939], dtype=np.float32)[:, np.newaxis, np.newaxis]


class RetrievalPredictor(chainer.Chain):

    def __init__(self, extractor, mean=_imagenet_mean,
                 size=(224, 224), scale_size=256, k=5):
        super(RetrievalPredictor, self).__init__()
        self.mean = mean
        self.scale_size = scale_size
        self.size = size
        self.k = 5

        with self.init_scope():
            self.extractor = extractor

        self.db_features = None

    def _prepare(self, img):
        """Prepare an image to be used for prediction.

        Args:
            img (~numpy.ndarray): An image. This is in CHW and RGB format.
                The range of its value is :math:`[0, 255]`.

        Returns:
            ~numpy.ndarray:
            A preprocessed image.

        """
        img = scale(img, size=self.scale_size)
        img = center_crop(img, self.size)
        img = img - self.mean

        return img

    def extract(self, imgs):
        """Extract features from raw images

        Args:
            imgs (~numpy.ndarray): Batch of images. An image is in CHW and RGB
                format.
                The range of its value is :math:`[0, 255]`.

        """
        imgs = [self._prepare(img) for img in imgs]
        imgs = self.xp.asarray(imgs).reshape(-1, 3, 224, 224)

        with chainer.function.no_backprop_mode():
            with chainer.using_config('train', False):
                imgs = chainer.Variable(imgs)
                activations = self.extractor(imgs)

        output = cuda.to_cpu(activations.data)
        return output

    def load_db(self, it):
        """Set database features.

        Args:
            it (chainer.dataset.Iterator)

        """
        if it._repeat:
            raise ValueError('This does not accept infinite length iterator. '
                             'Please set `repeat` option of iterator to False.')
        if it._shuffle:
            warnings.warn('`shuffle` is True. Is this OK?')

        imgs, (db_features,), (db_labels,) = apply_prediction_to_iterator(
            self.extract, it)
        del imgs
        self.db_features = np.array(list(db_features))
        self.db_labels = np.array(list(db_labels))

    def predict(self, imgs):
        """Find K nearest negihbors of query images from the database.

        Args
            imgs: batch of query images.

        """
        query_features = self.extract(imgs)

        if self.db_features is None:
            raise ValueError('Please prepare database features. '
                             'This can be done with method `load_db`.')

        top_k, _ = calc_nearest_neighbors(
            query_features, self.db_features, self.k)
        return top_k
