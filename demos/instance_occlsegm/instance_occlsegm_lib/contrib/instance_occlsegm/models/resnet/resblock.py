from __future__ import print_function

from chainer.functions.activation.relu import relu
from chainer import link
from chainer.links.connection.convolution_2d import Convolution2D
from chainer.links.normalization.batch_normalization import BatchNormalization


class BuildingBlock(link.Chain):

    """A building block that consists of several Bottleneck layers.

    Args:
        n_layer (int): Number of layers used in the building block.
        in_channels (int): Number of channels of input arrays.
        mid_channels (int): Number of channels of intermediate arrays.
        out_channels (int): Number of channels of output arrays.
        stride (int or tuple of ints): Stride of filter application.
        initialW (4-D array): Initial weight value used in
            the convolutional layers.
    """

    def __init__(self, n_layer, in_channels, mid_channels,
                 out_channels, stride, dilate=1, initialW=None):
        super(BuildingBlock, self).__init__()
        with self.init_scope():
            self.a = BottleneckA(
                in_channels, mid_channels, out_channels, stride, dilate,
                initialW)
            self._forward = ["a"]
            for i in range(n_layer - 1):
                name = 'b{}'.format(i + 1)
                bottleneck = BottleneckB(
                    out_channels, mid_channels, dilate, initialW)
                setattr(self, name, bottleneck)
                self._forward.append(name)

    def __call__(self, x):
        for name in self._forward:
            l = getattr(self, name)  # NOQA
            x = l(x)
        return x

    @property
    def forward(self):
        return [getattr(self, name) for name in self._forward]


class BottleneckA(link.Chain):

    """A bottleneck layer that reduces the resolution of the feature map.

    Args:
        in_channels (int): Number of channels of input arrays.
        mid_channels (int): Number of channels of intermediate arrays.
        out_channels (int): Number of channels of output arrays.
        stride (int or tuple of ints): Stride of filter application.
        initialW (4-D array): Initial weight value used in
            the convolutional layers.
    """

    def __init__(self, in_channels, mid_channels, out_channels,
                 stride=2, dilate=1, initialW=None):
        super(BottleneckA, self).__init__()
        with self.init_scope():
            self.conv1 = Convolution2D(
                in_channels, mid_channels, 1, stride, 0, initialW=initialW,
                nobias=True)
            self.bn1 = BatchNormalization(mid_channels)
            self.conv2 = Convolution2D(
                mid_channels, mid_channels, 3, 1, dilate, initialW=initialW,
                nobias=True, dilate=dilate)
            self.bn2 = BatchNormalization(mid_channels)
            self.conv3 = Convolution2D(
                mid_channels, out_channels, 1, 1, 0, initialW=initialW,
                nobias=True)
            self.bn3 = BatchNormalization(out_channels)
            self.conv4 = Convolution2D(
                in_channels, out_channels, 1, stride, 0, initialW=initialW,
                nobias=True)
            self.bn4 = BatchNormalization(out_channels)

    def __call__(self, x):
        h1 = relu(self.bn1(self.conv1(x)))
        h1 = relu(self.bn2(self.conv2(h1)))
        h1 = self.bn3(self.conv3(h1))
        h2 = self.bn4(self.conv4(x))
        return relu(h1 + h2)


class BottleneckB(link.Chain):

    """A bottleneck layer that maintains the resolution of the feature map.

    Args:
        in_channels (int): Number of channels of input and output arrays.
        mid_channels (int): Number of channels of intermediate arrays.
        initialW (4-D array): Initial weight value used in
            the convolutional layers.
    """

    def __init__(self, in_channels, mid_channels, dilate=1, initialW=None):
        super(BottleneckB, self).__init__()
        with self.init_scope():
            self.conv1 = Convolution2D(
                in_channels, mid_channels, 1, 1, 0, initialW=initialW,
                nobias=True)
            self.bn1 = BatchNormalization(mid_channels)
            self.conv2 = Convolution2D(
                mid_channels, mid_channels, 3, 1, dilate, initialW=initialW,
                nobias=True, dilate=dilate)
            self.bn2 = BatchNormalization(mid_channels)
            self.conv3 = Convolution2D(
                mid_channels, in_channels, 1, 1, 0, initialW=initialW,
                nobias=True)
            self.bn3 = BatchNormalization(in_channels)

    def __call__(self, x):
        h = relu(self.bn1(self.conv1(x)))
        h = relu(self.bn2(self.conv2(h)))
        h = self.bn3(self.conv3(h))
        return relu(h + x)
