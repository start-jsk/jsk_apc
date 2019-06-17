#!/usr/bin/env python

from ..models import BVLCAlex
import chainer
import fcn


def copy_alex_chainermodel(chainermodel_path, model):
    bvlc_model = BVLCAlex()
    chainer.serializers.load_hdf5(chainermodel_path, bvlc_model)
    for link in bvlc_model.children():
        link_name = link.name
        if link_name.startswith('fc'):
            continue
        if getattr(model, link_name):
            layer = getattr(model, link_name)
            if layer.W.data.shape == link.W.data.shape:
                layer.W.data = link.W.data
            else:
                print('link_name {0} has different shape {1} != {2}'.format(
                    link_name, layer.W.data.shape, link.W.data.shape))


def copy_vgg16_chainermodel(model):
    vgg16_model = fcn.models.VGG16()
    vgg16_path = vgg16_model.download()
    chainer.serializers.load_npz(vgg16_path, vgg16_model)
    for l in vgg16_model.children():
        if l.name.startswith('conv'):
            l1 = getattr(vgg16_model, l.name)
            l2 = getattr(model, l.name)
            assert l1.W.shape == l2.W.shape
            assert l1.b.shape == l2.b.shape
            l2.W.data[...] = l1.W.data[...]
            l2.b.data[...] = l1.b.data[...]
        elif l.name in ['fc6', 'fc7']:
            if not hasattr(model, l.name):
                continue
            l1 = getattr(vgg16_model, l.name)
            l2 = getattr(model, l.name)
            if l1.W.size == l2.W.size and l1.b.size == l2.b.size:
                l2.W.data[...] = l1.W.data.reshape(l2.W.shape)[...]
                l2.b.data[...] = l1.b.data.reshape(l2.b.shape)[...]
