#!/usr/bin/env python


def convert_to_chainermodel(caffemodel, model):
    model_links = [x.name for x in model.children()]
    caffemodel_links = [x.name for x in caffemodel.children()]
    for link_name in caffemodel_links:
        if any(x == link_name for x in model_links):
            caffemodel_layer = getattr(caffemodel, link_name)
            model_layer = getattr(model, link_name)
            print('link_name: {}'.format(link_name))
            print('W: {0} {1}'.format(
                model_layer.W.data.shape,
                caffemodel_layer.W.data.shape))
            assert model_layer.W.data.shape == caffemodel_layer.W.data.shape
            model_layer.W.data = caffemodel_layer.W.data
            print('b: {0} {1}'.format(
                model_layer.b.data.shape,
                caffemodel_layer.b.data.shape))
            assert model_layer.b.data.shape == caffemodel_layer.b.data.shape
            model_layer.b.data = caffemodel_layer.b.data
