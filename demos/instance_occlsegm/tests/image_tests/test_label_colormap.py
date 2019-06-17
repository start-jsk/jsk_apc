import instance_occlsegm_lib


def test_label_colormap():
    n_labels = 5
    cmap = instance_occlsegm_lib.image.label_colormap(n_labels)
    assert len(cmap) == n_labels
