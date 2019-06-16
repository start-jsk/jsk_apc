import grasp_fusion_lib


def test_label_colormap():
    n_labels = 5
    cmap = grasp_fusion_lib.image.label_colormap(n_labels)
    assert len(cmap) == n_labels
