class ClassSegmentationDatasetBase(object):

    def __init__(self, split):
        self._split = split

    def download(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        """Return pair of input & output.

        Returns:
            img: numpy.ndarray
                RGB image (H, W, 3).
            lbl: numpy.ndarray
                Label image (H, W).
        """
        raise NotImplementedError

    @property
    def split(self):
        return self._split

    @property
    def n_class(self):
        return len(self._class_names)

    @property
    def class_names(self):
        return self._class_names
