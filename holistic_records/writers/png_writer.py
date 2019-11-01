# Author: Jochen Gast <jochen.gast@visinf.tu-darmstadt.de>

import os

import matplotlib.pyplot as plt
import numpy as np

from . import types


def _imwrite(filename, image, imagesc=False, cmap=None):
    if imagesc:
        cmin = np.min(image[:])
        cmax = np.max(image[:])
        image = (image - cmin) / (cmax - cmin)
    plt.imsave(filename, image, cmap=cmap)


class PNGRecordWriter(types.RecordWriter):
    def __init__(self, args, root):
        super().__init__()
        self._args = args
        self._root = root
        if not os.path.exists(root):
            os.makedirs(root)

    def handle_image(self, record):
        record = types.record2list(record)
        batch_size = record.data.size(0)
        np_data = types.tensor2numpy(record.data)
        for i in range(batch_size):

            filename = "{}/{}/epoch-{:03d}/{}.png".format(
                self._root, record.dataset, record.epoch, record.example_basename[i])

            if not os.path.exists(os.path.dirname(filename)):
                os.makedirs(os.path.dirname(filename))

            if os.path.isfile(filename):
                raise ValueError("Error in PNGRecordWriter. '%s' already exists." % filename)

            np_image = np_data[i, ...].squeeze()

            _imwrite(filename=filename,
                     image=np_image,
                     imagesc=record.imagesc,
                     cmap=record.cmap)

    def handle_record(self, record):
        if isinstance(record, types.ImageRecord):
            return self.handle_image(record)
