# Author: Jochen Gast <jochen.gast@visinf.tu-darmstadt.de>

import os

from . import factory
from .writers import csv_writer
from .writers import mat_writer
from .writers import png_writer
from .writers import types


# ----------------------------------------------
# Epoch recorder:
#
#   To be initiated for every epoch.
#
# ----------------------------------------------
class EpochRecorder(object):
    def __init__(self,
                 args,
                 epoch,
                 dataset,
                 csv=True,
                 mat=True,
                 png=False,
                 mat_exhaustive=False):
        self._root = args.save
        self._epoch = epoch
        self._dataset = dataset
        self._record_writers = []

        if not os.path.exists(self._root):
            os.makedirs(self._root)

        if csv:
            writer = csv_writer.CSVRecordWriter(
                args, root=os.path.join(self._root, "csv"))
            self._record_writers.append(writer)

        if png:
            writer = png_writer.PNGRecordWriter(
                args, root=os.path.join(self._root, "png"))
            self._record_writers.append(writer)

        if mat:
            writer = mat_writer.MATRecordWriter(
                args, root=os.path.join(self._root, "mat"))
            self._record_writers.append(writer)

        if mat_exhaustive:
            writer = mat_writer.MATExhaustiveRecordWriter(
                args, root=os.path.join(self._root, "mat_exhaustive"))
            self._record_writers.append(writer)

    @property
    def root(self):
        return self._root

    @property
    def epoch(self):
        return self._epoch

    @property
    def dataset(self):
        return self._dataset

    @property
    def record_writers(self):
        return self._record_writers

    def _handle_record(self, record):
        for writer in self._record_writers:
            writer.handle_record(record)

    def add_scalars(self, example_basename, scalars, step=None, example_index=None):
        record = types.ScalarDictRecord(example_basename,
                                        data=scalars,
                                        step=step,
                                        example_index=example_index,
                                        epoch=self._epoch,
                                        dataset=self._dataset)
        self._handle_record(record)

    def add_image(self, example_basename, image, step=None, example_index=None,
                  imagesc=False, cmap=None):
        record = types.ImageRecord(example_basename,
                                   data=image,
                                   step=step,
                                   example_index=example_index,
                                   epoch=self._epoch,
                                   dataset=self._dataset,
                                   imagesc=imagesc,
                                   cmap=cmap)
        self._handle_record(record)

    def add_flow(self, example_basename, flow, step=None, example_index=None,
                 max_flow=None):
        record = types.FlowRecord(example_basename,
                                  data=flow,
                                  step=step,
                                  example_index=example_index,
                                  epoch=self._epoch,
                                  dataset=self._dataset,
                                  max_flow=max_flow)
        self._handle_record(record)

    def on_batch_end(self, example_dict, loss_dict, output_dict):
        record = types.DictionaryRecord(
            example_dict,
            loss_dict,
            output_dict,
            epoch=self._epoch,
            dataset=self._dataset)

        self._handle_record(record)


factory.register("EpochRecorder", EpochRecorder)
