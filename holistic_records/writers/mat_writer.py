# Author: Jochen Gast <jochen.gast@visinf.tu-darmstadt.de>

import os

import numpy as np
import torch

from utils.matlab import load, save
from . import types


def default_collate(inputs):
    if np.isscalar(inputs):
        return inputs
    else:
        return {key: default_collate([d[key] for d in inputs]) for key in inputs[0]}


class MATRecordWriter(types.RecordWriter):
    def __init__(self, args, root):
        super().__init__()
        self.args = args
        self.root = root
        if not os.path.exists(root):
            os.makedirs(root)

    def handle_image(self, record):
        record = types.record2list(record)
        batch_size = record.data.size(0)
        np_data = types.tensor2numpy(record.data)
        for i in range(batch_size):
            filename = "{}/{}/epoch-{:03d}/{}.mat".format(
                self.root, record.dataset, record.epoch, record.example_basename[i])

            if not os.path.exists(os.path.dirname(filename)):
                os.makedirs(os.path.dirname(filename))

            if os.path.isfile(filename):
                raise ValueError("Error in MATRecordWriter. '%s' already exists." % filename)

            np_image = np_data[i, ...].squeeze()
            _save_mat(filename, vardict={"data": np_image})

    def handle_scalar_dict(self, record):
        filename = "%s/%s_%s.mat" % (self.root, record.dataset, record.example_basename)

        dict_of_values = dict(record.data)
        dict_of_values["epoch"] = record.epoch

        if record.step is not None:
            dict_of_values["step"] = record.step

        if record.example_index is not None:
            dict_of_values["example_index"] = record.example_index

        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))

        if not os.path.isfile(filename):
            _save_mat(filename, vardict=dict_of_values)

        else:
            old_record = _load_mat(filename)
            new_record = {key: value for key, value in old_record.items()
                          if key in dict_of_values.keys()}

            for key, value in dict_of_values.items():
                if np.isscalar(old_record[key]):
                    new_record[key] = [old_record[key], value]
                else:
                    new_record[key] = np.append(old_record[key], value)

            _save_mat(filename, new_record)

    def handle_record(self, record):

        if isinstance(record, types.ImageRecord):
            return self.handle_image(record)

        elif isinstance(record, types.ScalarDictRecord):
            return self.handle_scalar_dict(record)


class MATExhaustiveRecordWriter(types.RecordWriter):
    def __init__(self, args, root):
        self.args = args
        self.root = root
        if not os.path.exists(root):
            os.makedirs(root)

    def handle_record(self, record):
        if isinstance(record, types.DictionaryRecord):
            return self.handle_dictionary(record)

    def handle_dictionary(self, record):
        example_dict = record.example_dict
        output_dict = record.output_dict
        filename = "%s/%s_exhaustive.mat" % (self.root, record.dataset)

        def _convert_to_numpy(x):
            if isinstance(x, torch.autograd.Variable) or isinstance(x, torch.Tensor) or isinstance(x, torch.LongTensor):
                return types.tensor2numpy(x)
            return x

        dict_of_values = {}
        for key, value in example_dict.items():
            dict_of_values[key] = _convert_to_numpy(value)
        for key, value in output_dict.items():
            dict_of_values[key] = _convert_to_numpy(value)

        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))

        if not os.path.isfile(filename):
            save(filename, vardict=dict_of_values)

        else:
            old_record = load(filename)
            new_record = {key: value for key, value in old_record.items()
                          if key in dict_of_values.keys()}

            for key, value in dict_of_values.items():
                if np.isscalar(old_record[key]):
                    new_record[key] = [old_record[key], value]
                else:
                    if isinstance(value, list):
                        new_record[key] = np.append(old_record[key], value)
                    elif isinstance(value, np.ndarray) and value.ndim == 1:
                        new_record[key] = np.append(old_record[key], value)
                    else:
                        new_record[key] = np.concatenate((old_record[key], value.squeeze()), axis=0)

            save(filename, new_record)
