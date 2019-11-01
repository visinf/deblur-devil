# Author: Jochen Gast <jochen.gast@visinf.tu-darmstadt.de>

import numpy as np
import torch
from torch import nn
from torch.utils import data


# ---------------------------------------------------------------------------------------------
# These are a couple of facacde objects abstracting away and combining some collective classes
# ---------------------------------------------------------------------------------------------


class ParameterSchedulerContainer(object):
    def __init__(self, schedulers):
        self.schedulers = schedulers

    def step(self, epoch=None):
        for scheduler in self.schedulers:
            scheduler.step(epoch=epoch)


class TracableModelAndLoss(nn.Module):
    def __init__(self, args, model_and_loss):
        super().__init__()
        self.args = args
        self.model_and_loss = model_and_loss

    def forward(self, example_dict):
        loss_dict = self.model_and_loss(example_dict)[0]
        return loss_dict[self.args.training_key]


class ModelAndLoss(nn.Module):
    def __init__(self, args, model, loss):
        super().__init__()
        self.args = args
        self.loss = loss
        self.model = model

    def num_parameters(self):
        return sum([p.numel() if p.requires_grad else 0 for p in self.parameters()])

    # ----------------------------------------------------------------------
    # NOTE: We merge inputs and targets into a single example dictionary.
    #       example_dict also contains meta infos about the given examples
    # ----------------------------------------------------------------------
    def forward(self, example_dict):
        # -------------------------------------------------------------------
        # Run forward pass to obtain model outputs.
        # Subsequently, computes losses and return resulting dictionaries
        # -------------------------------------------------------------------
        model_dict = self.model(example_dict)
        loss_dict = self.loss(model_dict, example_dict)
        return loss_dict, model_dict

    def as_tracable(self):
        return TracableModelAndLoss(self.args, self)


class CollateBatchesAndSamples:
    def __init__(self, args):
        self.args = args

    def __call__(self, example_dict):
        for key, value in example_dict.items():
            if isinstance(value, torch.Tensor) and value.dim() > 1:
                size = value.size()
                example_dict[key] = value.view(size[0] * size[1], *size[2:])
            elif isinstance(value, list):
                array = np.array(value)
                size = array.shape
                newshape = (size[0] * size[1], size[2:])
                example_dict[key] = np.reshape(array, *newshape)

        return example_dict


class _LoaderIter(object):
    def __init__(self, args, loader, collation=None):
        self.args = args
        self.collation = collation
        self.loader = loader
        self.it = loader.__iter__()
        self.tensor_keys = None

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        return self

    def __next__(self):
        example_dict = next(self.it)

        # -------------------------------------------------------------
        # Get input and target tensor keys
        # -------------------------------------------------------------
        if self.tensor_keys is None:
            input_keys = list(filter(lambda x: "input" in x, example_dict.keys()))
            target_keys = list(filter(lambda x: "target" in x, example_dict.keys()))
            self.tensor_keys = input_keys + target_keys

        # -------------------------------------------------------------
        # Possibly transfer to Cuda
        # -------------------------------------------------------------
        for key, value in example_dict.items():
            if key in self.tensor_keys:
                example_dict[key] = value.to(self.args.device)

        # -------------------------------------------------------------
        # Some Datasets produce multiple samples per example.
        # In this case, the first two dimensions correspond to batches
        # and samples per batch dimensions.
        # Hence, we collate the first two dimensions.
        # -------------------------------------------------------------
        if self.collation is not None:
            with torch.no_grad():
                example_dict = self.collation(example_dict)

        return example_dict


class LoaderAndCollation(object):
    def __init__(self, args, loader, collation=None):
        self.args = args
        self.collation = collation
        self.dataset = loader.dataset
        self.loader = loader

    def __iter__(self):
        return _LoaderIter(self.args, self.loader, self.collation)

    def __len__(self):
        return len(self.loader)

    # ---------------------------------------------
    # Slowly access the first time
    # ---------------------------------------------
    def first_item(self, device=None):
        loader = data.DataLoader(
            self.dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            drop_last=False)
        example_dict = loader.__iter__().next()
        if device is not None:
            input_keys = list(filter(lambda x: "input" in x, example_dict.keys()))
            target_keys = list(filter(lambda x: "target" in x, example_dict.keys()))
            tensor_keys = input_keys + target_keys
            for key, value in example_dict.items():
                if key in tensor_keys:
                    example_dict[key] = value.to(self.args.device)
        if self.collation is not None:
            with torch.no_grad():
                example_dict = self.collation(example_dict)
        return example_dict
