# Author: Jochen Gast <jochen.gast@visinf.tu-darmstadt.de>

import torch

from optim import factory
from utils import factories
from utils import type_inference as typeinf


def init():
    factories.import_submodules(__name__)

    # ------------------------------------------------------------------------------------
    # Add PyTorch's own optimizer classes
    # ------------------------------------------------------------------------------------
    _optimizer_classes = typeinf.module_classes_to_dict(torch.optim, exclude_classes="Optimizer")
    for name, constructor in _optimizer_classes.items():
        factory.register(name, constructor)
