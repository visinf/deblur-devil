# Author: Jochen Gast <jochen.gast@visinf.tu-darmstadt.de>

import logging

import numpy as np
from torch import nn


def _2modules(mod_or_modules):
    if isinstance(mod_or_modules, nn.Module):
        return [mod_or_modules]
    else:
        return mod_or_modules


def identity_(mod_or_modules):
    modules = _2modules(mod_or_modules)
    for layer in modules:
        nn.init.constant_(layer.weight, 1)
        nn.init.constant_(layer.bias, 0)


def _check_uninitialized(indict, layer):
    layer_type = str(type(layer))
    layer_name = layer.__class__.__name__

    accepted_uninitialized_types = ["Sequential", "ModuleList"]

    if "torch" in layer_type:
        if not any([a in layer_type for a in accepted_uninitialized_types]):
            num_parameters = sum([p.numel() for p in layer.parameters()])
            if num_parameters > 0:
                if layer_name in indict.keys():
                    indict[layer_name] += num_parameters
                else:
                    indict[layer_name] = num_parameters

    return indict


def fanmax_(mod_or_modules):
    logging.value("Initializing FAN_MAX")
    modules = _2modules(mod_or_modules)

    uninitialized_modules = {}

    for layer in modules:
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d):
            m = layer.in_channels
            n = layer.out_channels
            k = layer.kernel_size
            stddev = np.sqrt(2.0 / (np.maximum(m, n)) * np.prod(k))
            nn.init.normal_(layer.weight, mean=0.0, std=stddev)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0.0)
        elif isinstance(layer, nn.BatchNorm2d):
            identity_(layer)
        else:
            uninitialized_modules = _check_uninitialized(uninitialized_modules, layer)

    for name, num_params in uninitialized_modules.items():
        logging.value("Found uninitialized layer of type '{}' [{} params]".format(name, num_params))


def msra_(mod_or_modules, mode='fan_out', nonlinearity='relu'):
    logging.value("Initializing MSRA")
    modules = _2modules(mod_or_modules)

    uninitialized_modules = {}

    for layer in modules:
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(layer.weight, mode=mode, nonlinearity=nonlinearity)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0.0)
        elif isinstance(layer, nn.BatchNorm2d):
            identity_(layer)
        else:
            uninitialized_modules = _check_uninitialized(uninitialized_modules, layer)

    for name, num_params in uninitialized_modules.items():
        logging.value("Found uninitialized layer of type '{}' [{} params]".format(name, num_params))

# def initialize_weights(modules, init_type='dncnn', sigmoid=False):
#     for layer in modules:
#         if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d):
#             if init_type == 'dncnn':
#                 m = layer.in_channels
#                 n = layer.out_channels
#                 k = layer.kernel_size
#                 stddev = np.sqrt(2.0 / (np.maximum(m, n)) * np.prod(k))
#                 nn.init.normal_(layer.weight, mean=0.0, std=stddev)
#             elif init_type == 'normal':
#                 stddev = 0.04
#                 nn.init.normal_(layer.weight, mean=0.0, std=stddev)
#             elif init_type == 'msra_fan_in':
#                 nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
#             elif init_type == 'msra_fan_out':
#                 nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
#             else:
#                 raise ValueError("Unknown init_type {}".format(init_type))
#
#             if layer.bias is not None:
#                 nn.init.constant_(layer.bias, 0.0)
#
#         if isinstance(layer, nn.BatchNorm2d):
#             if sigmoid:
#                 nn.init.constant_(layer.weight, 0.005)
#             else:
#                 nn.init.constant_(layer.weight, 1)
#             nn.init.constant_(layer.bias, 0)
