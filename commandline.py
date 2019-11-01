# Author: Jochen Gast <jochen.gast@visinf.tu-darmstadt.de>

import argparse
import inspect
import logging
import os

import sys
import torch

import constants
from augmentations import factory as augmentations_factory
from datasets import factory as dataset_factory
from holistic_records import factory as records_factory
from losses import factory as loss_factory
from models import factory as model_factory
from optim import factory as optim_factory
from utils import json
from utils import strings
from utils import type_inference as typeinf
from visualizers import factory as visualizer_factory
from utils.help_formatter import LongChoicesFormatter


def _add_arguments_for_module(parser,
                              module_or_dict,
                              name,
                              default_class,
                              add_class_argument=True,  # adds class choice as argument
                              include_classes="*",
                              exclude_classes=(),
                              exclude_params=("self", "args"),
                              param_defaults=(),  # overwrite any default param
                              forced_default_types=(),  # set types for known arguments
                              unknown_default_types=()):  # set types for unknown arguments

    # -------------------------------------------------------------------------
    # Gets around the issue of mutable default arguments
    # -------------------------------------------------------------------------
    exclude_params = list(exclude_params)
    param_defaults = dict(param_defaults)
    forced_default_types = dict(forced_default_types)
    unknown_default_types = dict(unknown_default_types)

    # -------------------------------------------------------------------------
    # Determine possible choices from class names in module,
    # possibly apply include/exclude filters
    # -------------------------------------------------------------------------
    if isinstance(module_or_dict, dict):
        module_dict = module_or_dict
    else:
        module_dict = typeinf.module_classes_to_dict(
            module_or_dict,
            include_classes=include_classes,
            exclude_classes=exclude_classes)

    # -------------------------------------------------------------------------
    # Parse known arguments to determine choice for argument name
    # -------------------------------------------------------------------------
    if add_class_argument:
        parser.add_argument(
            "--%s" % name, type=str, default=default_class, choices=module_dict.keys())
        known_args = parser.parse_known_args(sys.argv[1:])[0]
    else:
        # build a temporary parser, and do not add the class as argument
        tmp_parser = argparse.ArgumentParser()
        tmp_parser.add_argument(
            "--%s" % name, type=str, default=default_class, choices=module_dict.keys())
        known_args = tmp_parser.parse_known_args(sys.argv[1:])[0]

    # these could be multiples, if 'append' is allowed for this argument
    class_name = vars(known_args)[name]

    # -------------------------------------------------------------------------
    # If classes are None, there is no point in trying to parse further arguments
    # -------------------------------------------------------------------------
    if class_name is None:
        return

    # -------------------------------------------------------------------------
    # Get constructor of that argument choice. For 'append' options, we will
    # construct by means of the last provided class name.
    # -------------------------------------------------------------------------
    class_constructor = module_dict[class_name]

    # -------------------------------------------------------------------------
    # Determine constructor argument names and defaults
    # -------------------------------------------------------------------------
    argspec = None
    try:
        argspec = inspect.getfullargspec(class_constructor.__init__)
        argspec_defaults = argspec.defaults if argspec.defaults is not None else []
        full_args = argspec.args
        default_args_dict = dict(zip(argspec.args[-len(argspec_defaults):], argspec_defaults))
    except TypeError:
        print(argspec)
        print(argspec.defaults)
        raise ValueError("unknown_default_types should be adjusted for module: '%s.py'" % name)

    def _get_type_from_arg(arg):
        if isinstance(arg, bool):
            return strings.as_bool_or_none
        else:
            return type(arg)

    # -------------------------------------------------------------------------
    # Add sub_arguments
    # -------------------------------------------------------------------------
    for argname in full_args:

        # ---------------------------------------------------------------------
        # Skip
        # ---------------------------------------------------------------------
        if argname in exclude_params:
            continue

        # ---------------------------------------------------------------------
        # Sub argument name
        # ---------------------------------------------------------------------
        sub_arg_name = "%s_%s" % (name, argname)

        # ---------------------------------------------------------------------
        # If a default argument is given, take that one
        # ---------------------------------------------------------------------
        if argname in param_defaults.keys():
            parser.add_argument(
                "--%s" % sub_arg_name,
                type=_get_type_from_arg(param_defaults[argname]),
                default=param_defaults[argname])

        # ---------------------------------------------------------------------
        # If a default parameter can be inferred from the module, pick that one
        # ---------------------------------------------------------------------
        elif argname in default_args_dict.keys():

            # -----------------------------------------------------------------
            # Check for forced default types
            # -----------------------------------------------------------------
            if argname in forced_default_types.keys():
                argtype = forced_default_types[argname]
            else:
                argtype = _get_type_from_arg(default_args_dict[argname])
            parser.add_argument(
                "--%s" % sub_arg_name, type=argtype, default=default_args_dict[argname])

        # ---------------------------------------------------------------------
        # Take from the unkowns list
        # ---------------------------------------------------------------------
        elif argname in unknown_default_types.keys():
            parser.add_argument("--%s" % sub_arg_name, type=unknown_default_types[argname])

        else:
            raise ValueError(
                "Do not know how to handle argument '%s' for class '%s'" % (argname, name))


def _add_special_arguments(parser):
    # -------------------------------------------------------------------------
    # Known arguments so far
    # -------------------------------------------------------------------------
    known_args = vars(parser.parse_known_args(sys.argv[1:])[0])

    # -------------------------------------------------------------------------
    # Add special arguments for training
    # -------------------------------------------------------------------------
    loss = known_args["loss"]
    if loss is not None:
        parser.add_argument("--training_key", type=str, default="total_loss")

    # -------------------------------------------------------------------------
    # Add special arguments for validation
    # -------------------------------------------------------------------------
    validation_dataset = known_args['validation_dataset']
    if validation_dataset is not None:
        parser.add_argument(
            "--validation_batch_size", type=int, default=-1)
        parser.add_argument(
            "--validation_keys", type=strings.as_stringlist_or_none, default="[total_loss]")
        parser.add_argument(
            "--validation_modes", type=strings.as_stringlist_or_none, default="[min]")

    # -------------------------------------------------------------------------
    # Add special arguments for checkpoints
    # -------------------------------------------------------------------------
    checkpoint = known_args["checkpoint"]
    if checkpoint is not None:
        parser.add_argument(
            "--checkpoint_mode", type=str, default="latest", choices=["latest", "best"])
        parser.add_argument(
            "--checkpoint_include_params", type=strings.as_stringlist_or_none, default=['*'])
        parser.add_argument(
            "--checkpoint_exclude_params", type=strings.as_stringlist_or_none, default=[])
        parser.add_argument(
            "--checkpoint_translations", type=strings.as_dict_or_none, default={})
        parser.add_argument(
            "--checkpoint_fuzzy_translation_keys", type=strings.as_stringlist_or_none, default=[])

    # -------------------------------------------------------------------------
    # Add special arguments for optimizer groups
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--optimizer_group", action="append", type=strings.as_dict_or_none, default=None)

    # -------------------------------------------------------------------------
    # Add special argument for parameter scheduling groups
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--param_scheduler_group", action="append", type=strings.as_dict_or_none, default=None)

    # -------------------------------------------------------------------------
    # Thought about it: general filtering of all parmeters in the optimizer
    # But then again: You can do this in the model class as well as using the
    # parameter groups
    # -------------------------------------------------------------------------
    # parser.add_argument(
    #     "--optimizer_include_params", type=strings.as_stringlist_or_none, default=['*'])
    # parser.add_argument(
    #     "--optimizer_exclude_params", type=strings.as_stringlist_or_none, default=[])


# class __CustomHelpFormatter(argparse.HelpFormatter):
#     def _metavar_formatter(self, action, default_metavar):
#         if action.metavar is not None:
#             result = action.metavar
#         elif action.choices is not None:
#             choice_strs = [str(choice) for choice in sorted(action.choices)]
#             result = '{%s}' % ', '.join(choice_strs)
#             # print(type(result))
#             # result = '{ ...'
#             # for choice in choice_strs:
#             #     result += '\n'
#             #     result += ' '*5
#             #     result += choice
#             # result += '  }'

#         else:
#             result = default_metavar

#         def format(tuple_size):
#             if isinstance(result, tuple):
#                 return result
#             else:
#                 return (result, ) * tuple_size

#         return format


def _parse_arguments():
    # -------------------------------------------------------------------------
    # Argument parser and shortcut function to add arguments
    # -------------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="Deep Visual Inference.",
        formatter_class=LongChoicesFormatter)
    add = parser.add_argument

    # -------------------------------------------------------------------------
    # Standard arguments
    # -------------------------------------------------------------------------
    add('--batch_size', type=int, default=1)
    add('--checkpoint', type=strings.as_string_or_none, default=None)
    add('--clip_grad', type=float, default=0)
    add('--cuda', type=strings.as_bool_or_none, default=True)
    add('--logging_loss_graph', type=strings.as_bool_or_none, default=False)
    add('--logging_model_graph', type=strings.as_bool_or_none, default=False)
    add('--loshchilov_weight_decay', type=float, default=0)
    add('--proctitle', type=str, default='__random__')
    add('--save', type=str, default=None)
    add('--seed', type=strings.as_int_or_none, default=0)
    add('--start_epoch', type=int, default=1)
    add('--total_epochs', type=int, default=10)
    add('--prefix', type=str, default='')

    # -------------------------------------------------------------------------
    # Arguments inferred from losses
    # -------------------------------------------------------------------------
    loss_dict = loss_factory.get_dict()
    _add_arguments_for_module(
        parser,
        loss_dict,
        name="loss",
        default_class=None,
        exclude_classes=["_*", "Variable"],
        exclude_params=["self", "args"])

    # -------------------------------------------------------------------------
    # Arguments inferred from models
    # -------------------------------------------------------------------------
    model_dict = model_factory.get_dict()
    _add_arguments_for_module(
        parser,
        model_dict,
        name="model",
        default_class="FlowNet1S",
        exclude_classes=["_*", "Variable"],
        exclude_params=["self", "args"])

    # -------------------------------------------------------------------------
    # Arguments inferred from augmentations for training
    # -------------------------------------------------------------------------
    augmentations_dict = augmentations_factory.get_dict()
    _add_arguments_for_module(
        parser,
        augmentations_dict,
        name="training_augmentation",
        default_class=None,
        exclude_classes=["_*"],
        exclude_params=["self", "args"],
        forced_default_types={"crop": strings.as_intlist_or_none})

    # -------------------------------------------------------------------------
    # Arguments inferred from augmentations for validation
    # -------------------------------------------------------------------------
    _add_arguments_for_module(
        parser,
        augmentations_dict,
        name="validation_augmentation",
        default_class=None,
        exclude_classes=["_*"],
        exclude_params=["self", "args"])

    # -------------------------------------------------------------------------
    # Arguments inferred from datasets for training
    # -------------------------------------------------------------------------
    dataset_dict = dataset_factory.get_dict()
    _add_arguments_for_module(
        parser,
        dataset_dict,
        name="training_dataset",
        default_class=None,
        exclude_params=["self", "args", "is_cropped"],
        exclude_classes=["_*"],
        unknown_default_types={"root": str},
        forced_default_types={"photometric_augmentations": strings.as_dict_or_none,
                              "affine_augmentations": strings.as_dict_or_none,
                              "random_crop": strings.as_intlist_or_none})

    # -------------------------------------------------------------------------
    # Arguments inferred from datasets for validation
    # -------------------------------------------------------------------------
    _add_arguments_for_module(
        parser,
        dataset_dict,
        name="validation_dataset",
        default_class=None,
        exclude_params=["self", "args", "is_cropped"],
        exclude_classes=["_*"],
        unknown_default_types={"root": str},
        forced_default_types={"photometric_augmentations": strings.as_dict_or_none,
                              "affine_augmentations": strings.as_dict_or_none,
                              "random_crop": strings.as_intlist_or_none})

    # -------------------------------------------------------------------------
    # Arguments inferred from PyTorch optimizers
    # -------------------------------------------------------------------------
    optim_dict = optim_factory.get_dict()
    _add_arguments_for_module(
        parser,
        optim_dict,
        name="optimizer",
        default_class=None,
        exclude_classes=["_*", "Optimizer", "constructor"],
        exclude_params=["self", "args", "params"],
        forced_default_types={"lr": float,
                              "momentum": float,
                              "betas": strings.as_floatlist_or_none,
                              "dampening": float,
                              "weight_decay": float,
                              "nesterov": strings.as_bool_or_none})

    # -------------------------------------------------------------------------
    # Arguments inferred from PyTorch lr schedulers
    # -------------------------------------------------------------------------
    _add_arguments_for_module(
        parser,
        torch.optim.lr_scheduler,
        name="lr_scheduler",
        default_class=None,
        exclude_classes=["_*", "constructor"],
        exclude_params=["self", "args", "optimizer"],
        forced_default_types={"min_lr": float, "eta_min": float},
        unknown_default_types={"T_max": int,
                               "lr_lambda": str,
                               "step_size": int,
                               "milestones": strings.as_intlist_or_none,
                               "gamma": float})

    # -------------------------------------------------------------------------
    # Arguments inferred from holistic records
    # -------------------------------------------------------------------------
    records_dict = records_factory.get_dict()
    _add_arguments_for_module(
        parser,
        records_dict,
        default_class="EpochRecorder",
        name="holistic_records",
        add_class_argument=False,
        exclude_classes=["_*"],
        exclude_params=["self", "args", "root", "epoch", "dataset"])

    # -------------------------------------------------------------------------
    # Arguments inferred from visualizer factory
    # -------------------------------------------------------------------------
    visualizer_dict = visualizer_factory.get_dict()
    _add_arguments_for_module(
        parser,
        visualizer_dict,
        name="visualizer",
        default_class=None,
        exclude_classes=["_*"],
        exclude_params=['self', 'args',
                        'model_and_loss',
                        'optimizer',
                        'param_scheduler',
                        'lr_scheduler',
                        'train_loader',
                        'validation_loader'])

    # -------------------------------------------------------------------------
    # Special arguments
    # -------------------------------------------------------------------------
    _add_special_arguments(parser)

    # -------------------------------------------------------------------------
    # Parse arguments
    # -------------------------------------------------------------------------
    args = None
    try:
        args = parser.parse_args()
    except argparse.ArgumentError as e:
        parser.error(str(e))

    # -------------------------------------------------------------------------
    # Parse default arguments from a dummy commandline not specifying any args
    # -------------------------------------------------------------------------
    defaults = vars(parser.parse_known_args(['--dummy'])[0])

    # -------------------------------------------------------------------------
    # Consistency checks
    # -------------------------------------------------------------------------
    args.cuda = args.cuda and torch.cuda.is_available()

    # check validation modes for allowed values
    if args.validation_dataset is not None:
        for vmode in args.validation_modes:
            if vmode not in ['min', 'max']:
                raise ValueError("Validation mode must be 'min' or 'max' (Was '{}').".format(vmode))
        if len(args.validation_keys) != len(args.validation_modes):
            raise ValueError("Inconsistent number of validation keys and modes (Were {} and {})".format(
                len(args.validation_keys), len(args.validation_modes)))

    return args, defaults


def postprocess_args(args):
    # ----------------------------------------------------------------------------
    # Get appropriate class constructors from modules
    # ----------------------------------------------------------------------------
    args.model_class = model_factory.get_dict()[args.model]

    if args.optimizer is not None:
        args.optimizer_class = optim_factory.get_dict()[args.optimizer]

    if args.loss is not None:
        args.loss_class = loss_factory.get_dict()[args.loss]

    if args.lr_scheduler is not None:
        scheduler_classes = typeinf.module_classes_to_dict(torch.optim.lr_scheduler)
        args.lr_scheduler_class = scheduler_classes[args.lr_scheduler]

    if args.training_dataset is not None:
        args.training_dataset_class = \
            dataset_factory.get_dict()[args.training_dataset]

    if args.validation_dataset is not None:
        args.validation_dataset_class = \
            dataset_factory.get_dict()[args.validation_dataset]

    if args.training_augmentation is not None:
        args.training_augmentation_class = \
            augmentations_factory.get_dict()[args.training_augmentation]

    if args.validation_augmentation is not None:
        args.validation_augmentation_class = \
            augmentations_factory.get_dict()[args.validation_augmentation]

    if args.visualizer is not None:
        args.visualizer_class = visualizer_factory.get_dict()[args.visualizer]

    # ----------------------------------------------------------------------------
    # holistic records
    # ----------------------------------------------------------------------------
    holistic_records_args = typeinf.kwargs_from_args(args, "holistic_records")
    setattr(args, "holistic_records_kwargs", holistic_records_args)

    return args


def parse_save_dir():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', type=str, default=None)
    known_args = vars(parser.parse_known_args(sys.argv[1:])[0])
    if 'save' not in known_args.keys():
        raise ValueError("'save' argument must be specified in commandline!")
    return known_args['save']


def parse_arguments(blocktitle):
    # ----------------------------------------------------------------------------
    # Get parse commandline and default arguments
    # ----------------------------------------------------------------------------
    args, defaults = _parse_arguments()

    # ----------------------------------------------------------------------------
    # Write arguments to file, as json and txt
    # ----------------------------------------------------------------------------
    json.write_dictionary_to_file(
        vars(args), filename=os.path.join(args.save, "args.json"), sortkeys=True)
    json.write_dictionary_to_file(
        vars(args), filename=os.path.join(args.save, "args.txt"), sortkeys=True)

    # ----------------------------------------------------------------------------
    # Log arguments
    # ----------------------------------------------------------------------------
    non_default_args = []
    with logging.block(blocktitle, emph=True):
        for argument, value in sorted(vars(args).items()):
            reset = constants.COLOR_RESET
            if value == defaults[argument]:
                color = reset
            else:
                non_default_args.append((argument, value))
                color = constants.COLOR_NON_DEFAULT_ARGUMENT
            if isinstance(value, dict):
                dict_string = strings.dict_as_string(value)
                logging.info("{}{}: {}{}".format(color, argument, dict_string, reset))
            else:
                logging.info("{}{}: {}{}".format(color, argument, value, reset))

    # ----------------------------------------------------------------------------
    # Remember non defaults
    # ----------------------------------------------------------------------------
    args.non_default_args = dict((pair[0], pair[1]) for pair in non_default_args)

    # ----------------------------------------------------------------------------
    # Postprocess
    # ----------------------------------------------------------------------------
    args = postprocess_args(args)

    return args
