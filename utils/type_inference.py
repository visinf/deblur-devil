# Author: Jochen Gast <jochen.gast@visinf.tu-darmstadt.de>

import inspect

from . import strings
from .strings import filter_list_of_strings


# Some of these functions are adapted from: https://github.com/NVIDIA/flownet2-pytorch/blob/master/utils/tools.py

# -------------------------------------------------------------------------------------------------
# Looks for sub arguments in the argument structure.
# Retrieve sub arguments for modules such as optimizer_*
# -------------------------------------------------------------------------------------------------
def kwargs_from_args(args, name, exclude=[]):
    if isinstance(exclude, str):
        exclude = [exclude]
    exclude += ["class"]
    args_dict = vars(args)
    name += "_"
    subargs_dict = {
        key[len(name):]: value for key, value in args_dict.items()
        if name in key and all([key != name + x for x in exclude])
    }
    return subargs_dict


def module_classes_to_dict(module, include_classes="*", exclude_classes=()):
    # -------------------------------------------------------------------------
    # If arguments are strings, convert them to a list
    # -------------------------------------------------------------------------
    if include_classes is not None:
        if isinstance(include_classes, str):
            include_classes = [include_classes]

    if exclude_classes is not None:
        if isinstance(exclude_classes, str):
            exclude_classes = [exclude_classes]

    # -------------------------------------------------------------------------
    # Obtain dictionary from given module
    # -------------------------------------------------------------------------
    item_dict = dict([(name, getattr(module, name)) for name in dir(module)])

    # -------------------------------------------------------------------------
    # Filter classes
    # -------------------------------------------------------------------------
    item_dict = dict([
        (name, value) for name, value in item_dict.items() if inspect.isclass(getattr(module, name))
    ])

    filtered_keys = filter_list_of_strings(
        item_dict.keys(), include=include_classes, exclude=exclude_classes)

    # -------------------------------------------------------------------------
    # Construct dictionary from matched results
    # -------------------------------------------------------------------------
    result_dict = dict([(name, value) for name, value in item_dict.items() if name in filtered_keys])

    return result_dict


# -------------------------------------------------------------------------------------------------
# Create class instance from kwargs dictionary.
# Filters out keys that not in the constructor
# -------------------------------------------------------------------------------------------------
def instance_from_kwargs(class_constructor, args=(), kwargs={}):
    argspec = inspect.getargspec(class_constructor.__init__)
    full_args = argspec.args
    filtered_args = dict([(k, v) for k, v in kwargs.items() if k in full_args])
    instance = class_constructor(*args, **filtered_args)
    return instance


def instance_from_kwargs_with_forced_types(class_constructor,
                                           args=(),
                                           kwargs={},
                                           forced_default_types=(),  # set types for known arguments
                                           unknown_default_types=()):  # set types for unknown arguments

    # -------------------------------------------------------------------------
    # Gets around the issue of mutable default arguments
    # -------------------------------------------------------------------------
    forced_default_types = dict(forced_default_types)
    unknown_default_types = dict(unknown_default_types)

    # -------------------------------------------------------------------------
    # Determine constructor argument names and defaults
    # -------------------------------------------------------------------------
    try:
        argspec = inspect.getargspec(class_constructor.__init__)
        argspec_defaults = argspec.defaults if argspec.defaults is not None else []
        default_args_dict = dict(zip(argspec.args[-len(argspec_defaults):], argspec_defaults))
    except TypeError:
        print(argspec)
        print(argspec.defaults)
        raise ValueError("unknown_default_types should be adjusted for module")

    def _get_type_from_arg(arg):
        if isinstance(arg, bool):
            return strings.as_bool_or_none
        else:
            return type(arg)

    kwargs_with_correct_types = kwargs

    # -------------------------------------------------------------------------
    # Add sub_arguments
    # -------------------------------------------------------------------------
    for argname, argvalue in kwargs_with_correct_types.items():
        # ---------------------------------------------------------------------
        # If a default parameter can be inferred from the module, pick that one
        # ---------------------------------------------------------------------
        if argname in default_args_dict.keys():
            # -----------------------------------------------------------------
            # Check for forced default types
            # -----------------------------------------------------------------
            if argname in forced_default_types.keys():
                argtype = forced_default_types[argname]
            else:
                argtype = _get_type_from_arg(default_args_dict[argname])
            kwargs_with_correct_types[argname] = argtype(argvalue)

        # ---------------------------------------------------------------------
        # Take from the unkowns list
        # ---------------------------------------------------------------------
        elif argname in unknown_default_types.keys():
            argtype = unknown_default_types[argname]
            kwargs_with_correct_types[argname] = argtype(argvalue)

    return instance_from_kwargs(class_constructor, args=args, kwargs=kwargs_with_correct_types)
