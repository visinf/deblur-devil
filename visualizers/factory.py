import inspect

from utils import factories

__registry = {}


def register(name, module_class):
    factories.try_register(name, module_class, __registry, inspect.stack()[1])


def get_dict():
    return __registry
