# Author: Jochen Gast <jochen.gast@visinf.tu-darmstadt.de>

import logging
import importlib
import logging
import os
from utils import system


def try_register(name, module_class, registry, calling_frame):
    if name in registry:
        block_info = "Warning in {}[{}]:".format(calling_frame.filename, calling_frame.lineno)
        with logging.block(block_info):
            code_info = "{} yields duplicate factory entry!".format(calling_frame.code_context[0][0:-1])
            logging.value(code_info)

    registry[name] = module_class


def _package_contents(package_name):
    module_filenames = system.get_filenames(package_name, "*.py")
    module_filenames = [mod[len(package_name)+1:] for mod in module_filenames]
    module_filenames = [os.path.splitext(mod)[0] for mod in module_filenames]
    module_filenames = list(filter(lambda x: ('__init__' not in x), module_filenames))
    module_filenames = [x.replace('/', '.') for x in module_filenames]
    return set(module_filenames)


def import_submodules(package_name):
    with logging.block(package_name + '...'):
        content = _package_contents(package_name)
        for name in content:
            if name != "__init__":
                import_target = "%s.%s" % (package_name, name)
                try:
                    __import__(import_target)
                except Exception as err:
                    logging.info("ImportError in {}: {}".format(import_target, str(err)))
