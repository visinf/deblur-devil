# Author: Jochen Gast <jochen.gast@visinf.tu-darmstadt.de>

import logging
import os
import shutil

import torch
from torch import nn

from utils import json
from utils import strings
from utils import system


def load_state_dict_into_module(state_dict, module, strict=True):
    own_state = module.state_dict()
    for name, param in state_dict.items():
        if name in own_state:
            if isinstance(param, nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            try:
                own_state[name].resize_as_(param)
                own_state[name].copy_(param)
            except Exception:
                raise RuntimeError('While copying the parameter named {}, '
                                   'whose dimensions in the model are {} and '
                                   'whose dimensions in the checkpoint are {}.'
                                   .format(name, own_state[name].size(), param.size()))
        elif strict:
            logging.info('Unexpected key "{}" in state_dict'.format(name))
            raise KeyError('')
    if strict:
        missing = set(own_state.keys()) - set(state_dict.keys())
        if len(missing) > 0:
            logging.info('Missing keys in state_dict: ')
            logging.value("{}".format(missing))
            raise KeyError('')


def restore_module_from_state_dict(module,
                                   checkpoint_state_dict,
                                   include_params='*',
                                   exclude_params=(),
                                   translations=(),
                                   fuzzy_translation_keys=()):
    include_params = list(include_params)
    exclude_params = list(exclude_params)
    fuzzy_translation_keys = list(fuzzy_translation_keys)
    translations = dict(translations)

    # This list is only for book keeping
    actual_translations = []

    # Fuzzy matching is just a lazy way to specify translations:
    # A given match will get searcher for and added to the manual translations,
    # if not already existing
    for checkpoint_key in checkpoint_state_dict.keys():
        for module_key in module.state_dict().keys():
            for fuzzy_key in fuzzy_translation_keys:
                if fuzzy_key in checkpoint_key and fuzzy_key in module_key:
                    new_translation_key = checkpoint_key.replace(fuzzy_key, '')
                    new_translation_value = module_key.replace(fuzzy_key, '')
                    if new_translation_key not in translations.keys():
                        translations[new_translation_key] = new_translation_value

    def _translate(key_in):
        key_out = key_in
        # first check fuzz matching
        for k, v in translations.items():
            key_out = key_out.replace(k, v)
        if key_out != key_in:
            actual_translations.append((key_in, key_out))
        return key_out

    # -----------------------------------------------------------------------------------------
    # translate keys in checkpoint_state_dict according to translations
    # -----------------------------------------------------------------------------------------
    checkpoint_state_dict = {_translate(key): value for key, value in checkpoint_state_dict.items()}

    # -----------------------------------------------------------------------------------------
    # Filter afterwards
    # -----------------------------------------------------------------------------------------
    restore_keys = strings.filter_list_of_strings(
        checkpoint_state_dict.keys(),
        include=include_params,
        exclude=exclude_params)
    checkpoint_state_dict = {key: value for key, value in checkpoint_state_dict.items() if key in restore_keys}

    # if parameter lists are given, don't be strict with loading from checkpoints
    default_parameters_given = \
        len(include_params) == 1 and include_params[0] == '*' and len(exclude_params) == 0

    strict = True
    if not default_parameters_given:
        strict = False

    load_state_dict_into_module(checkpoint_state_dict, module, strict=strict)

    return restore_keys, actual_translations


def restore_module_from_filename(module,
                                 filename,
                                 key='state_dict',
                                 include_params='*',
                                 exclude_params=(),
                                 translations=(),
                                 fuzzy_translation_keys=()):
    include_params = list(include_params)
    exclude_params = list(exclude_params)
    fuzzy_translation_keys = list(fuzzy_translation_keys)
    translations = dict(translations)

    # ------------------------------------------------------------------------------
    # Make sure file exists
    # ------------------------------------------------------------------------------
    if not os.path.isfile(filename):
        logging.info("Could not find checkpoint file '%s'!" % filename)
        quit()

    # ------------------------------------------------------------------------------
    # Load checkpoint from file including the state_dict
    # ------------------------------------------------------------------------------
    cpu_device = torch.device('cpu')
    checkpoint_dict = torch.load(filename, map_location=cpu_device)
    checkpoint_state_dict = checkpoint_dict[key]

    try:
        restore_keys, actual_translations = restore_module_from_state_dict(
            module,
            checkpoint_state_dict,
            include_params=include_params,
            exclude_params=exclude_params,
            translations=translations,
            fuzzy_translation_keys=fuzzy_translation_keys)
    except KeyError:
        with logging.block('Checkpoint keys:'):
            logging.value(checkpoint_state_dict.keys())
        with logging.block('Module keys:'):
            logging.value(module.state_dict().keys())
        logging.info("Could not load checkpoint because of key errors. Checkpoint translations gone wrong?")
        quit()

    return checkpoint_dict, restore_keys, actual_translations


# --------------------------------------------------------------------------
# Checkpoint loader/saver.
# --------------------------------------------------------------------------
class CheckpointSaver:
    def __init__(self,
                 prefix="checkpoint",
                 latest_postfix="_latest_",
                 best_postfix="_best_",
                 model_key="state_dict",
                 extension=".ckpt"):

        self.prefix = prefix
        self.model_key = model_key
        self.latest_postfix = latest_postfix
        self.best_postfix = best_postfix
        self.extension = extension

    def restore(self,
                filename,
                model_and_loss,
                include_params='*',
                exclude_params=(),
                translations={},
                fuzzy_translation_keys=()):

        # -----------------------------------------------------------------------------------------
        # Make sure file exists
        # -----------------------------------------------------------------------------------------
        if not os.path.isfile(filename):
            logging.info("Could not find checkpoint file '%s'!" % filename)
            quit()

        checkpoint_dict, restore_keys, actual_translations = \
            restore_module_from_filename(model_and_loss,
                                         filename,
                                         key=self.model_key,
                                         include_params=include_params,
                                         exclude_params=exclude_params,
                                         translations=translations,
                                         fuzzy_translation_keys=fuzzy_translation_keys)

        if len(actual_translations) > 0:
            logging.info("  Translations:")
            for pair in actual_translations:
                logging.info("    %s  =>  %s" % pair)

        logging.info("  Restore keys:")
        for key in restore_keys:
            logging.info("    %s" % key)

        # -----------------------------------------------------------------------------------------
        # Get checkpoint statistics without the state dict
        # -----------------------------------------------------------------------------------------
        checkpoint_stats = {
            key: value for key, value in checkpoint_dict.items() if key != self.model_key
        }

        return checkpoint_stats, filename

        # # -----------------------------------------------------------------------------------------
        # # Load checkpoint from file including the state_dict
        # # -----------------------------------------------------------------------------------------
        # cpu_device = torch.device('cpu')
        # checkpoint_with_state = torch.load(filename, map_location=cpu_device)

        # # -----------------------------------------------------------------------------------------
        # # Load filtered state dictionary
        # # -----------------------------------------------------------------------------------------
        # state_dict = checkpoint_with_state[self.model_key]
        # restore_keys = strings.filter_list_of_strings(
        #     state_dict.keys(),
        #     include=include_params,
        #     exclude=exclude_params)

        # state_dict = {key: value for key, value in state_dict.items() if key in restore_keys}

        # # if parameter lists are given, don't be strict with loading from checkpoints
        # strict = True
        # if include_params != "*" or len(exclude_params) != 0:
        #     strict = False

        # checkpoints.load_state_dict_into_module(state_dict, model_and_loss, strict=strict)
        # logging.info("  Restore keys:")
        # for key in restore_keys:
        #     logging.info("    %s" % key)

        # # -----------------------------------------------------------------------------------------
        # Get checkpoint statistics without the state dict
        # -----------------------------------------------------------------------------------------
        # checkpoint_stats = {
        #     key: value for key, value in checkpoint_with_state.items() if key != self._model_key
        # }
        # return checkpoint_stats, filename

    def restore_latest(self,
                       directory,
                       model_and_loss,
                       include_params='*',
                       exclude_params=(),
                       translations={},
                       fuzzy_translation_keys=()):

        latest_checkpoint_filename = os.path.join(
            directory, self.prefix + self.latest_postfix + self.extension)

        return self.restore(
            latest_checkpoint_filename,
            model_and_loss=model_and_loss,
            include_params=include_params,
            exclude_params=exclude_params,
            translations=translations,
            fuzzy_translation_keys=fuzzy_translation_keys)

    def restore_best(self,
                     directory,
                     model_and_loss,
                     include_params='*',
                     exclude_params=(),
                     translations={},
                     fuzzy_translation_keys=()):

        best_checkpoint_filename = os.path.join(
            directory, self.prefix + self.best_postfix + self.extension)

        return self.restore(
            best_checkpoint_filename,
            model_and_loss=model_and_loss,
            include_params=include_params,
            exclude_params=exclude_params,
            translations=translations,
            fuzzy_translation_keys=fuzzy_translation_keys)

    def save_latest(self, directory, model_and_loss, stats_dict,
                    store_as_best=False, store_prefixes='total_loss'):

        # -----------------------------------------------------------------------------------------
        # Mutable default args..
        # -----------------------------------------------------------------------------------------
        store_as_best = list(store_as_best)

        # -----------------------------------------------------------------------------------------
        # Make sure directory exists
        # -----------------------------------------------------------------------------------------
        system.ensure_dir(directory)

        # -----------------------------------------------------------------------------------------
        # Save
        # -----------------------------------------------------------------------------------------
        save_dict = dict(stats_dict)
        save_dict[self.model_key] = model_and_loss.state_dict()
        latest_checkpoint_filename = os.path.join(
            directory, self.prefix + self.latest_postfix + self.extension)
        latest_statistics_filename = os.path.join(
            directory, self.prefix + self.latest_postfix + '.json')

        # convert to cpu tensors before writing the checkpoint files
        cpu_device = torch.device('cpu')

        def map_dict_to_cpu(x):
            if torch.is_tensor(x):
                return x.to(cpu_device)
            elif isinstance(x, dict):
                return dict((key, map_dict_to_cpu(value)) for key, value in x.items())
            else:
                return x

        save_dict = map_dict_to_cpu(save_dict)
        torch.save(save_dict, latest_checkpoint_filename)
        json.write_dictionary_to_file(stats_dict, filename=latest_statistics_filename)

        # -----------------------------------------------------------------------------------------
        # Possibly store as best
        # -----------------------------------------------------------------------------------------
        for store, prefix in zip(store_as_best, store_prefixes):
            if store:
                best_checkpoint_filename = os.path.join(
                    directory, self.prefix + self.best_postfix + prefix + self.extension)

                best_statistics_filename = os.path.join(
                    directory, self.prefix + self.best_postfix + prefix + ".json")

                shortfile = best_checkpoint_filename.rsplit("/", 1)[1]
                shortpath = os.path.dirname(best_checkpoint_filename).rsplit("/", 1)[1]
                shortname = os.path.join(shortpath, shortfile)
                logging.info("Save ckpt to ../%s" % shortname)
                shutil.copyfile(latest_checkpoint_filename, best_checkpoint_filename)
                shutil.copyfile(latest_statistics_filename, best_statistics_filename)
