# Author: Jochen Gast <jochen.gast@visinf.tu-darmstadt.de>

import fnmatch
import logging
import os
import random

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader

import constants
import facade
from utils import checkpoints
from utils import proctitles
from utils import strings
from utils import type_inference as typeinf


def configure_runtime_augmentations(args):
    with logging.block("Runtime Augmentations", emph=True):

        training_augmentation = None
        validation_augmentation = None

        # ----------------------------------------------------
        # Training Augmentation
        # ----------------------------------------------------
        if args.training_augmentation is not None:
            kwargs = typeinf.kwargs_from_args(args, "training_augmentation")
            logging.value("training_augmentation: ", args.training_augmentation)
            with logging.block():
                logging.values(kwargs)
            kwargs["args"] = args
            training_augmentation = typeinf.instance_from_kwargs(
                args.training_augmentation_class, kwargs=kwargs)
            training_augmentation = training_augmentation.to(args.device)
        else:
            logging.info("training_augmentation: None")

        # ----------------------------------------------------
        # Training Augmentation
        # ----------------------------------------------------
        if args.validation_augmentation is not None:
            kwargs = typeinf.kwargs_from_args(args, "validation_augmentation")
            logging.value("validation_augmentation: ", args.training_augmentation)
            with logging.block():
                logging.values(kwargs)
            kwargs["args"] = args
            validation_augmentation = typeinf.instance_from_kwargs(
                args.validation_augmentation_class, kwargs=kwargs)
            validation_augmentation = validation_augmentation.to(args.device)

        else:
            logging.info("validation_augmentation: None")

    return training_augmentation, validation_augmentation


def configure_model_and_loss(args):
    with logging.block("Model and Loss", emph=True):

        kwargs = typeinf.kwargs_from_args(args, "model")
        kwargs["args"] = args
        model = typeinf.instance_from_kwargs(args.model_class, kwargs=kwargs)

        loss = None
        if args.loss is not None:
            kwargs = typeinf.kwargs_from_args(args, "loss")
            kwargs["args"] = args
            loss = typeinf.instance_from_kwargs(args.loss_class, kwargs=kwargs)
        else:
            logging.info("Loss is None; you need to pick a loss!")
            quit()

        model_and_loss = facade.ModelAndLoss(args, model, loss)

        logging.value("Batch Size: ", args.batch_size)
        if loss is not None:
            logging.value("Loss: ", args.loss)
        logging.value("Network: ", args.model)
        logging.value("Number of parameters: ", model_and_loss.num_parameters())
        if loss is not None:
            logging.value("Training Key: ", args.training_key)
        if args.validation_dataset is not None:
            logging.value("Validation Keys: ", args.validation_keys)
            logging.value("Validation Modes: ", args.validation_modes)

    return model_and_loss


def configure_random_seed(args):
    with logging.block("Random Seeds", emph=True):
        seed = args.seed
        if seed is not None:
            # python
            random.seed(seed)
            logging.value("Python seed: ", seed)
            # numpy
            seed += 1
            np.random.seed(seed)
            logging.value("Numpy seed: ", seed)
            # torch
            seed += 1
            torch.manual_seed(seed)
            logging.value("Torch CPU seed: ", seed)
            # torch cuda
            seed += 1
            torch.cuda.manual_seed(seed)
            logging.value("Torch CUDA seed: ", seed)
        else:
            logging.info("None")


def configure_checkpoint_saver(args, model_and_loss):
    with logging.block('Checkpoint', emph=True):
        checkpoint_saver = checkpoints.CheckpointSaver()
        checkpoint_stats = None

        if args.checkpoint is None:
            logging.info('No checkpoint given.')
            logging.info('Starting from scratch with random initialization.')

        elif os.path.isfile(args.checkpoint):
            checkpoint_stats, filename = checkpoint_saver.restore(
                filename=args.checkpoint,
                model_and_loss=model_and_loss,
                include_params=args.checkpoint_include_params,
                exclude_params=args.checkpoint_exclude_params,
                translations=args.checkpoint_translations,
                fuzzy_translation_keys=args.checkpoint_fuzzy_translation_keys)

        elif os.path.isdir(args.checkpoint):
            if args.checkpoint_mode == 'best':
                logging.info('Loading best checkpoint in %s' % args.checkpoint)
                checkpoint_stats, filename = checkpoint_saver.restore_best(
                    directory=args.checkpoint,
                    model_and_loss=model_and_loss,
                    include_params=args.checkpoint_include_params,
                    exclude_params=args.checkpoint_exclude_params,
                    translations=args.checkpoint_translations,
                    fuzzy_translation_keys=args.checkpoint_fuzzy_translation_keys)

            elif args.checkpoint_mode == 'latest':
                logging.info('Loading latest checkpoint in %s' % args.checkpoint)
                checkpoint_stats, filename = checkpoint_saver.restore_latest(
                    directory=args.checkpoint,
                    model_and_loss=model_and_loss,
                    include_params=args.checkpoint_include_params,
                    exclude_params=args.checkpoint_exclude_params,
                    translations=args.checkpoint_translations,
                    fuzzy_translation_keys=args.checkpoint_fuzzy_translation_keys)
            else:
                logging.info('Unknown checkpoint_restore \'%s\' given!' % args.checkpoint_restore)
                quit()
        else:
            logging.info('Could not find checkpoint file or directory \'%s\'' % args.checkpoint)
            quit()

    return checkpoint_saver, checkpoint_stats


def configure_proctitle(args):
    if args.proctitle == '__random__':
        args.proctitle = proctitles.get_random_title()
    proctitles.setproctitle(args.proctitle)
    return args


def configure_collation(args):
    training_collation = None
    validation_collation = None
    try:
        getattr(args, 'training_dataset' + '_' + 'num_samples_per_example')
        training_collation = facade.CollateBatchesAndSamples(args)
    except Exception:
        pass
    try:
        getattr(args, 'validation_dataset' + '_' + 'num_samples_per_example')
        validation_collation = facade.CollateBatchesAndSamples(args)
    except Exception:
        pass

    return training_collation, validation_collation


# -------------------------------------------------------------------------------------------------
# Configure data loading
# -------------------------------------------------------------------------------------------------
def configure_data_loaders(args):
    with logging.block("Datasets", emph=True):

        def _sizes_to_str(value):
            if np.isscalar(value):
                return '1L'
            else:
                sizes = str([d for d in value.size()])
                return ' '.join([strings.replace_index(sizes, 1, '#')])

        def _log_statistics(loader, dataset):
            example_dict = loader.first_item()  # get sizes from first dataset example
            for key, value in sorted(example_dict.items()):
                if key == "index" or "name" in key:  # no need to display these
                    continue
                if isinstance(value, str):
                    logging.value("%s: " % key, value)
                elif isinstance(value, list) or isinstance(value, tuple):
                    logging.value("%s: " % key, _sizes_to_str(value[0]))
                else:
                    logging.value("%s: " % key, _sizes_to_str(value))
            logging.value("num_examples: ", len(dataset))

        # -----------------------------------------------------------------------------------------
        # GPU parameters
        # -----------------------------------------------------------------------------------------
        gpuargs = {"pin_memory": constants.DATALOADER_PIN_MEMORY} if args.cuda else {}

        train_loader_and_collation = None
        validation_loader_and_collation = None

        # -----------------------------------------------------------------
        # This figures out from the args alone, whether we need batch collcation
        # -----------------------------------------------------------------
        train_collation, validation_collation = configure_collation(args)

        # -----------------------------------------------------------------------------------------
        # Training dataset
        # -----------------------------------------------------------------------------------------
        if args.training_dataset is not None:
            # ----------------------------------------------
            # Figure out training_dataset arguments
            # ----------------------------------------------
            kwargs = typeinf.kwargs_from_args(args, "training_dataset")
            kwargs["args"] = args

            # ----------------------------------------------
            # Create training dataset and loader
            # ----------------------------------------------
            logging.value("Training Dataset: ", args.training_dataset)
            with logging.block():
                train_dataset = typeinf.instance_from_kwargs(
                    args.training_dataset_class, kwargs=kwargs)
                if args.batch_size > len(train_dataset):
                    logging.info("Problem: batch_size bigger than number of training dataset examples!")
                    quit()
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=args.batch_size,
                    shuffle=constants.TRAINING_DATALOADER_SHUFFLE,
                    drop_last=constants.TRAINING_DATALOADER_DROP_LAST,
                    num_workers=args.training_dataset_num_workers,
                    **gpuargs)
                train_loader_and_collation = facade.LoaderAndCollation(
                    args, loader=train_loader, collation=train_collation)
                _log_statistics(train_loader_and_collation, train_dataset)

        # -----------------------------------------------------------------------------------------
        # Validation dataset
        # -----------------------------------------------------------------------------------------
        if args.validation_dataset is not None:
            # ----------------------------------------------
            # Figure out validation_dataset arguments
            # ----------------------------------------------
            kwargs = typeinf.kwargs_from_args(args, "validation_dataset")
            kwargs["args"] = args

            # ------------------------------------------------------
            # per default batch_size is the same as for training,
            # unless a validation_batch_size is specified.
            # -----------------------------------------------------
            validation_batch_size = args.batch_size
            if args.validation_batch_size > 0:
                validation_batch_size = args.validation_batch_size

            # ----------------------------------------------
            # Create validation dataset and loader
            # ----------------------------------------------
            logging.value("Validation Dataset: ", args.validation_dataset)
            with logging.block():
                validation_dataset = typeinf.instance_from_kwargs(
                    args.validation_dataset_class, kwargs=kwargs)
                if validation_batch_size > len(validation_dataset):
                    logging.info("Problem: validation_batch_size bigger than number of validation dataset examples!")
                    quit()
                validation_loader = DataLoader(
                    validation_dataset,
                    batch_size=validation_batch_size,
                    shuffle=constants.VALIDATION_DATALOADER_SHUFFLE,
                    drop_last=constants.VALIDATION_DATALOADER_DROP_LAST,
                    num_workers=args.validation_dataset_num_workers,
                    **gpuargs)
                validation_loader_and_collation = facade.LoaderAndCollation(
                    args, loader=validation_loader, collation=validation_collation)
                _log_statistics(validation_loader_and_collation, validation_dataset)

    return train_loader_and_collation, validation_loader_and_collation


# ------------------------------------------------------------
# Generator for trainable parameters by pattern matching
# ------------------------------------------------------------
def _generate_trainable_params(model_and_loss, match="*"):
    for name, p in model_and_loss.named_parameters():
        if fnmatch.fnmatch(name, match):
            if p.requires_grad:
                yield p


def _param_names_and_trainable_generator(model_and_loss, match="*"):
    names = []
    for name, p in model_and_loss.named_parameters():
        if fnmatch.fnmatch(name, match):
            if p.requires_grad:
                names.append(name)

    return names, _generate_trainable_params(model_and_loss, match=match)


# -------------------------------------------------------------------------------------------------
# Build optimizer:
# -------------------------------------------------------------------------------------------------
def configure_optimizer(args, model_and_loss):
    optimizer = None
    with logging.block("Optimizer", emph=True):
        logging.value(
            "Algorithm: ",
            args.optimizer if args.optimizer is not None else "None")
        if args.optimizer is not None:
            if model_and_loss.num_parameters() == 0:
                logging.info("No trainable parameters detected.")
                logging.info("Setting optimizer to None.")
            else:
                with logging.block():
                    # -------------------------------------------
                    # Figure out all optimizer arguments
                    # -------------------------------------------
                    all_kwargs = typeinf.kwargs_from_args(args, "optimizer")

                    # -------------------------------------------
                    # Get the split of param groups
                    # -------------------------------------------
                    kwargs_without_groups = {
                        key: value for key, value in all_kwargs.items() if key != "group"
                    }
                    param_groups = all_kwargs["group"]

                    # ----------------------------------------------------------------------
                    # Print arguments (without groups)
                    # ----------------------------------------------------------------------
                    logging.values(kwargs_without_groups)

                    # ----------------------------------------------------------------------
                    # Construct actual optimizer params
                    # ----------------------------------------------------------------------
                    kwargs = dict(kwargs_without_groups)
                    if param_groups is None:
                        # ---------------------------------------------------------
                        # Add all trainable parameters if there is no param groups
                        # ---------------------------------------------------------
                        all_trainable_parameters = _generate_trainable_params(model_and_loss)
                        kwargs["params"] = all_trainable_parameters
                    else:
                        # -------------------------------------------
                        # Add list of parameter groups instead
                        # -------------------------------------------
                        trainable_parameter_groups = []
                        dnames, dparams = _param_names_and_trainable_generator(model_and_loss)
                        dnames = set(dnames)
                        dparams = set(list(dparams))
                        with logging.block("parameter_groups:"):
                            for group in param_groups:
                                #  log group settings
                                group_match = group["params"]
                                group_args = {
                                    key: value for key, value in group.items() if key != "params"
                                }

                                with logging.block("%s: %s" % (group_match, group_args)):
                                    # retrieve parameters by matching name
                                    gnames, gparams = _param_names_and_trainable_generator(
                                        model_and_loss, match=group_match)
                                    # log all names affected
                                    for n in sorted(gnames):
                                        logging.info(n)
                                    # set generator for group
                                    group_args["params"] = gparams
                                    # append parameter group
                                    trainable_parameter_groups.append(group_args)
                                    # update remaining trainable parameters
                                    dnames -= set(gnames)
                                    dparams -= set(list(gparams))

                            # append default parameter group
                            trainable_parameter_groups.append({"params": list(dparams)})
                            # and log its parameter names
                            with logging.block("default:"):
                                for dname in sorted(dnames):
                                    logging.info(dname)

                        # set params in optimizer kwargs
                        kwargs["params"] = trainable_parameter_groups

                    # -------------------------------------------
                    # Create optimizer instance
                    # -------------------------------------------
                    optimizer = typeinf.instance_from_kwargs(args.optimizer_class, kwargs=kwargs)

    return optimizer


# -------------------------------------------------------------------------------------------------
# Configure learning rate scheduler
# -------------------------------------------------------------------------------------------------
def configure_lr_scheduler(args, optimizer):
    with logging.block("Learning Rate Scheduler", emph=True):
        logging.value(
            "Scheduler: ",
            args.lr_scheduler if args.lr_scheduler is not None else "None")
        lr_scheduler = None
        if args.lr_scheduler is not None:
            kwargs = typeinf.kwargs_from_args(args, "lr_scheduler")
            with logging.block():
                logging.values(kwargs)
            kwargs["optimizer"] = optimizer
            lr_scheduler = typeinf.instance_from_kwargs(args.lr_scheduler_class, kwargs=kwargs)
    return lr_scheduler


class _ReplaceByLearningRate(optim.Optimizer):
    def __init__(self, params, lr):
        defaults = dict(lr=lr)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                p.data.fill_(group['lr'])
        return loss


class _SchedulerOptimizerPair(object):
    def __init__(self, scheduler, optimizer):
        self.scheduler = scheduler
        self.optimizer = optimizer

    def step(self, epoch=None):
        self.optimizer.step()
        self.scheduler.step(epoch=epoch)


def _configure_parameter_scheduler_group(kwargs):
    # get parameter group parameters
    init = kwargs["init"]
    params = kwargs["params"]
    scheduler = kwargs["scheduler"]

    # key word dictionary for this group
    kwargs = {key: value for key, value in kwargs.items() if key not in ["params", "scheduler", "init"]}

    # create optimizer for this group
    optimizer = _ReplaceByLearningRate(params, lr=init)
    kwargs["optimizer"] = optimizer

    # create scheduler for this group
    scheduler_class = typeinf.module_classes_to_dict(torch.optim.lr_scheduler)[scheduler]
    scheduler = typeinf.instance_from_kwargs_with_forced_types(
        scheduler_class, kwargs=kwargs,
        forced_default_types={"min_lr": float, "eta_min": float},
        unknown_default_types={"T_max": int,
                               "lr_lambda": str,
                               "step_size": int,
                               "milestones": strings.as_intlist_or_none,
                               "gamma": float})

    # create a pair of scheduler/optimizer that updates this group on step()
    scheduler = _SchedulerOptimizerPair(scheduler, optimizer)
    return scheduler


def configure_parameter_scheduler(args, model_and_loss):
    param_groups = args.param_scheduler_group
    with logging.block("Parameter Scheduler", emph=True):
        if param_groups is None:
            logging.info("None")
        else:
            logging.value("Info: ", "Please set lr=0 for scheduled parameters!")
            scheduled_parameter_groups = []
            with logging.block("parameter_groups:"):
                for group_kwargs in param_groups:
                    group_match = group_kwargs["params"]
                    group_args = {
                        key: value for key, value in group_kwargs.items() if key != "params"
                    }
                    with logging.block("%s: %s" % (group_match, group_args)):
                        gnames, gparams = _param_names_and_trainable_generator(
                            model_and_loss, match=group_match)
                        for n in sorted(gnames):
                            logging.info(n)
                        group_args['params'] = gparams
                        scheduled_parameter_groups.append(group_args)

            # create schedulers for every parameter group
            schedulers = [_configure_parameter_scheduler_group(kwargs) for kwargs in scheduled_parameter_groups]

            # create container of parameter schedulers
            scheduler = facade.ParameterSchedulerContainer(schedulers)
            return scheduler

    return None


# ----------------------------------------------------------
# Construct a runtime visualizer
# ----------------------------------------------------------
def configure_visualizers(args,
                          model_and_loss,
                          optimizer,
                          param_scheduler,
                          lr_scheduler,
                          train_loader,
                          validation_loader):
    with logging.block("Runtime Visualizers", emph=True):
        logging.value(
            "Visualizer: ",
            args.visualizer if args.visualizer is not None else "None")
        visualizer = None
        if args.visualizer is not None:
            kwargs = typeinf.kwargs_from_args(args, "visualizer")
            logging.values(kwargs)
            kwargs["args"] = args
            kwargs["model_and_loss"] = model_and_loss
            kwargs["optimizer"] = optimizer
            kwargs["param_scheduler"] = param_scheduler
            kwargs["lr_scheduler"] = lr_scheduler
            kwargs["train_loader"] = train_loader
            kwargs["validation_loader"] = validation_loader
            visualizer = typeinf.instance_from_kwargs(
                args.visualizer_class, kwargs=kwargs)
    return visualizer
