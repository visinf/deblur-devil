# Author: Jochen Gast <jochen.gast@visinf.tu-darmstadt.de>

import collections
import logging
import warnings

import numpy as np
import torch

import constants
import logger
from holistic_records.recorder import EpochRecorder
from utils import proctitles
from utils import summary
from utils import system
from utils import timing
from utils.moving_averages import MovingAverage


# from utils.timing import tic, toc

# ---------------------------------------------------------------
# Progressbar for sequences of type 'iterable'
# ---------------------------------------------------------------
def create_progressbar(iterable,
                       desc='',
                       train=False,
                       unit="it",
                       initial=0,
                       offset=0,
                       invert_iterations=False,
                       logging_on_update=False,
                       logging_on_close=True,
                       track_eta=False):
    # ---------------------------------------------------------------
    # Pick colors
    # ---------------------------------------------------------------
    reset = constants.COLOR_RESET
    block_arrow_col = constants.COLOR_BLOCK_ARROW
    emph_col = constants.COLOR_BLOCK_EMPH
    # bar_col = constants.COLOR_PROGRESS_BAR
    stats_col = constants.COLOR_PROGRESS_STATS

    # ---------------------------------------------------------------
    # Specify progressbar layout:
    #   l_bar, bar, r_bar, n, n_fmt, total, total_fmt, percentage,
    #   rate, rate_fmt, rate_noinv, rate_noinv_fmt, rate_inv,
    #   rate_inv_fmt, elapsed, remaining, desc, postfix.
    # ---------------------------------------------------------------
    bar_format = ""
    bar_format += "%s==>%s%s {desc}:%s" % (block_arrow_col, reset, emph_col, reset)
    bar_format += " {percentage:3.0f}%"  # percentage
    # bar_format += "%s|{bar}|%s " % (bar_col, reset)  # bar
    # bar_format += " %s|%s " % (dim, reset)  # bar
    bar_format += " {n_fmt:>3}/{total_fmt:<3}"  # i/n counter
    bar_format += " {elapsed}<{remaining}"  # eta
    if invert_iterations:
        bar_format += " {rate_inv_fmt}"  # iteration timings
    else:
        bar_format += " {rate_noinv_fmt}"
    bar_format += "  %s{postfix}%s" % (stats_col, reset)  # postfix

    # ---------------------------------------------------------------
    # Specify TQDM arguments
    # ---------------------------------------------------------------
    tqdm_args = {
        "iterable": iterable,
        "desc": desc,  # Prefix for the progress bar
        "total": len(iterable),  # The number of expected iterations
        "leave": True,  # Leave progress bar when done
        "miniters": 1 if train else None,  # Min update interval (iterations)
        "unit": unit,  # String be used to define the unit of each iteration
        "initial": initial,  # The initial counter value.
        "ncols": constants.TQDM_NCOLS,  # The width of the entire output message
        "dynamic_ncols": False,  # Allow window resizes
        "smoothing": constants.TQDM_SMOOTHING,  # Smoothing factor for speed estimates
        "bar_format": bar_format,  # Specify a custom bar string formatting
        "position": offset,  # Specify vertical line offset
        "ascii": True,
        "logging_on_update": logging_on_update,
        "logging_on_close": logging_on_close,
        "track_eta": track_eta
    }

    return logger.tqdm_with_logging(**tqdm_args)


def format_moving_averages_as_progress_dict(moving_averages_dict,
                                            moving_averages_postfix="avg",
                                            timing_dict=None):
    values = [
        (key + moving_averages_postfix, "%1.4f" % moving_averages_dict[key].mean())
        for key in sorted(moving_averages_dict.keys())
    ]
    progress_dict = collections.OrderedDict(values)
    progress_dict.update(timing_dict)
    return progress_dict


def format_learning_rate(lr):
    if np.isscalar(lr):
        return "{}".format(lr)
    else:
        return "{}".format(str(lr[0]) if len(lr) == 1 else lr)


def format_telegram_status_update(args, epoch, eta_str='', total_progress_stats=None):
    if epoch == 0:
        result = "{}:{} on {} @{}/{}\n".format(
            args.prefix, args.model, args.actual_device, 0, args.total_epochs)
        for key, value in sorted(args.non_default_args.items()):
            result += "  {}: {}\n".format(key, value)
        return result[0:-1]

    else:
        result = "{}:{} on {} [{}/{}<{}]: ".format(
            args.prefix, args.model, args.actual_device, epoch, args.total_epochs, eta_str)
        if total_progress_stats is not None:
            result += str(total_progress_stats)[1:-1].replace('\'', '')
        return result


def format_telegram_throw_nan(args):
    result = "{}:{} on {}:\n  loss is NaN\n".format(args.prefix, args.model, args.actual_device)
    return result


def format_epoch_header_stats(args, lr):
    return "Model: {}  lr: {}".format(args.model, format_learning_rate(lr))


def format_epoch_header_machine_stats(args):
    return "{}{}-- {} on {} --{}{}".format(
        constants.COLOR_RESET,
        constants.COLOR_TIMESTAMP,
        args.actual_device,
        system.hostname(),
        system.screen_identifier(),
        constants.COLOR_RESET)


def configure_holistic_epoch_recorder(args, epoch, loader):
    epoch_recorder = EpochRecorder(
        args,
        epoch=epoch,
        dataset=loader.dataset.__class__.__name__,
        **args.holistic_records_kwargs)
    return epoch_recorder


class RuntimeEpoch:
    def __init__(self,
                 args,
                 model_and_loss,
                 loader,
                 augmentation=None,
                 optimizer=None,
                 recorder=None,
                 visualizer=None,
                 add_progress_stats=(),
                 desc="Epoch"):

        self.args = args
        self.desc = desc
        self.add_progress_stats = dict(add_progress_stats)
        self.augmentation = augmentation
        self.loader = loader
        self.model_and_loss = model_and_loss
        self.optimizer = optimizer
        self.recorder = recorder
        self.visualizer = visualizer

    def step(self, example_dict, train):

        # -------------------------------------------------------------
        # Optionally perform augmentations
        # -------------------------------------------------------------
        if self.augmentation is not None:
            with torch.no_grad():
                example_dict = self.augmentation(example_dict)

        # -------------------------------------------------------------
        # Extract batch size from first input
        # -------------------------------------------------------------
        batch_size = example_dict["input1"].size(0)

        # -----------------------------------------------------------------
        # Training Epoch
        # -----------------------------------------------------------------
        if train:

            # -------------------------------------------------------------
            # Before the backward pass, use the optimizer object to zero
            # all of the gradients for the variables it will update (which
            # are the learnable weights of the model). This is because by
            # default, gradients are accumulated in buffers( i.e, not
            # overwritten) whenever .backward() is called. Checkout docs
            # of torch.autograd.backward for more details.
            # -------------------------------------------------------------
            self.optimizer.zero_grad()

            # -------------------------------------------------------------
            # Run forward pass to get losses and outputs.
            # -------------------------------------------------------------
            loss_dict, model_dict = self.model_and_loss(example_dict)

            # -------------------------------------------------------------
            # Check training_key for for NaNs, key is usually "total_loss"
            # -------------------------------------------------------------
            loss = loss_dict[self.args.training_key]
            is_loss_nan = np.isnan(loss.item())
            if is_loss_nan:
                logging.telegram(format_telegram_throw_nan(self.args))
                assert (not is_loss_nan), "training loss is NaN"

            # -------------------------------------------------------------
            # Backward pass: compute gradient of the loss with respect to
            #                model parameters
            # -------------------------------------------------------------
            loss.backward()

            # -------------------------------------------------------------
            # Optional: Apply gradient clipping
            # -------------------------------------------------------------
            if self.args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model_and_loss.parameters(), self.args.clip_grad)

            # -------------------------------------------------------------
            # Optional: Use Loshchilov weight decay.
            #           Note that weight decay is supposed to be 0
            #           in this case.
            # -------------------------------------------------------------
            if self.args.loshchilov_weight_decay > 0:
                if self.args.optimizer_weight_decay > 0:
                    msg = "{} {}".format(
                        "Detected L2 weight decay when using Loshchilov decay.",
                        "Please set weight decay to 0!")
                    warnings.warn(msg, UserWarning)
                with torch.no_grad():
                    for group in self.optimizer.param_groups:
                        wr = self.args.loshchilov_weight_decay * group["lr"]
                        for param in group["params"]:
                            param.data = param.data.add(- wr, param.data)

            # -------------------------------------------------------------
            # Calling the step function on an Optimizer makes an update to
            # its parameters
            # -------------------------------------------------------------
            self.optimizer.step()

        # -----------------------------------------------------------------
        # Validation epoch
        # -----------------------------------------------------------------
        else:

            # -------------------------------------------------------------
            # Just run forward pass to get losses and outputs. Done.
            # -------------------------------------------------------------
            loss_dict, model_dict = self.model_and_loss(example_dict)

        # -------------------------------------------------------------
        # Return success flag, loss and output dictionary
        # -------------------------------------------------------------
        return loss_dict, model_dict, batch_size

    def run(self, train):
        # ---------------------------------------
        # Tell model when we want to train
        # ---------------------------------------
        if train:
            self.model_and_loss.train()
        else:
            self.model_and_loss.eval()

        # ---------------------------------------
        # Keep track of moving averages
        # ---------------------------------------
        moving_averages_dict = None

        # ---------------------------------------
        # Progress bar arguments
        # ---------------------------------------
        progressbar_args = {
            "iterable": self.loader,
            "desc": self.desc,
            "train": train,
            "offset": 0,
            "logging_on_update": False,
            "logging_on_close": True
        }

        # ---------------------------------------
        # Perform training/evaluation steps
        # ---------------------------------------
        with create_progressbar(**progressbar_args) as progress:
            total_steps = len(progress)
            for k, example_dict in enumerate(progress):
                # ---------------------------------------
                # possibly forward results to visualizer
                # ---------------------------------------
                if self.visualizer is not None:
                    self.visualizer.on_step_init(
                        example_dict, train=train, step=k, total_steps=total_steps)

                # ---------------------------------------
                # Perform training/evaluation step
                # ---------------------------------------
                loss_dict_per_step, model_dict, batch_size = self.step(example_dict, train=train)

                # ---------------------------------------
                # possibly forward results to visualizer
                # ---------------------------------------
                if self.visualizer is not None:
                    self.visualizer.on_step_finished(
                        example_dict, model_dict, loss_dict_per_step,
                        train=train, step=k, total_steps=total_steps)

                # --------------------------------------------------------
                # Possibly initialize moving averages
                # --------------------------------------------------------
                if moving_averages_dict is None:
                    moving_averages_dict = {
                        key: MovingAverage() for key in loss_dict_per_step.keys()
                    }

                # --------------------------------------------------------
                # Add moving averages
                # --------------------------------------------------------
                for key, loss in loss_dict_per_step.items():
                    moving_averages_dict[key].add_average(loss.item(), addcount=batch_size)

                # --------------------------------------------------------
                # Stop timing and accumulate results
                # --------------------------------------------------------
                timing_dict = timing.stats()

                # --------------------------------------------------------
                # View statistics in progress bar
                # --------------------------------------------------------
                postfix = "_ema" if train else "_avg"
                progress_stats = format_moving_averages_as_progress_dict(
                    moving_averages_dict=moving_averages_dict,
                    moving_averages_postfix=postfix,
                    timing_dict=timing_dict)
                progress.set_postfix(progress_stats)

        # -------------------------------------------------------------
        # Return moving average dictionary
        # -------------------------------------------------------------
        ma_dict = {key: ma.mean() for key, ma in moving_averages_dict.items()}
        return ma_dict


def exec_runtime(args,
                 checkpoint_saver,
                 model_and_loss,
                 optimizer,
                 lr_scheduler,
                 param_scheduler,
                 train_loader,
                 validation_loader,
                 training_augmentation,
                 validation_augmentation,
                 visualizer):
    # --------------------------------------------------------------------------------
    # Validation schedulers are a bit special:
    # They need special treatment as they want to be called with a validation loss..
    # --------------------------------------------------------------------------------
    validation_scheduler = (lr_scheduler is not None and args.lr_scheduler == "ReduceLROnPlateau")

    # --------------------------------------------------------
    # Log some runtime info
    # --------------------------------------------------------
    with logging.block("Runtime", emph=True):
        logging.value("start_epoch: ", args.start_epoch)
        logging.value("total_epochs: ", args.total_epochs)

    # ---------------------------------------
    # Total progress bar arguments
    # ---------------------------------------
    progressbar_args = {
        "desc": "Total",
        "initial": args.start_epoch - 1,
        "invert_iterations": True,
        "iterable": range(1, args.total_epochs + 1),
        "logging_on_close": True,
        "logging_on_update": True,
        "unit": "ep",
        "track_eta": True
    }

    # --------------------------------------------------------
    # Total progress bar
    # --------------------------------------------------------
    print(''), logging.logbook('')
    total_progress = create_progressbar(**progressbar_args)
    total_progress_stats = {}
    print("\n")

    # -------------------------------------------------k-------
    # Remember validation losses
    # --------------------------------------------------------
    best_validation_losses = None
    store_as_best = None
    if validation_loader is not None:
        num_validation_losses = len(args.validation_keys)
        best_validation_losses = [
            float("inf") if args.validation_modes[i] == 'min' else -float("inf")
            for i in range(num_validation_losses)
        ]
        store_as_best = [False for _ in range(num_validation_losses)]

    # ----------------------------------------------------------------
    # Send Telegram message
    # ----------------------------------------------------------------
    logging.telegram(format_telegram_status_update(args, epoch=0))

    avg_loss_dict = {}
    for epoch in range(args.start_epoch, args.total_epochs + 1):

        # --------------------------------
        # Make Epoch %i/%i header message
        # --------------------------------
        epoch_header = "Epoch {}/{}{}{}".format(
            epoch, args.total_epochs, " " * 24,
            format_epoch_header_machine_stats(args))

        with logger.LoggingBlock(epoch_header, emph=True):

            # -------------------------------------------------------------------------------
            # Let TensorBoard know where we are..
            # -------------------------------------------------------------------------------
            summary.set_global_step(epoch)

            # -----------------------------------------------------------------
            # Update standard learning scheduler and get current learning rate
            # -----------------------------------------------------------------
            #  Starting with PyTorch 1.1 the expected validation order is:
            #       optimize(...)
            #       validate(...)
            #       scheduler.step()..

            # ---------------------------------------------------------------------
            # Update parameter schedule before the epoch
            # Note: Parameter schedulers are tuples of (optimizer, schedule)
            # ---------------------------------------------------------------------
            if param_scheduler is not None:
                param_scheduler.step(epoch=epoch)

            # -----------------------------------------------------------------
            # Get current learning rate from either optimizer or scheduler
            # -----------------------------------------------------------------
            lr = args.optimizer_lr if args.optimizer is not None else "None"
            if lr_scheduler is not None:
                lr = [group['lr'] for group in optimizer.param_groups] \
                    if args.optimizer is not None else "None"

            # --------------------------------------------------------
            # Current Epoch header stats
            # --------------------------------------------------------
            logging.info(format_epoch_header_stats(args, lr))

            # -------------------------------------------
            # Create and run a training epoch
            # -------------------------------------------
            if train_loader is not None:
                if visualizer is not None:
                    visualizer.on_epoch_init(lr, train=True, epoch=epoch, total_epochs=args.total_epochs)

                ema_loss_dict = RuntimeEpoch(
                    args,
                    desc="Train",
                    augmentation=training_augmentation,
                    loader=train_loader,
                    model_and_loss=model_and_loss,
                    optimizer=optimizer,
                    visualizer=visualizer).run(train=True)

                if visualizer is not None:
                    visualizer.on_epoch_finished(
                        ema_loss_dict, train=True, epoch=epoch, total_epochs=args.total_epochs)

            # -------------------------------------------
            # Create and run a validation epoch
            # -------------------------------------------
            if validation_loader is not None:
                if visualizer is not None:
                    visualizer.on_epoch_init(
                        lr, train=False, epoch=epoch, total_epochs=args.total_epochs)

                # ---------------------------------------------------
                # Construct holistic recorder for epoch
                # ---------------------------------------------------
                epoch_recorder = configure_holistic_epoch_recorder(
                    args, epoch=epoch, loader=validation_loader)

                with torch.no_grad():
                    avg_loss_dict = RuntimeEpoch(
                        args,
                        desc="Valid",
                        augmentation=validation_augmentation,
                        loader=validation_loader,
                        model_and_loss=model_and_loss,
                        recorder=epoch_recorder,
                        visualizer=visualizer).run(train=False)

                    try:
                        epoch_recorder.add_scalars("evaluation_losses", avg_loss_dict)
                    except Exception:
                        pass

                    if visualizer is not None:
                        visualizer.on_epoch_finished(
                            avg_loss_dict, train=False,
                            epoch=epoch, total_epochs=args.total_epochs)

                # ----------------------------------------------------------------
                # Evaluate valdiation losses
                # ----------------------------------------------------------------
                validation_losses = [avg_loss_dict[vkey] for vkey in args.validation_keys]
                for i, (vkey, vmode) in enumerate(zip(args.validation_keys, args.validation_modes)):
                    if vmode == 'min':
                        store_as_best[i] = validation_losses[i] < best_validation_losses[i]
                    else:
                        store_as_best[i] = validation_losses[i] > best_validation_losses[i]
                    if store_as_best[i]:
                        best_validation_losses[i] = validation_losses[i]

                # ----------------------------------------------------------------
                # Update validation scheduler, if one is in place
                # We use the first key in validation keys as the relevant one
                # ----------------------------------------------------------------
                if lr_scheduler is not None:
                    if validation_scheduler:
                        lr_scheduler.step(validation_losses[0], epoch=epoch)
                    else:
                        lr_scheduler.step(epoch=epoch)

                # ----------------------------------------------------------------
                # Also show best loss on total_progress
                # ----------------------------------------------------------------
                total_progress_stats = {
                    "best_" + vkey + "_avg": "%1.4f" % best_validation_losses[i]
                    for i, vkey in enumerate(args.validation_keys)
                }
                total_progress.set_postfix(total_progress_stats)

            # ----------------------------------------------------------------
            # Bump total progress
            # ----------------------------------------------------------------
            total_progress.update()
            print('')

            # ----------------------------------------------------------------
            # Get ETA string for display in loggers
            # ----------------------------------------------------------------
            eta_str = total_progress.eta_str()

            # ----------------------------------------------------------------
            # Send Telegram status udpate
            # ----------------------------------------------------------------
            total_progress_stats['lr'] = format_learning_rate(lr)
            logging.telegram(format_telegram_status_update(
                args,
                eta_str=eta_str,
                epoch=epoch,
                total_progress_stats=total_progress_stats))

            # ----------------------------------------------------------------
            # Update ETA in progress title
            # ----------------------------------------------------------------
            eta_proctitle = "{} finishes in {}".format(args.proctitle, eta_str)
            proctitles.setproctitle(eta_proctitle)

            # ----------------------------------------------------------------
            # Store checkpoint
            # ----------------------------------------------------------------
            if checkpoint_saver is not None and validation_loader is not None:
                checkpoint_saver.save_latest(
                    directory=args.save,
                    model_and_loss=model_and_loss,
                    stats_dict=dict(avg_loss_dict, epoch=epoch),
                    store_as_best=store_as_best,
                    store_prefixes=args.validation_keys)

            # ----------------------------------------------------------------
            # Vertical space between epochs
            # ----------------------------------------------------------------
            print(''), logging.logbook('')

    # ----------------------------------------------------------------
    # Finish up
    # ----------------------------------------------------------------
    logging.telegram_flush()
    total_progress.close()
    logging.info("Finished.")
