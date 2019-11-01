# Author: Jochen Gast <jochen.gast@visinf.tu-darmstadt.de>

import logging
import os

import torch

import augmentations
import commandline
import configuration as config
import constants
import datasets
import logger
import losses
import models
import optim
import runtime
import visualizers
from utils import colored_traceback
from utils import system
from utils import zipsource


def main():
    # ---------------------------------------------------
    # Set working directory to folder containing main.py
    # ---------------------------------------------------
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    # ----------------------------------------------------------------
    # Activate syntax highlighting in tracebacks for better debugging
    # ----------------------------------------------------------------
    colored_traceback.add_hook()

    # -----------------------------------------------------------
    # Configure logging
    # -----------------------------------------------------------
    logging_filename = os.path.join(
        commandline.parse_save_dir(), constants.LOGGING_LOGBOOK_FILENAME)
    logger.configure_logging(logging_filename)

    # ----------------------------------------------------------------
    # Register type factories before parsing the commandline.
    # NOTE: We decided to explicitly call these init() functions, to
    #       have more precise control over the timeline
    # ----------------------------------------------------------------
    with logging.block("Registering factories", emph=True):
        augmentations.init()
        datasets.init()
        losses.init()
        models.init()
        optim.init()
        visualizers.init()
        logging.info('Done!')

    # -----------------------------------------------------------
    # Parse commandline after factories have been filled
    # -----------------------------------------------------------
    args = commandline.parse_arguments(blocktitle="Commandline Arguments")

    # -----------------------
    # Telegram configuration
    # -----------------------
    with logging.block("Telegram", emph=True):
        logger.configure_telegram(constants.LOGGING_TELEGRAM_MACHINES_FILENAME)

    # ----------------------------------------------------------------------
    # Log git repository hash and make a compressed copy of the source code
    # ----------------------------------------------------------------------
    with logging.block("Source Code", emph=True):
        logging.value("Git Hash: ", system.git_hash())
        # Zip source code and copy to save folder
        filename = os.path.join(args.save, constants.LOGGING_ZIPSOURCE_FILENAME)
        zipsource.create_zip(filename=filename, directory=os.getcwd())
        logging.value("Archieved code: ", filename)

    # ----------------------------------------------------
    # Change process title for `top` and `pkill` commands
    # This is more "informative" in `nvidia-smi` ;-)
    # ----------------------------------------------------
    args = config.configure_proctitle(args)

    # -------------------------------------------------
    # Set random seed for python, numpy, torch, cuda..
    # -------------------------------------------------
    config.configure_random_seed(args)

    # -----------------------------------------------------------
    # Machine stats
    # -----------------------------------------------------------
    with logging.block("Machine Statistics", emph=True):
        if args.cuda:
            args.device = torch.device("cuda:0")
            logging.value("Cuda: ", torch.version.cuda)
            logging.value("Cuda device count: ", torch.cuda.device_count())
            logging.value("Cuda device name: ", torch.cuda.get_device_name(0))
            logging.value("CuDNN: ", torch.backends.cudnn.version())
            device_no = 0
            if 'CUDA_VISIBLE_DEVICES' in os.environ.keys():
                device_no = os.environ['CUDA_VISIBLE_DEVICES']
            args.actual_device = "gpu:%s" % device_no
        else:
            args.device = torch.device("cpu")
            args.actual_device = "cpu"
        logging.value("Hostname: ", system.hostname())
        logging.value("PyTorch: ", torch.__version__)
        logging.value("PyTorch device: ", args.actual_device)

    # ------------------------------------------------------
    # Fetch data loaders. Quit if no data loader is present
    # ------------------------------------------------------
    train_loader, validation_loader = config.configure_data_loaders(args)

    # -------------------------------------------------------------------------
    # Check whether any dataset could be found
    # -------------------------------------------------------------------------
    success = any(loader is not None for loader in [train_loader, validation_loader])
    if not success:
        logging.info("No dataset could be loaded successfully. Please check dataset paths!")
        quit()

    # -------------------------------------------------------------------------
    # Configure runtime augmentations
    # -------------------------------------------------------------------------
    training_augmentation, validation_augmentation = config.configure_runtime_augmentations(args)

    # ----------------------------------------------------------
    # Configure model and loss.
    # ----------------------------------------------------------
    model_and_loss = config.configure_model_and_loss(args)

    # --------------------------------------------------------
    # Print model visualization
    # --------------------------------------------------------
    if args.logging_model_graph:
        with logging.block("Model Graph", emph=True):
            logger.log_module_info(model_and_loss.model)
    if args.logging_loss_graph:
        with logging.block("Loss Graph", emph=True):
            logger.log_module_info(model_and_loss.loss)

    # -------------------------------------------------------------------------
    # Possibly resume from checkpoint
    # -------------------------------------------------------------------------
    checkpoint_saver, checkpoint_stats = config.configure_checkpoint_saver(args, model_and_loss)
    if checkpoint_stats is not None:
        with logging.block():
            logging.info("Checkpoint Statistics:")
            with logging.block():
                logging.values(checkpoint_stats)
        # ---------------------------------------------------------------------
        # Set checkpoint stats
        # ---------------------------------------------------------------------
        if args.checkpoint_mode in ["resume_from_best", "resume_from_latest"]:
            args.start_epoch = checkpoint_stats["epoch"]

    # ---------------------------------------------------------------------
    # Checkpoint and save directory
    # ---------------------------------------------------------------------
    with logging.block("Save Directory", emph=True):
        if args.save is None:
            logging.info("No 'save' directory specified!")
            quit()
        logging.value("Save directory: ", args.save)
        if not os.path.exists(args.save):
            os.makedirs(args.save)

    # ------------------------------------------------------------
    # If this is just an evaluation: overwrite savers and epochs
    # ------------------------------------------------------------
    if args.training_dataset is None and args.validation_dataset is not None:
        args.start_epoch = 1
        args.total_epochs = 1
        train_loader = None
        checkpoint_saver = None
        args.optimizer = None
        args.lr_scheduler = None

    # ----------------------------------------------------
    # Tensorboard summaries
    # ----------------------------------------------------
    logger.configure_tensorboard_summaries(args.save)

    # -------------------------------------------------------------------
    # From PyTorch API:
    # If you need to move a model to GPU via .cuda(), please do so before
    # constructing optimizers for it. Parameters of a model after .cuda()
    # will be different objects with those before the call.
    # In general, you should make sure that optimized parameters live in
    # consistent locations when optimizers are constructed and used.
    # -------------------------------------------------------------------
    model_and_loss = model_and_loss.to(args.device)

    # ----------------------------------------------------------
    # Configure optimizer
    # ----------------------------------------------------------
    optimizer = config.configure_optimizer(args, model_and_loss)

    # ----------------------------------------------------------
    # Configure learning rate
    # ----------------------------------------------------------
    lr_scheduler = config.configure_lr_scheduler(args, optimizer)

    # --------------------------------------------------------------------------
    # Configure parameter scheduling
    # --------------------------------------------------------------------------
    param_scheduler = config.configure_parameter_scheduler(args, model_and_loss)

    # quit()

    # ----------------------------------------------------------
    # Cuda optimization
    # ----------------------------------------------------------
    if args.cuda:
        torch.backends.cudnn.benchmark = constants.CUDNN_BENCHMARK

    # ----------------------------------------------------------
    # Configurate runtime visualization
    # ----------------------------------------------------------
    visualizer = config.configure_visualizers(
        args,
        model_and_loss=model_and_loss,
        optimizer=optimizer,
        param_scheduler=param_scheduler,
        lr_scheduler=lr_scheduler,
        train_loader=train_loader,
        validation_loader=validation_loader)
    if visualizer is not None:
        visualizer = visualizer.to(args.device)

    # ----------------------------------------------------------
    # Kickoff training, validation and/or testing
    # ----------------------------------------------------------
    return runtime.exec_runtime(
        args,
        checkpoint_saver=checkpoint_saver,
        lr_scheduler=lr_scheduler,
        param_scheduler=param_scheduler,
        model_and_loss=model_and_loss,
        optimizer=optimizer,
        train_loader=train_loader,
        training_augmentation=training_augmentation,
        validation_augmentation=validation_augmentation,
        validation_loader=validation_loader,
        visualizer=visualizer)


if __name__ == "__main__":
    main()
