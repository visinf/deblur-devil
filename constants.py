# Author: Jochen Gast <jochen.gast@visinf.tu-darmstadt.de>

import os

import colorama

# ---------------
# Logging colors
# ---------------
COLOR_BLOCK_ARROW = colorama.Fore.CYAN
COLOR_BLOCK_EMPH = colorama.Style.BRIGHT
COLOR_KEY_VALUE = colorama.Fore.CYAN
COLOR_NON_DEFAULT_ARGUMENT = colorama.Fore.CYAN
COLOR_PROGRESS_BAR = colorama.Style.DIM
COLOR_PROGRESS_STATS = colorama.Fore.GREEN
COLOR_RESET = colorama.Style.RESET_ALL
COLOR_TIMESTAMP = colorama.Style.DIM + colorama.Fore.WHITE

# ------------------
# Help format colors
# ------------------
COLOR_HELP_ARG = colorama.Style.DIM + colorama.Fore.WHITE
COLOR_HELP_ARG_VALUE = colorama.Fore.GREEN
COLOR_HELP_ARG_OPTION = colorama.Fore.CYAN

# -----------------
# Logging settings
# -----------------
LOGGING_GLOBAL_INDENT = 2  # indent between sections
LOGGING_LOGBOOK_FILENAME = "logbook.txt"
LOGGING_TELEGRAM_MACHINES_FILENAME = os.path.join(os.environ['HOME'], ".machines.json")
LOGGING_TIMEZONE = "Europe/Berlin"
LOGGING_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
LOGGING_ZIPSOURCE_FILENAME = "src.zip"

# ----------------
# Flush intervals
# ----------------
TELEGRAM_FLUSH_SECS = 600
TENSORBOARD_FLUSH_SECS = 600

# -----------------------
# Progressbar parameters
# -----------------------
TQDM_NCOLS = 120  # The width of the entire output message
TQDM_SMOOTHING = 0  # Smoothing speed estimates (0.0 = avg -> 1.0 = current speed)

# --------------------
# Dataloader defaults
# --------------------
DATALOADER_PIN_MEMORY = True
TRAINING_DATALOADER_SHUFFLE = True
TRAINING_DATALOADER_DROP_LAST = True
VALIDATION_DATALOADER_SHUFFLE = False
VALIDATION_DATALOADER_DROP_LAST = False

# ------------------
# CUDNN OPTIMIZATION
# ------------------
CUDNN_BENCHMARK = True
